import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from datasets import load_dataset


# Model names
BIAS_MODEL = "premsa/political-bias-prediction-allsides-BERT"
GENERAL_BIAS_MODEL = "valurank/distilroberta-bias"
SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


class BiasEvaluator:
    """Evaluate bias detection model on benchmark datasets."""

    def __init__(self, model_name: str):
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Get prediction for a single text."""
        if not text or text.strip() == "":
            return {
                "label": 0,
                "confidence": 0.0,
                "probabilities": [1.0, 0.0, 0.0],
            }

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        pred_label = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_label].item()

        return {
            "label": pred_label,
            "confidence": confidence,
            "probabilities": probs[0].cpu().numpy(),
        }

    def predict_batch(self, texts: list[str]) -> list[int]:
        """Predict labels using true batching."""
        valid_texts = [t for t in texts if t and str(t).strip()]
        skipped = len(texts) - len(valid_texts)
        if skipped:
            print(f"Skipped {skipped} empty texts")

        print(f"Predicting on {len(valid_texts)} valid samples...")

        predictions = []
        batch_size = 32

        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            batch_preds = torch.argmax(probs, dim=-1).cpu().tolist()
            predictions.extend(batch_preds)

            if (i // batch_size + 1) % 10 == 0:
                print(
                    f"  Processed {i + len(batch)}/{len(valid_texts)} "
                    f"samples..."
                )

        return predictions

    def evaluate_on_dataset(
        self,
        texts: list[str],
        true_labels: list[int],
        remap_fn=None,
    ) -> dict:
        """
        Evaluate model on a dataset.

        Args:
            texts: Input texts.
            true_labels: Ground truth integer labels.
            remap_fn: Optional function to remap raw predictions
                      (e.g. 3-class → binary, or invert labels).

        Returns metrics: accuracy, precision, recall, F1.
        """
        raw_predictions = self.predict_batch(texts)

        predictions = (
            [remap_fn(p) for p in raw_predictions]
            if remap_fn
            else raw_predictions
        )

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predictions,
            average="weighted",
            zero_division=0,
        )

        cm = confusion_matrix(true_labels, predictions)
        report = classification_report(
            true_labels,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_samples": len(texts),
        }


def _load_mbib_split(
    split_name: str,
    sample_size: int | None,
    seed: int = 42,
):
    """
    Load, shuffle, and filter an MBIB split.

    IMPORTANT: shuffle before selecting to avoid class imbalance —
    MBIB splits are ordered by class so the first N samples are
    all one label without shuffling.
    """
    dataset = load_dataset("mediabiasgroup/mbib-base")

    if split_name not in dataset:
        raise ValueError(
            f"Split '{split_name}' not found. "
            f"Available: {list(dataset.keys())}"
        )

    ds = dataset[split_name]

    # ← CRITICAL: shuffle first so sample is class-balanced
    ds = ds.shuffle(seed=seed)

    if sample_size is not None:
        print(f"⚠️  TESTING MODE: Using first {sample_size} shuffled samples")
        ds = ds.select(range(min(sample_size, len(ds))))

    pairs = [
        (t, l)
        for t, l in zip(ds["text"], ds["label"])
        if t and str(t).strip()
    ]

    if not pairs:
        raise ValueError(f"No valid samples found in split '{split_name}'")

    texts, labels = zip(*pairs)

    # Log class distribution so we can verify balance
    from collections import Counter
    dist = Counter(labels)
    print(f"Class distribution: {dict(dist)}")

    return list(texts), list(labels)


# ─────────────────────────────────────────────
# 1. POLITICAL BIAS
#    premsa AllSides BERT (3-class L/C/R)
#    → MBIB political_bias (binary)
#
#    Remap per Wessel et al. (2023):
#      Left(0)   → Biased(1)
#      Center(1) → Unbiased(0)
#      Right(2)  → Biased(1)
# ─────────────────────────────────────────────
def evaluate_political_bias(sample_size=None):
    """
    Evaluate AllSides BERT on MBIB political_bias split.

    The model outputs Left/Center/Right (3-class).
    MBIB uses binary biased/unbiased labels.
    Per Wessel et al. (2023): Left and Right collapse to
    Biased(1), Center collapses to Unbiased(0).
    """
    print(
        "\nLoading MBIB political_bias split "
        "(Wessel et al., 2023)..."
    )

    texts, true_labels = _load_mbib_split("political_bias", sample_size)
    print(f"Loaded {len(texts)} valid samples")

    # Left(0) → 1, Center(1) → 0, Right(2) → 1
    def political_to_binary(pred: int) -> int:
        return 0 if pred == 1 else 1

    evaluator = BiasEvaluator(BIAS_MODEL)

    print(
        f"\n📊 Evaluating political bias model on MBIB "
        f"political_bias ({len(texts)} samples)..."
    )

    results = evaluator.evaluate_on_dataset(
        texts,
        true_labels,
        remap_fn=political_to_binary,
    )

    print("\n" + "=" * 60)
    print("POLITICAL BIAS EVALUATION (MBIB - political_bias split)")
    print("Model: premsa/political-bias-prediction-allsides-BERT")
    print("Remap: Left+Right → Biased(1), Center → Unbiased(0)")
    print("Per Wessel et al. (2023) binary collapse")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print(f"Samples:   {results['num_samples']}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("             Unbiased  Biased")
    for i, row in enumerate(results["confusion_matrix"]):
        label = ["Unbiased", "Biased  "][i] if i < 2 else str(i)
        print(f"  {label}    {row}")

    return results


# ─────────────────────────────────────────────
# 2. GENERAL / LEXICAL BIAS
#    valurank/distilroberta-bias (binary)
#    → MBIB linguistic_bias split
#
#    ⚠️ Model label scheme is INVERTED vs MBIB:
#      Model 0 = BIASED   → MBIB 1
#      Model 1 = NEUTRAL  → MBIB 0
#    Remap: 1 - pred
# ─────────────────────────────────────────────
def evaluate_general_bias(sample_size=None):
    """
    Evaluate distilroberta-bias on MBIB linguistic_bias split.

    linguistic_bias tests loaded language, word choice, and
    framing — the exact phenomena distilroberta-bias was trained
    to detect (Spinde et al., 2023; Wessel et al., 2023).

    NOTE: Model outputs 0=BIASED, 1=NEUTRAL (inverted vs MBIB).
    Remap applied: MBIB_label = 1 - model_pred.
    """
    print(
        "\nLoading MBIB linguistic_bias split "
        "(Wessel et al., 2023)..."
    )

    texts, true_labels = _load_mbib_split("linguistic_bias", sample_size)
    print(f"Loaded {len(texts)} valid samples")

    # ← INVERT: model uses 0=BIASED, 1=NEUTRAL
    #   MBIB uses 0=Unbiased, 1=Biased
    def invert_label(pred: int) -> int:
        return 1 - pred

    evaluator = BiasEvaluator(GENERAL_BIAS_MODEL)

    print(
        f"\n📊 Evaluating general bias model on MBIB "
        f"linguistic_bias ({len(texts)} samples)..."
    )

    results = evaluator.evaluate_on_dataset(
        texts,
        true_labels,
        remap_fn=invert_label,
    )

    print("\n" + "=" * 60)
    print("GENERAL BIAS EVALUATION (MBIB - linguistic_bias split)")
    print("Model: valurank/distilroberta-bias")
    print("Remap: model 0=BIASED→1, model 1=NEUTRAL→0")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print(f"Samples:   {results['num_samples']}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("             Unbiased  Biased")
    for i, row in enumerate(results["confusion_matrix"]):
        label = ["Unbiased", "Biased  "][i] if i < 2 else str(i)
        print(f"  {label}    {row}")

    return results


# ─────────────────────────────────────────────
# 3. SENTIMENT — SST-2 GLUE
# ─────────────────────────────────────────────
def evaluate_sentiment():
    """Evaluate sentiment model on SST-2 benchmark."""
    print("\nLoading SST-2 sentiment dataset (GLUE)...")

    dataset = load_dataset("glue", "sst2", split="validation")

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    texts = dataset["sentence"]
    true_labels = dataset["label"]
    predictions = []

    print(f"Evaluating on {len(texts)} samples...")

    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(texts)} samples...")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            pred = torch.argmax(
                model(**inputs).logits, dim=-1
            ).item()
            predictions.append(pred)

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predictions,
        average="binary",
    )

    print("\n" + "=" * 60)
    print("SENTIMENT EVALUATION (SST-2 GLUE benchmark)")
    print("Model: distilbert-base-uncased-finetuned-sst-2-english")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Samples:   {len(texts)}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


# ─────────────────────────────────────────────
# SANITY CHECKS
# ─────────────────────────────────────────────
def quick_test_political():
    """Sanity check — political bias model."""
    print("\n🧪 QUICK SANITY CHECK (Political Bias)\n")

    evaluator = BiasEvaluator(BIAS_MODEL)

    test_texts = [
        "Biden's socialist agenda is ruining America.",
        "The economy shows steady growth across sectors.",
        "Trump's conservative policies protect our values.",
    ]

    print("Inspecting raw predictions (0=Left, 1=Center, 2=Right):")
    for text in test_texts:
        pred = evaluator.predict(text)
        label = (
            ["Left", "Center", "Right"][pred["label"]]
            if pred["label"] < 3 else str(pred["label"])
        )
        print(
            f"  {label} ({pred['confidence']:.2f}) | {text[:55]}..."
        )

    # Raw 3-class: 0=Left, 1=Center, 2=Right
    results = evaluator.evaluate_on_dataset(test_texts, [0, 1, 2])

    print("\n" + "=" * 60)
    print("QUICK TEST (Political Bias — 3-class)")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.2f}")
    print("(Note: model predicts RIGHT for Left text — known model quirk)")


def quick_test_general():
    """
    Sanity check — general bias model.

    Model label scheme: 0=BIASED, 1=NEUTRAL (inverted vs MBIB).
    After remap (1-pred): biased text → 1, neutral text → 0.
    """
    print("\n🧪 QUICK SANITY CHECK (General Bias)\n")

    evaluator = BiasEvaluator(GENERAL_BIAS_MODEL)

    test_texts = [
        "The radical left is destroying our country with dangerous policies.",
        "The government released its annual budget report today.",
        "Corrupt politicians are stealing from hardworking taxpayers.",
    ]

    print("Raw model outputs (0=BIASED, 1=NEUTRAL in this model):")
    for text in test_texts:
        pred = evaluator.predict(text)
        raw = "BIASED" if pred["label"] == 0 else "NEUTRAL"
        remapped = "Biased(1)" if pred["label"] == 0 else "Unbiased(0)"
        print(
            f"  Raw:{raw} → Remapped:{remapped} | "
            f"Conf:{pred['confidence']:.2f} | {text[:50]}..."
        )

    # MBIB convention after remap: biased=1, unbiased=0
    # Texts 1 and 3 are biased → 1, text 2 is neutral → 0
    test_labels = [1, 0, 1]

    results = evaluator.evaluate_on_dataset(
        test_texts,
        test_labels,
        remap_fn=lambda p: 1 - p,  # invert model labels
    )

    print("\n" + "=" * 60)
    print("QUICK TEST (General Bias — after remap)")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.2f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 NEWSCOPE MODEL EVALUATION SUITE")
    print("=" * 70)

    # ── Sanity checks ──────────────────────────────────────────────
    quick_test_political()
    quick_test_general()

    # ── 1. Sentiment (SST-2) ───────────────────────────────────────
    print("\n\n1️⃣  SENTIMENT ANALYSIS (SST-2 GLUE)")
    print("-" * 70)
    evaluate_sentiment()

    # ── 2. Political bias (MBIB political_bias — shuffled + remap) ─
    # Set sample_size=None for full dissertation run
    print("\n\n2️⃣  POLITICAL BIAS (MBIB political_bias — shuffled)")
    print("-" * 70)
    evaluate_political_bias(sample_size=1000)

    # ── 3. General bias (MBIB linguistic_bias — shuffled + invert) ─
    print("\n\n3️⃣  GENERAL BIAS (MBIB linguistic_bias — shuffled)")
    print("-" * 70)
    evaluate_general_bias(sample_size=1000)

    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70 + "\n")
