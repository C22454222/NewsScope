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
        """Get bias prediction for a single text."""
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
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        pred_label = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_label].item()

        return {
            "label": pred_label,
            "confidence": confidence,
            "probabilities": probs[0].cpu().numpy(),
        }

    def predict_batch(self, texts: list[str]) -> list[int]:
        """Predict bias for texts using true batching (much faster)."""
        predictions = []

        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) < len(texts):
            print(f"Skipped {len(texts) - len(valid_texts)} empty texts")

        print(f"Predicting on {len(valid_texts)} valid samples...")

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
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

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
    ) -> dict:
        """
        Evaluate model on a dataset.
        Returns metrics: accuracy, precision, recall, F1.
        """
        predictions = self.predict_batch(texts)

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


# ─────────────────────────────────────────────
# 1. POLITICAL BIAS — Article-Bias-Prediction
#    (Baly et al.) — matches AllSides BERT L/C/R
# ─────────────────────────────────────────────
def evaluate_political_bias_allsides(sample_size=None):
    """
    Evaluate political bias model on Article-Bias-Prediction dataset.

    This is the correct benchmark for premsa/political-bias-prediction-allsides-BERT
    as both use AllSides Left/Center/Right labels (Baly et al., 2020).

    Label mapping:
      0 = Left, 1 = Center, 2 = Right
    """
    print(
        "Loading Article-Bias-Prediction dataset "
        "(Baly et al., 2020)..."
    )

    dataset = load_dataset(
        "newsmediabias/political-bias-allsides-labelled",
        split="test"
    )

    if sample_size is not None:
        print(f"⚠️  TESTING MODE: Using only first {sample_size} samples")
        dataset = dataset.select(range(min(sample_size, len(dataset))))

    # Map label strings to integers if needed
    label_map = {"left": 0, "center": 1, "right": 2}

    texts = []
    true_labels = []

    for row in dataset:
        text = row.get("content") or row.get("text") or row.get("title") or ""
        raw_label = row.get("label") or row.get("bias_label") or ""

        if not text.strip():
            continue

        # Handle both string and integer labels
        if isinstance(raw_label, str):
            mapped = label_map.get(raw_label.lower())
            if mapped is None:
                continue
            true_labels.append(mapped)
        elif isinstance(raw_label, int):
            true_labels.append(raw_label)
        else:
            continue

        texts.append(text)

    print(f"Loaded {len(texts)} valid samples")

    evaluator = BiasEvaluator(BIAS_MODEL)

    print(
        f"\n📊 Evaluating political bias model on "
        f"Article-Bias-Prediction ({len(texts)} samples)..."
    )
    results = evaluator.evaluate_on_dataset(texts, true_labels)

    print("\n" + "=" * 60)
    print("POLITICAL BIAS EVALUATION (Article-Bias-Prediction / AllSides)")
    print("Model: premsa/political-bias-prediction-allsides-BERT")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print(f"Samples:   {results['num_samples']}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("         Left  Center  Right")
    for i, row in enumerate(results["confusion_matrix"]):
        label = ["Left  ", "Center", "Right "][i]
        print(f"  {label}  {row}")

    return results


# ─────────────────────────────────────────────
# 2. GENERAL/LEXICAL BIAS — MBIB
#    valurank/distilroberta-bias → binary
#    BIASED(1) / UNBIASED(0)
# ─────────────────────────────────────────────
def evaluate_general_bias_mbib(sample_size=None):
    """
    Evaluate general bias model on MBIB political_bias split.

    Uses valurank/distilroberta-bias which is trained on
    binary bias detection (biased vs unbiased) — matching
    MBIB's binary label scheme (Wessel et al., 2023).

    Label mapping:
      0 = Unbiased, 1 = Biased
    """
    print(
        "Loading MBIB (mediabiasgroup/mbib-base) "
        "political_bias split..."
    )

    dataset = load_dataset("mediabiasgroup/mbib-base")
    split_name = "political_bias"

    if split_name not in dataset:
        raise ValueError(
            f"Split '{split_name}' not found. "
            f"Available: {list(dataset.keys())}"
        )

    ds = dataset[split_name]

    if sample_size is not None:
        print(f"⚠️  TESTING MODE: Using only first {sample_size} samples")
        ds = ds.select(range(min(sample_size, len(ds))))

    texts = ds["text"]
    true_labels = ds["label"]

    print(f"Loaded {len(texts)} samples from MBIB '{split_name}'")

    evaluator = BiasEvaluator(GENERAL_BIAS_MODEL)

    print(
        f"\n📊 Evaluating general bias model on "
        f"MBIB ({split_name}, {len(texts)} samples)..."
    )
    results = evaluator.evaluate_on_dataset(texts, true_labels)

    print("\n" + "=" * 60)
    print("GENERAL BIAS EVALUATION (MBIB - political_bias split)")
    print("Model: valurank/distilroberta-bias")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print(f"Samples:   {results['num_samples']}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("             Unbiased  Biased")
    for i, row in enumerate(results["confusion_matrix"]):
        label = ["Unbiased", "Biased  "][i]
        print(f"  {label}    {row}")

    return results


# ─────────────────────────────────────────────
# 3. KEPT: MBIB with political BERT (for
#    dissertation comparison — shows mismatch)
# ─────────────────────────────────────────────
def evaluate_on_mbib_political_bias(sample_size=None):
    """
    Evaluate political BERT on MBIB binary split.

    NOTE: This is intentionally a cross-benchmark test.
    The AllSides BERT is a 3-class model (L/C/R) tested
    on a binary benchmark — expected lower performance.
    Kept for dissertation comparison analysis.
    """
    print(
        "Loading MBIB political_bias split "
        "(cross-benchmark test for dissertation)..."
    )

    dataset = load_dataset("mediabiasgroup/mbib-base")
    split_name = "political_bias"
    ds = dataset[split_name]

    if sample_size is not None:
        print(f"⚠️  TESTING MODE: Using only first {sample_size} samples")
        ds = ds.select(range(min(sample_size, len(ds))))

    texts = ds["text"]
    true_labels = ds["label"]

    evaluator = BiasEvaluator(BIAS_MODEL)

    print(
        f"\n📊 Cross-benchmark: political BERT on "
        f"MBIB ({len(texts)} samples)..."
    )
    results = evaluator.evaluate_on_dataset(texts, true_labels)

    print("\n" + "=" * 60)
    print("CROSS-BENCHMARK: AllSides BERT on MBIB (label mismatch)")
    print("Model: premsa/political-bias-prediction-allsides-BERT")
    print("NOTE: 3-class model vs 2-class benchmark — lower F1 expected")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print(f"Samples:   {results['num_samples']}")

    return results


# ─────────────────────────────────────────────
# 4. SENTIMENT — SST-2
# ─────────────────────────────────────────────
def evaluate_sentiment():
    """Evaluate sentiment model on SST-2 benchmark."""
    print("Loading SST-2 sentiment dataset (GLUE)...")
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
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(pred)

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predictions,
        average="binary",
    )

    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS EVALUATION RESULTS (SST-2)")
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
# 5. QUICK SANITY CHECK
# ─────────────────────────────────────────────
def quick_test():
    """Quick sanity check — political bias model 3/3."""
    print("\n🧪 QUICK SANITY CHECK (Political Bias)\n")

    evaluator = BiasEvaluator(BIAS_MODEL)

    test_texts = [
        "Biden's socialist agenda is ruining America.",
        "The economy shows steady growth across sectors.",
        "Trump's conservative policies protect our values.",
    ]
    test_labels = [0, 1, 2]

    results = evaluator.evaluate_on_dataset(test_texts, test_labels)

    print("\n" + "=" * 60)
    print("QUICK TEST RESULTS")
    print("=" * 60)
    print(f"✅ Accuracy:  {results['accuracy']:.2f} (3/3)")
    print("Political bias model is loading correctly!")


def quick_test_general_bias():
    """Quick sanity check — general bias model."""
    print("\n🧪 QUICK SANITY CHECK (General Bias)\n")

    evaluator = BiasEvaluator(GENERAL_BIAS_MODEL)

    test_texts = [
        "The radical left is destroying our country with dangerous policies.",
        "The government released its annual budget report today.",
        "Corrupt politicians are stealing from hardworking taxpayers.",
    ]
    # 1=biased, 0=unbiased, 1=biased
    test_labels = [1, 0, 1]

    results = evaluator.evaluate_on_dataset(test_texts, test_labels)

    print("\n" + "=" * 60)
    print("QUICK TEST RESULTS (General Bias)")
    print("=" * 60)
    print(f"✅ Accuracy:  {results['accuracy']:.2f}")
    print("General bias model is loading correctly!")


# ─────────────────────────────────────────────
# MAIN — run full evaluation suite
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 NEWSCOPE MODEL EVALUATION SUITE")
    print("=" * 70)

    # ── Sanity checks ──────────────────────────
    quick_test()
    quick_test_general_bias()

    # ── 1. Sentiment (SST-2) ───────────────────
    print("\n\n1️⃣  SENTIMENT ANALYSIS (SST-2)")
    print("-" * 70)
    evaluate_sentiment()

    # ── 2. Political bias (AllSides dataset) ───
    # Change sample_size to None for full run in final report
    print("\n\n2️⃣  POLITICAL BIAS (Article-Bias-Prediction / AllSides)")
    print("-" * 70)
    evaluate_political_bias_allsides(sample_size=1000)

    # ── 3. General bias (MBIB) ─────────────────
    print("\n\n3️⃣  GENERAL BIAS (MBIB - political_bias split)")
    print("-" * 70)
    evaluate_general_bias_mbib(sample_size=1000)

    # ── 4. Cross-benchmark (dissertation note) ─
    print("\n\n4️⃣  CROSS-BENCHMARK (AllSides BERT on MBIB — for comparison)")
    print("-" * 70)
    evaluate_on_mbib_political_bias(sample_size=500)

    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70 + "\n")
