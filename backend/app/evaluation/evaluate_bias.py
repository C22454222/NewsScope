"""
NewsScope Model Evaluation Suite
=================================
Benchmarks:

  1. Sentiment     → SST-2 GLUE validation (872 samples)
                     In-distribution. Expected F1: ~0.91–0.93

  2. Political     → [Primary]   siddharthmb/article-bias-prediction-media-splits
                                 AllSides 3-class (Left/Center/Right), in-distribution
                                 No remapping. Expected weighted F1: ~0.75–0.94
                     [Secondary] hyperpartisan_news_detection / bypublisher
                                 Binary hyperpartisan, article-level
                                 Remap: Left+Right → Hyper(1), Center → Not(0)
                     [Baseline]  mediabiasgroup/mbib-base / political_bias
                                 Binary collapse — known ceiling ~50–60%

  3. General       → [Primary]   mediabiasgroup/BABE
                                 Expert-annotated sentences, 3 700 samples
                                 Remap: 1 - pred. Expected macro F1: ~0.80
                     [Secondary] valurank/wikirev-bias
                                 In-distribution (Wikipedia edits). Remap: 1 - pred
                     [Baseline]  mediabiasgroup/mbib-base / linguistic_bias
                                 Known ceiling ~50–60% (Wikipedia→News domain shift)

References:
  Baly et al. (2020)    Article-Bias-Prediction (AllSides)
  Kiesel et al. (2019)  SemEval-2019 Task 4 — Hyperpartisan News Detection
  Spinde et al. (2022)  BABE — Bias Annotations By Experts
  Wessel et al. (2023)  MBIB — Media Bias Identification Benchmark
"""

import torch
from collections import Counter

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Model identifiers ──────────────────────────────────────────────────────────
POLITICAL_MODEL = "premsa/political-bias-prediction-allsides-BERT"
GENERAL_BIAS_MODEL = "valurank/distilroberta-bias"
SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# ── Label maps ─────────────────────────────────────────────────────────────────
# premsa model outputs: 0=Left, 1=Center, 2=Right
POLITICAL_LABELS = {0: "Left", 1: "Center", 2: "Right"}


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════


class BiasEvaluator:
    """Load a HuggingFace sequence-classification model and run evaluations."""

    def __init__(self, model_name: str):
        print(f"  Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {self.device}")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Return prediction dict for a single text."""
        if not text or not str(text).strip():
            return {"label": 0, "confidence": 0.0, "probabilities": []}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)

        pred = torch.argmax(probs, dim=-1).item()
        return {
            "label": pred,
            "confidence": probs[0][pred].item(),
            "probabilities": probs[0].cpu().numpy(),
        }

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[int]:
        """
        Predict labels over a list of texts using true batching.

        Skips empty/null texts and returns predictions in original order.
        Skipped positions default to label 0.
        """
        valid = [(i, t) for i, t in enumerate(texts) if t and str(t).strip()]
        skipped = len(texts) - len(valid)
        if skipped:
            print(f"  Skipped {skipped} empty/null texts")

        if not valid:
            return [0] * len(texts)

        indices, valid_texts = zip(*valid)
        print(f"  Running inference on {len(valid_texts)} samples...")

        raw_preds: dict[int, int] = {}

        for i in range(0, len(valid_texts), batch_size):
            batch_texts = list(valid_texts[i:i + batch_size])
            batch_indices = indices[i: i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                probs = torch.softmax(self.model(**inputs).logits, dim=-1)

            for idx, pred in zip(
                batch_indices, torch.argmax(probs, dim=-1).cpu().tolist()
            ):
                raw_preds[idx] = pred

            processed = i + len(batch_texts)
            if (processed // batch_size) % 10 == 0:
                print(f"    {processed}/{len(valid_texts)}")

        return [raw_preds.get(i, 0) for i in range(len(texts))]

    def evaluate_on_dataset(
        self,
        texts: list[str],
        true_labels: list[int],
        remap_fn=None,
        average: str = "weighted",
    ) -> dict:
        """
        Predict over texts and compute classification metrics.

        Args:
            texts:       Input texts.
            true_labels: Ground-truth integer labels.
            remap_fn:    Optional callable to remap raw predictions (e.g. invert
                         labels or collapse a 3-class output to binary).
            average:     sklearn averaging strategy for precision/recall/F1.
                         Use "weighted" (multi-class), "macro", or "binary".

        Returns:
            Dict with accuracy, precision, recall, f1_score, confusion_matrix,
            classification_report, num_samples.
        """
        raw = self.predict_batch(texts)
        predictions = [remap_fn(p) for p in raw] if remap_fn else raw

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average=average, zero_division=0
        )
        cm = confusion_matrix(true_labels, predictions)
        report = classification_report(
            true_labels, predictions, output_dict=True, zero_division=0
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


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _print_results(
    title: str,
    model_name: str,
    results: dict,
    notes: str = "",
) -> None:
    """Pretty-print a results dict returned by evaluate_on_dataset."""
    print("\n" + "=" * 70)
    print(title)
    print(f"Model  : {model_name}")
    if notes:
        print(f"Notes  : {notes}")
    print("=" * 70)
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1_score']:.4f}")
    print(f"Samples   : {results['num_samples']}")

    print("\nConfusion Matrix:")
    for row in results["confusion_matrix"]:
        print(f"  {row}")

    print("\nPer-class Report:")
    for label, metrics in results["classification_report"].items():
        if isinstance(metrics, dict):
            print(
                f"  {str(label):<10}  "
                f"P={metrics['precision']:.3f}  "
                f"R={metrics['recall']:.3f}  "
                f"F1={metrics['f1-score']:.3f}  "
                f"N={int(metrics['support'])}"
            )


def _log_distribution(labels: list[int], label_map: dict | None = None) -> None:
    dist = Counter(labels)
    if label_map:
        dist = {label_map.get(k, k): v for k, v in sorted(dist.items())}
    print(f"  Class distribution: {dist}")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADERS
# ══════════════════════════════════════════════════════════════════════════════


def _filter_and_unzip(
    texts_raw, labels_raw, valid_labels: set | None = None
) -> tuple[list[str], list[int]]:
    """Drop empty texts and optionally restrict to valid label values."""
    pairs = [
        (str(t), int(l))
        for t, l in zip(texts_raw, labels_raw)
        if t and str(t).strip() and (valid_labels is None or int(l) in valid_labels)
    ]
    if not pairs:
        raise ValueError("No valid samples after filtering.")
    texts, labels = zip(*pairs)
    return list(texts), list(labels)


def _load_allsides(
    sample_size: int | None = None, seed: int = 42
) -> tuple[list[str], list[int]]:
    """
    Load Article-Bias-Prediction (AllSides) dataset.

    Fields used: 'content' (text), 'bias' (ClassLabel: 0=Left, 1=Center, 2=Right).
    Source: siddharthmb/article-bias-prediction-media-splits (Baly et al., 2020).

    IMPORTANT: Verify bias ClassLabel ordering on first run by checking
    ds.features['bias'].names — it should be ['left', 'center', 'right'].
    If the order differs, adjust POLITICAL_LABELS accordingly.
    """
    print("  Loading Article-Bias-Prediction (AllSides)...")
    try:
        ds = load_dataset(
            "siddharthmb/article-bias-prediction-media-splits", split="test"
        )
    except Exception:
        try:
            # Some mirrors only have a train split
            ds = load_dataset(
                "siddharthmb/article-bias-prediction-media-splits", split="train"
            )
            print("  ⚠️  No test split — using train split")
        except Exception as e:
            raise RuntimeError(
                f"AllSides dataset unavailable: {e}\n"
                "  Alt: clone https://github.com/ramybaly/Article-Bias-Prediction\n"
                "  and load from data/splits/test.jsonl"
            ) from e

    # Log label ordering so we can verify 0=Left, 1=Center, 2=Right
    if hasattr(ds.features.get("bias", None), "names"):
        print(f"  bias ClassLabel ordering: {ds.features['bias'].names}")

    ds = ds.shuffle(seed=seed)
    if sample_size:
        ds = ds.select(range(min(sample_size, len(ds))))

    texts, labels = _filter_and_unzip(ds["content"], ds["bias"])
    _log_distribution(labels, POLITICAL_LABELS)
    return texts, labels


def _load_hyperpartisan(
    sample_size: int | None = None, seed: int = 42
) -> tuple[list[str], list[int]]:
    """
    Load SemEval-2019 Task 4 — Hyperpartisan News Detection (bypublisher).

    Labels: False/0 = Not Hyperpartisan, True/1 = Hyperpartisan.
    Source: hyperpartisan_news_detection / bypublisher (Kiesel et al., 2019).

    NOTE: This dataset may require accepting terms on HuggingFace.
    """
    print("  Loading SemEval-2019 Task 4 — Hyperpartisan (bypublisher)...")
    try:
        ds = load_dataset(
            "hyperpartisan_news_detection", "bypublisher", split="train"
        )
    except Exception as e:
        raise RuntimeError(
            f"Hyperpartisan dataset unavailable: {e}\n"
            "  Accept terms at: https://huggingface.co/datasets/hyperpartisan_news_detection\n"
            "  or download from: https://pan.webis.de/semeval19/semeval19-web/"
        ) from e

    ds = ds.shuffle(seed=seed)
    if sample_size:
        ds = ds.select(range(min(sample_size, len(ds))))

    # 'hyperpartisan' field is a boolean → cast to int
    texts, labels = _filter_and_unzip(ds["text"], [int(x) for x in ds["hyperpartisan"]])
    _log_distribution(labels, {0: "Not-Hyperpartisan", 1: "Hyperpartisan"})
    return texts, labels


def _load_babe(
    sample_size: int | None = None, seed: int = 42
) -> tuple[list[str], list[int]]:
    """
    Load BABE — Bias Annotations By Experts.

    3,700 expert-annotated news sentences across 46 outlets and 15 topics.
    Labels: 0=Non-biased, 1=Biased.
    Source: mediabiasgroup/BABE (Spinde et al., 2022).
    Best reported macro F1: 0.804.
    """
    print("  Loading BABE (Bias Annotations By Experts)...")
    for split_name in ("test", "train"):
        try:
            ds = load_dataset("mediabiasgroup/BABE", split=split_name)
            if split_name == "train":
                print("  ⚠️  No test split — using full BABE dataset")
            break
        except Exception:
            continue
    else:
        raise RuntimeError(
            "Could not load BABE.\n"
            "  See: https://huggingface.co/datasets/mediabiasgroup/BABE"
        )

    ds = ds.shuffle(seed=seed)
    if sample_size:
        ds = ds.select(range(min(sample_size, len(ds))))

    # Keep only binary-labelled rows (some versions include a "no agreement" class)
    texts, labels = _filter_and_unzip(ds["text"], ds["label"], valid_labels={0, 1})
    _log_distribution(labels, {0: "Non-biased", 1: "Biased"})
    return texts, labels


def _load_wikirev_bias(
    sample_size: int | None = None, seed: int = 42
) -> tuple[list[str], list[int]]:
    """
    Load valurank/wikirev-bias (Wikipedia neutralisation edits).

    In-distribution dataset for distilroberta-bias — highest expected F1.
    Model label scheme: 0=BIASED, 1=NEUTRAL → remap with 1 - pred.

    NOTE: Dataset may be private or require HuggingFace access.
    """
    print("  Loading valurank/wikirev-bias...")
    for split_name in ("test", "train"):
        try:
            ds = load_dataset("valurank/wikirev-bias", split=split_name)
            if split_name == "train":
                print("  ⚠️  No test split — using train split")
            break
        except Exception:
            continue
    else:
        raise RuntimeError(
            "Could not load wikirev-bias. Dataset may be private.\n"
            "  See: https://huggingface.co/valurank/distilroberta-bias"
        )

    ds = ds.shuffle(seed=seed)
    if sample_size:
        ds = ds.select(range(min(sample_size, len(ds))))

    texts, labels = _filter_and_unzip(ds["text"], ds["label"])
    _log_distribution(labels)
    return texts, labels


def _load_mbib_split(
    split_name: str,
    sample_size: int | None = None,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """
    Load a named split from MBIB (Wessel et al., 2023).

    CRITICAL: shuffle BEFORE selecting to avoid class imbalance —
    MBIB splits are ordered by class label.
    """
    print(f"  Loading MBIB split: {split_name}...")
    ds = load_dataset("mediabiasgroup/mbib-base")

    if split_name not in ds:
        raise ValueError(
            f"Split '{split_name}' not found. Available: {list(ds.keys())}"
        )

    split = ds[split_name].shuffle(seed=seed)

    if sample_size is not None:
        print(f"  ⚠️  Sampling {sample_size} shuffled examples")
        split = split.select(range(min(sample_size, len(split))))

    texts, labels = _filter_and_unzip(split["text"], split["label"])
    _log_distribution(labels, {0: "Unbiased", 1: "Biased"})
    return texts, labels


# ══════════════════════════════════════════════════════════════════════════════
# 1. SENTIMENT — SST-2 GLUE
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_sentiment() -> dict:
    """
    Evaluate distilbert-base-uncased-finetuned-sst-2-english on SST-2 GLUE
    validation set (872 samples).

    In-distribution benchmark.
    Labels: 0=Negative, 1=Positive.
    Expected: accuracy ~0.91, F1 ~0.91–0.93.
    """
    print("\n📊 SENTIMENT — SST-2 GLUE Validation")
    ds = load_dataset("glue", "sst2", split="validation")

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model.to(device)
    model.eval()

    texts = list(ds["sentence"])
    true_labels = list(ds["label"])
    predictions: list[int] = []

    print(f"  Evaluating on {len(texts)} samples...")
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"  {i}/{len(texts)}...")
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            predictions.append(
                torch.argmax(model(**inputs).logits, dim=-1).item()
            )

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="binary", zero_division=0
    )

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "num_samples": len(texts),
        "confusion_matrix": confusion_matrix(true_labels, predictions).tolist(),
        "classification_report": classification_report(
            true_labels, predictions, output_dict=True, zero_division=0
        ),
    }

    _print_results(
        "SENTIMENT — SST-2 GLUE Validation",
        SENTIMENT_MODEL,
        results,
        notes="In-distribution | Expected F1 ~0.91–0.93",
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. POLITICAL BIAS
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_political_allsides(sample_size: int | None = None) -> dict:
    """
    [PRIMARY] Evaluate AllSides BERT on the Article-Bias-Prediction test split.

    In-distribution: model and dataset share the same label space.
    No remapping applied — both use 0=Left, 1=Center, 2=Right.
    Expected weighted F1: ~0.75–0.94 (Baly et al., 2020).
    """
    print("\n📊 POLITICAL BIAS [Primary] — Article-Bias-Prediction (AllSides, 3-class)")
    texts, true_labels = _load_allsides(sample_size)
    print(f"  Loaded {len(texts)} samples")

    evaluator = BiasEvaluator(POLITICAL_MODEL)
    results = evaluator.evaluate_on_dataset(
        texts,
        true_labels,
        remap_fn=None,       # no remap — native 3-class evaluation
        average="weighted",
    )

    _print_results(
        "POLITICAL BIAS — AllSides Test Split (3-class, in-distribution)",
        POLITICAL_MODEL,
        results,
        notes="0=Left, 1=Center, 2=Right | No remapping | Expected F1 ~0.75–0.94",
    )
    return results


def evaluate_political_hyperpartisan(sample_size: int | None = None) -> dict:
    """
    [SECONDARY] Evaluate AllSides BERT on SemEval-2019 Task 4 (Hyperpartisan).

    Remap: Left(0) → Hyperpartisan(1)
           Center(1) → Not-Hyperpartisan(0)
           Right(2) → Hyperpartisan(1)

    State-of-the-art macro F1 on this benchmark: ~0.77–0.78
    (Kiesel et al., 2019).
    """
    print("\n📊 POLITICAL BIAS [Secondary] — SemEval-2019 Task 4 (Hyperpartisan)")
    texts, true_labels = _load_hyperpartisan(sample_size)
    print(f"  Loaded {len(texts)} samples")

    # Center(1) maps to Not-Hyperpartisan(0); Left or Right maps to Hyperpartisan(1)
    def to_hyperpartisan(pred: int) -> int:
        return 0 if pred == 1 else 1

    evaluator = BiasEvaluator(POLITICAL_MODEL)
    results = evaluator.evaluate_on_dataset(
        texts,
        true_labels,
        remap_fn=to_hyperpartisan,
        average="binary",
    )

    _print_results(
        "POLITICAL BIAS — SemEval-2019 Task 4 (binary hyperpartisan)",
        POLITICAL_MODEL,
        results,
        notes="Remap: Left+Right → Hyperpartisan(1), Center → Not(0)",
    )
    return results


def evaluate_political_mbib(sample_size: int | None = None) -> dict:
    """
    [BASELINE] Evaluate AllSides BERT on MBIB political_bias split.

    Kept for comparison with Wessel et al. (2023) published results.
    Known ceiling ~50–60% due to 3-class→binary collapse + domain mismatch.

    Remap: Left(0) → Biased(1)
           Center(1) → Unbiased(0)
           Right(2) → Biased(1)
    """
    print("\n📊 POLITICAL BIAS [Baseline] — MBIB political_bias")
    texts, true_labels = _load_mbib_split("political_bias", sample_size)
    print(f"  Loaded {len(texts)} samples")

    def to_binary(pred: int) -> int:
        return 0 if pred == 1 else 1

    evaluator = BiasEvaluator(POLITICAL_MODEL)
    results = evaluator.evaluate_on_dataset(
        texts,
        true_labels,
        remap_fn=to_binary,
        average="weighted",
    )

    _print_results(
        "POLITICAL BIAS — MBIB political_bias (Wessel 2023 baseline)",
        POLITICAL_MODEL,
        results,
        notes=(
            "Remap: Left+Right → Biased(1), Center → Unbiased(0) | "
            "Known ceiling ~50–60%"
        ),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. GENERAL / LEXICAL BIAS
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_general_babe(sample_size: int | None = None) -> dict:
    """
    [PRIMARY] Evaluate distilroberta-bias on BABE.

    BABE is the gold-standard sentence-level media bias benchmark:
    3,700 expert-annotated sentences, 46 outlets, 15 topics.

    Model label scheme: 0=BIASED, 1=NEUTRAL (inverted vs BABE convention).
    Remap: 1 - pred  →  model-BIASED(0) → label-Biased(1)
                        model-NEUTRAL(1) → label-Non-biased(0)

    Best reported macro F1 on BABE: 0.804 (Spinde et al., 2022).
    """
    print("\n📊 GENERAL BIAS [Primary] — BABE (Expert-Annotated)")
    texts, true_labels = _load_babe(sample_size)
    print(f"  Loaded {len(texts)} samples")

    evaluator = BiasEvaluator(GENERAL_BIAS_MODEL)
    results = evaluator.evaluate_on_dataset(
        texts,
        true_labels,
        remap_fn=lambda p: 1 - p,
        average="weighted",
    )

    _print_results(
        "GENERAL BIAS — BABE (Expert-Annotated News Sentences)",
        GENERAL_BIAS_MODEL,
        results,
        notes=(
            "Remap: model-0=BIASED→1, model-1=NEUTRAL→0 | "
            "Best reported macro F1: 0.804"
        ),
    )
    return results


def evaluate_general_wikirev(sample_size: int | None = None) -> dict:
    """
    [SECONDARY] Evaluate distilroberta-bias on valurank/wikirev-bias.

    In-distribution: model was trained on Wikipedia neutralisation edits.
    Highest expected F1 of all three general bias benchmarks.

    Model label scheme: 0=BIASED, 1=NEUTRAL.
    Remap: 1 - pred.

    NOTE: Dataset may be private. If unavailable, BABE is the primary benchmark.
    """
    print("\n📊 GENERAL BIAS [Secondary] — wikirev-bias (In-Distribution)")
    texts, true_labels = _load_wikirev_bias(sample_size)
    print(f"  Loaded {len(texts)} samples")

    evaluator = BiasEvaluator(GENERAL_BIAS_MODEL)
    results = evaluator.evaluate_on_dataset(
        texts,
        true_labels,
        remap_fn=lambda p: 1 - p,
        average="weighted",
    )

    _print_results(
        "GENERAL BIAS — valurank/wikirev-bias (In-Distribution)",
        GENERAL_BIAS_MODEL,
        results,
        notes="Remap: model-0=BIASED→1, model-1=NEUTRAL→0 | Highest expected F1",
    )
    return results


def evaluate_general_mbib(sample_size: int | None = None) -> dict:
    """
    [BASELINE] Evaluate distilroberta-bias on MBIB linguistic_bias split.

    Kept for comparison with Wessel et al. (2023) published results.
    Known ceiling ~50–60% due to Wikipedia→News domain shift.

    Model label scheme: 0=BIASED, 1=NEUTRAL.
    Remap: 1 - pred.
    """
    print("\n📊 GENERAL BIAS [Baseline] — MBIB linguistic_bias")
    texts, true_labels = _load_mbib_split("linguistic_bias", sample_size)
    print(f"  Loaded {len(texts)} samples")

    evaluator = BiasEvaluator(GENERAL_BIAS_MODEL)
    results = evaluator.evaluate_on_dataset(
        texts,
        true_labels,
        remap_fn=lambda p: 1 - p,
        average="weighted",
    )

    _print_results(
        "GENERAL BIAS — MBIB linguistic_bias (Wessel 2023 baseline)",
        GENERAL_BIAS_MODEL,
        results,
        notes=(
            "Remap: model-0=BIASED→1, model-1=NEUTRAL→0 | "
            "Known ceiling ~50–60%"
        ),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SANITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════


def quick_test_political() -> None:
    """
    Sanity-check political bias model against three hand-crafted examples.

    Checks raw model outputs before running full benchmarks so you can
    catch label-ordering issues early.
    """
    print("\n🧪 SANITY CHECK — Political Bias Model")
    evaluator = BiasEvaluator(POLITICAL_MODEL)

    # (text, expected_label)  — expected based on AllSides framing
    examples = [
        ("Biden's socialist agenda is destroying America.", 0),   # Left-attacking → expect Right
        ("The government released its annual economic report.", 1),  # Neutral → Center
        ("Trump's conservative policies protect our freedoms.", 2),  # Right-leaning → Right
    ]

    print("\n  Raw predictions (0=Left, 1=Center, 2=Right):")
    for text, expected in examples:
        pred = evaluator.predict(text)
        got = POLITICAL_LABELS.get(pred["label"], str(pred["label"]))
        exp = POLITICAL_LABELS.get(expected, str(expected))
        mark = "✓" if pred["label"] == expected else "✗"
        print(
            f"  {mark} Got:{got:<7} Expected:{exp:<7} "
            f"Conf:{pred['confidence']:.2f} | {text[:60]}"
        )


def quick_test_general() -> None:
    """
    Sanity-check general bias model with three hand-crafted examples.

    Model raw label scheme: 0=BIASED, 1=NEUTRAL (inverted vs convention).
    After remap (1-pred): biased text → 1, neutral text → 0.
    """
    print("\n🧪 SANITY CHECK — General Bias Model")
    evaluator = BiasEvaluator(GENERAL_BIAS_MODEL)

    # (text, expected_remapped_label: 0=unbiased, 1=biased)
    examples = [
        ("The radical left is destroying our country with dangerous policies.", 1),
        ("The committee published its findings in a quarterly report.", 0),
        ("Corrupt politicians are stealing from hardworking taxpayers.", 1),
    ]

    print("\n  Raw outputs then remapped (model: 0=BIASED, 1=NEUTRAL):")
    for text, expected in examples:
        pred = evaluator.predict(text)
        raw_label = "BIASED" if pred["label"] == 0 else "NEUTRAL"
        remapped = 1 - pred["label"]
        mark = "✓" if remapped == expected else "✗"
        print(
            f"  {mark} Raw:{raw_label:<7} → Remapped:{remapped} "
            f"Expected:{expected} Conf:{pred['confidence']:.2f} | {text[:55]}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪  NEWSCOPE MODEL EVALUATION SUITE")
    print("=" * 70)

    # ── Sanity checks (run first to catch label-ordering issues) ────────────
    quick_test_political()
    quick_test_general()

    # ── 1. Sentiment ────────────────────────────────────────────────────────
    print("\n\n1️⃣  SENTIMENT — SST-2 GLUE")
    print("-" * 70)
    evaluate_sentiment()

    # ── 2. Political Bias ───────────────────────────────────────────────────
    # Primary:  AllSides test split (3-class, in-distribution)
    # Secondary: SemEval-2019 Hyperpartisan (binary, article-level)
    # Baseline: MBIB political_bias (binary collapse — ~50–60% ceiling)
    print("\n\n2️⃣  POLITICAL BIAS")
    print("-" * 70)

    try:
        evaluate_political_allsides(sample_size=None)      # full dataset
    except RuntimeError as e:
        print(f"  ⚠️  AllSides unavailable: {e}")

    try:
        evaluate_political_hyperpartisan(sample_size=None)
    except RuntimeError as e:
        print(f"  ⚠️  Hyperpartisan unavailable: {e}")

    evaluate_political_mbib(sample_size=1000)              # baseline comparison

    # ── 3. General Bias ─────────────────────────────────────────────────────
    # Primary:  BABE (expert-annotated, gold standard)
    # Secondary: wikirev-bias (in-distribution, may be private)
    # Baseline: MBIB linguistic_bias (~50–60% ceiling, domain shift)
    print("\n\n3️⃣  GENERAL BIAS")
    print("-" * 70)

    try:
        evaluate_general_babe(sample_size=None)            # full dataset
    except RuntimeError as e:
        print(f"  ⚠️  BABE unavailable: {e}")

    try:
        evaluate_general_wikirev(sample_size=None)
    except RuntimeError as e:
        print(f"  ⚠️  wikirev-bias unavailable: {e}")

    evaluate_general_mbib(sample_size=1000)                # baseline comparison

    print("\n" + "=" * 70)
    print("✅  EVALUATION COMPLETE")
    print("=" * 70 + "\n")
