"""
Benchmark script for NewsScope analysis models.

Runtime: ~5 minutes (vs ~30 min full run)

Usage:
    python -m app.evaluation.evaluate_bias           # quick (300 samples)
    python -m app.evaluation.evaluate_bias --full    # full (all samples)
"""

import sys
import warnings
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

QUICK = "--full" not in sys.argv
N_SAMPLES = 300 if QUICK else None   # None = use all available

print(f"\n{'='*70}")
print(f"NewsScope Model Evaluation {'(QUICK ~5min)' if QUICK else '(FULL ~30min)'}")
print(f"{'='*70}\n")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_results(title: str, model: str, notes: str, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, zero_division=0,
        target_names=None,
    )

    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"Model  : {model}")
    print(f"Notes  : {notes}")
    print(f"{'='*70}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Samples   : {len(y_true)}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nPer-class Report:\n{report}")


def load_pipeline(model_name: str, task: str = "text-classification"):
    from transformers import pipeline
    print(f"  Loading model: {model_name}...")
    return pipeline(task, model=model_name, device=-1, truncation=True)


def run_inference(pipe, texts: list[str], batch_size: int = 32) -> list[str]:
    results = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        outputs = pipe(batch, truncation=True, max_length=512)
        results.extend([o["label"] for o in outputs])
        done = min(i + batch_size, total)
        if done % 100 == 0 or done == total:
            print(f"    {done}/{total}")
    return results


# ─────────────────────────────────────────────
# 1. SENTIMENT — SST-2
# ─────────────────────────────────────────────

print("\n1️⃣  SENTIMENT")
print("-" * 70)

try:
    from datasets import load_dataset

    SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    print("  Loading SST-2 validation split...")
    sst2 = load_dataset("glue", "sst2", split="validation")

    texts = sst2["sentence"]
    labels = sst2["label"]  # 0=negative, 1=positive

    if N_SAMPLES:
        texts = texts[:N_SAMPLES]
        labels = labels[:N_SAMPLES]

    print(f"  Loaded {len(texts)} samples")

    pipe_sentiment = load_pipeline(SENTIMENT_MODEL)
    preds_raw = run_inference(pipe_sentiment, texts)

    # SST-2: NEGATIVE=0, POSITIVE=1
    label_map = {"NEGATIVE": 0, "POSITIVE": 1}
    y_pred = [label_map.get(p.upper(), 0) for p in preds_raw]
    y_true = list(labels)

    print_results(
        title="SENTIMENT — SST-2 GLUE Validation",
        model=SENTIMENT_MODEL,
        notes="In-distribution | Expected F1 ~0.91–0.93",
        y_true=y_true,
        y_pred=y_pred,
    )

except Exception as e:
    print(f"  ⚠️  Sentiment benchmark failed: {e}")


# ─────────────────────────────────────────────
# 2. POLITICAL BIAS — AllSides
# ─────────────────────────────────────────────

print("\n\n2️⃣  POLITICAL BIAS")
print("-" * 70)

BIAS_MODEL = "bucketresearch/politicalBiasBERT"

try:
    # F541 fix: plain string, no placeholders needed
    print("\n📊 POLITICAL BIAS — AllSides Test Split (3-class)")
    print("  Loading Article-Bias-Prediction (AllSides)...")

    ds = load_dataset(
        "cajcodes/political-bias",
        split="test",
    )

    # Class labels: 0=Left, 1=Center, 2=Right
    texts = ds["text"] if "text" in ds.column_names else ds["content"]
    labels = ds["label"] if "label" in ds.column_names else ds["bias"]

    if N_SAMPLES:
        indices = list(range(len(texts)))
        np.random.seed(42)
        np.random.shuffle(indices)
        indices = indices[:N_SAMPLES]
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]

    print(f"  Loaded {len(texts)} samples")

    pipe_bias = load_pipeline(BIAS_MODEL)
    preds_raw = run_inference(pipe_bias, texts)

    # bucketresearch/politicalBiasBERT outputs: Left, Center, Right
    pred_map = {"Left": 0, "Center": 1, "Right": 2}
    y_pred = [pred_map.get(p, 1) for p in preds_raw]
    y_true = list(labels)

    print_results(
        title="POLITICAL BIAS — AllSides (3-class, in-distribution)",
        model=BIAS_MODEL,
        notes="0=Left, 1=Center, 2=Right | Expected F1 ~0.70–0.88",
        y_true=y_true,
        y_pred=y_pred,
    )

except Exception as e:
    print(f"  ⚠️  Political bias (AllSides) benchmark failed: {e}")

# MBIB baseline — only in full mode (slow)
if not QUICK:
    try:
        # F541 fix: plain string
        print("\n📊 POLITICAL BIAS [Baseline] — MBIB political_bias")
        print("  Loading MBIB split: political_bias...")

        mbib = load_dataset(
            "mediabiasgroup/MBIB",
            "political_bias",
            split="test",
        )

        texts = mbib["text"][:1000]
        raw_labels = mbib["label"][:1000]
        # E741 fix: renamed l → lbl
        # Remap: Left+Right → Biased(1), Center → Unbiased(0)
        y_true = [0 if str(lbl) in ("Center", "1") else 1 for lbl in raw_labels]

        preds_raw = run_inference(pipe_bias, texts)
        y_pred = [0 if p == "Center" else 1 for p in preds_raw]

        print_results(
            title="POLITICAL BIAS — MBIB political_bias (baseline)",
            model=BIAS_MODEL,
            notes="Remap: Left+Right→Biased, Center→Unbiased | Ceiling ~50–60%",
            y_true=y_true,
            y_pred=y_pred,
        )

    except Exception as e:
        print(f"  ⚠️  MBIB political bias baseline failed: {e}")


# ─────────────────────────────────────────────
# 3. GENERAL BIAS — BABE
# ─────────────────────────────────────────────

print("\n\n3️⃣  GENERAL BIAS")
print("-" * 70)

GENERAL_BIAS_MODEL = "valurank/distilroberta-bias"

try:
    # F541 fix: plain string
    print("\n📊 GENERAL BIAS — BABE (Expert-Annotated)")
    print("  Loading BABE test split...")

    babe = load_dataset("mediabiasgroup/BABE", split="test")

    texts = babe["text"]
    raw_labels = babe["label"]  # 'Non-biased' or 'Biased'

    if N_SAMPLES:
        np.random.seed(42)
        indices = np.random.choice(len(texts), N_SAMPLES, replace=False)
        texts = [texts[i] for i in indices]
        raw_labels = [raw_labels[i] for i in indices]

    # E741 fix: renamed l → lbl
    # 0=Non-biased, 1=Biased
    y_true = [1 if str(lbl) == "Biased" else 0 for lbl in raw_labels]
    print(f"  Loaded {len(texts)} samples")

    pipe_general = load_pipeline(GENERAL_BIAS_MODEL)
    preds_raw = run_inference(pipe_general, texts)

    # distilroberta-bias: model-0=BIASED, model-1=NEUTRAL
    # Remap: BIASED→1, NEUTRAL→0
    def remap_general(label: str) -> int:
        u = label.upper()
        if "BIASED" in u and "UN" not in u:
            return 1
        return 0

    y_pred = [remap_general(p) for p in preds_raw]

    print_results(
        title="GENERAL BIAS — BABE (Expert-Annotated News Sentences)",
        model=GENERAL_BIAS_MODEL,
        notes="Remap: BIASED→1, NEUTRAL→0 | Best reported macro F1: 0.804",
        y_true=y_true,
        y_pred=y_pred,
    )

except Exception as e:
    print(f"  ⚠️  General bias (BABE) benchmark failed: {e}")


print(f"\n{'='*70}")
# F541 fix: plain string
print("✅  EVALUATION COMPLETE")
print(f"{'='*70}\n")
