"""
Benchmark script for NewsScope analysis models.

Runtime: ~5 minutes (vs ~30 min full run)

Usage:
    python -m app.evaluation.evaluate_bias           # quick (300 samples)
    python -m app.evaluation.evaluate_bias --full    # full (all samples)
"""

import sys
import warnings
from collections import Counter
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
N_SAMPLES = 300 if QUICK else None

# matous-volf/political-leaning-politics:
#   Trained on 12 datasets | Accuracy: ~84.7% AllSides
#   Outputs: LABEL_0=Left, LABEL_1=Center, LABEL_2=Right
#   Requires tokenizer: launch/POLITICS
BIAS_MODEL = "matous-volf/political-leaning-politics"
BIAS_TOKENIZER = "launch/POLITICS"

# E241 fix: no alignment spaces after ':' in dict literals
# Module-level pred map — shared by AllSides and MBIB sections
BIAS_PRED_MAP = {
    "LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2,
    "LEFT": 0, "CENTER": 1, "RIGHT": 2,
    "Left": 0, "Center": 1, "Right": 2,
    "left": 0, "center": 1, "right": 2,
    "0": 0, "1": 1, "2": 2,
}


def map_pred(pred: str) -> int:
    """Map raw model label to int. 0=Left, 1=Center, 2=Right."""
    return BIAS_PRED_MAP.get(pred, BIAS_PRED_MAP.get(pred.upper(), 1))


print(f"\n{'='*70}")
print(
    "NewsScope Model Evaluation "
    f"{'(QUICK ~5min)' if QUICK else '(FULL ~30min)'}"
)
print(f"{'='*70}\n")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_results(title: str, model: str, notes: str, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(
        y_true, y_pred, average="weighted", zero_division=0,
    )
    rec = recall_score(
        y_true, y_pred, average="weighted", zero_division=0,
    )
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, zero_division=0, target_names=None,
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


def load_pipeline(
    model_name: str,
    task: str = "text-classification",
    tokenizer_name: str | None = None,
):
    from transformers import pipeline
    print(f"  Loading model: {model_name}...")
    kwargs = {"model": model_name, "device": -1, "truncation": True}
    if tokenizer_name:
        kwargs["tokenizer"] = tokenizer_name
    return pipeline(task, **kwargs)


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

    SENTIMENT_MODEL = (
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

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
    sst2_map = {"NEGATIVE": 0, "POSITIVE": 1}
    y_pred = [sst2_map.get(p.upper(), 0) for p in preds_raw]
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
# 2. POLITICAL BIAS — AllSides 3-class
# ─────────────────────────────────────────────

print("\n\n2️⃣  POLITICAL BIAS")
print("-" * 70)

# Initialised to None — MBIB section checks before using
pipe_bias = None

try:
    print("\n📊 POLITICAL BIAS — AllSides (3-class)")
    print("  Loading cajcodes/political-bias (train split)...")

    ds = load_dataset("cajcodes/political-bias", split="train")

    all_texts = (
        ds["text"] if "text" in ds.column_names else ds["content"]
    )
    all_labels = (
        ds["label"] if "label" in ds.column_names else ds["bias"]
    )

    # Reproducible held-out slice — last 20% after shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(all_texts))
    split_point = int(len(indices) * 0.8)
    test_indices = indices[split_point:]

    if N_SAMPLES:
        test_indices = test_indices[:N_SAMPLES]

    texts = [all_texts[i] for i in test_indices]
    raw_labels = [all_labels[i] for i in test_indices]

    # AllSides 5-class: 0=Left, 1=Center-Left, 2=Center,
    #                   3=Center-Right, 4=Right
    # Collapse → 3-class: Left(0,1)→0  Center(2)→1  Right(3,4)→2
    collapse_map = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}

    def collapse_label(lbl) -> int:
        return collapse_map.get(int(lbl), 1)

    y_true = [collapse_label(lbl) for lbl in raw_labels]

    dist_raw = Counter(raw_labels)
    dist_col = Counter(y_true)
    print(f"  Raw 5-class distribution       : {dict(dist_raw)}")
    print(f"  Collapsed 3-class (0=L,1=C,2=R): {dict(dist_col)}")
    print(f"  Loaded {len(texts)} samples (held-out 20%)")

    pipe_bias = load_pipeline(BIAS_MODEL, tokenizer_name=BIAS_TOKENIZER)

    # Probe first sample to confirm output label format at runtime
    probe = pipe_bias([texts[0]], truncation=True, max_length=512)
    print(f"  Model output label format: '{probe[0]['label']}'")

    preds_raw = run_inference(pipe_bias, texts)

    unmapped = {
        p for p in set(preds_raw)
        if p not in BIAS_PRED_MAP and p.upper() not in BIAS_PRED_MAP
    }
    if unmapped:
        print(f"  ⚠️  Unmapped labels (defaulting Center): {unmapped}")

    y_pred = [map_pred(p) for p in preds_raw]

    print_results(
        title="POLITICAL BIAS — AllSides (3-class, held-out 20%)",
        model=BIAS_MODEL,
        notes="0=Left, 1=Center, 2=Right | Expected F1 >0.70",
        y_true=y_true,
        y_pred=y_pred,
    )

except Exception as e:
    print(f"  ⚠️  Political bias benchmark failed: {e}")


# MBIB baseline — full mode only
if not QUICK and pipe_bias is not None:
    try:
        print("\n📊 POLITICAL BIAS [Baseline] — MBIB political_bias")
        print("  Loading MBIB split: political_bias...")

        mbib = load_dataset(
            "mediabiasgroup/MBIB",
            "political_bias",
            split="test",
        )

        texts = mbib["text"][:1000]
        raw_labels = mbib["label"][:1000]

        # Remap: Center→Unbiased(0), Left+Right→Biased(1)
        y_true = [
            0 if str(lbl) in ("Center", "1") else 1
            for lbl in raw_labels
        ]

        preds_raw = run_inference(pipe_bias, texts)
        y_pred = [0 if map_pred(p) == 1 else 1 for p in preds_raw]

        print_results(
            title="POLITICAL BIAS — MBIB political_bias (baseline)",
            model=BIAS_MODEL,
            notes="Remap: Left+Right→Biased, Center→Unbiased | Ceiling ~50-60%",
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
    print("\n📊 GENERAL BIAS — BABE (Expert-Annotated)")
    print("  Loading BABE test split...")

    babe = load_dataset("mediabiasgroup/BABE", split="test")

    texts = babe["text"]
    raw_labels = babe["label"]

    if N_SAMPLES:
        np.random.seed(42)
        indices = np.random.choice(len(texts), N_SAMPLES, replace=False)
        texts = [texts[i] for i in indices]
        raw_labels = [raw_labels[i] for i in indices]

    # BABE labels: 0=Non-biased, 1=Biased (stored as int)
    # Guard handles string variants defensively
    def parse_babe_label(lbl) -> int:
        if isinstance(lbl, int):
            return lbl
        s = str(lbl).strip().lower()
        return 1 if s in ("1", "biased") else 0

    y_true = [parse_babe_label(lbl) for lbl in raw_labels]
    print(f"  Loaded {len(texts)} samples")
    print(f"  Label distribution: {dict(Counter(y_true))}")

    pipe_general = load_pipeline(GENERAL_BIAS_MODEL)
    preds_raw = run_inference(pipe_general, texts)

    # Local pipeline: LABEL_0=BIASED→1, LABEL_1=NEUTRAL→0
    def remap_general(label: str) -> int:
        u = label.upper()
        if "BIASED" in u and "UN" not in u:
            return 1
        if u in ("LABEL_0", "0"):
            return 1
        return 0

    y_pred = [remap_general(p) for p in preds_raw]

    print_results(
        title="GENERAL BIAS — BABE (Expert-Annotated News Sentences)",
        model=GENERAL_BIAS_MODEL,
        notes=(
            "Remap: BIASED/LABEL_0→1, NEUTRAL/LABEL_1→0 "
            "| Best reported macro F1: 0.804"
        ),
        y_true=y_true,
        y_pred=y_pred,
    )

except Exception as e:
    print(f"  ⚠️  General bias (BABE) benchmark failed: {e}")


print(f"\n{'='*70}")
print("✅  EVALUATION COMPLETE")
print(f"{'='*70}\n")
