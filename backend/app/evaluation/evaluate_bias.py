"""
NewsScope Model Benchmark — all three production models.

Sentiment : distilbert/distilbert-base-uncased-finetuned-sst-2-english
            Dataset : SST-2 GLUE validation                 (~0.91 F1)
Political : facebook/bart-large-mnli (zero-shot NLI)
            Dataset : custom balanced L/C/R eval set        (~0.60+ F1)
General   : valurank/distilroberta-bias
            Dataset : BABE expert-annotated                 (~0.69 F1)

Run locally only — never imported by FastAPI or deployed to Render.

Usage:
    python benchmark.py           # quick mode (300 samples max)
    python benchmark.py --full    # full datasets

Flake8: 0 errors/warnings.
"""

import sys
import warnings
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from transformers import pipeline

warnings.filterwarnings("ignore")

QUICK = "--full" not in sys.argv
N_SAMPLES = 300 if QUICK else None

SENTIMENT_MODEL = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
BIAS_MODEL = "facebook/bart-large-mnli"
GENERAL_BIAS_MODEL = "valurank/distilroberta-bias"

CANDIDATE_LABELS = ["left-wing", "centrist", "right-wing"]

ZERO_SHOT_MAP = {"left-wing": 0, "centrist": 1, "right-wing": 2}

print(f"\n{'=' * 70}")
print("NewsScope Model Benchmark")
print(f"Mode     : {'QUICK (~5 min)' if QUICK else 'FULL (~30 min)'}")
print(f"Political: {BIAS_MODEL} (zero-shot — matches production)")
print(f"{'=' * 70}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────


def print_results(
    title: str,
    model: str,
    notes: str,
    y_true: list,
    y_pred: list,
) -> None:
    """Print accuracy, weighted F1, confusion matrix, and per-class report."""
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    print(f"\n{'=' * 70}")
    print(title)
    print(f"Model  : {model}")
    print(f"Notes  : {notes}")
    print(f"{'=' * 70}")
    print(f"Accuracy : {accuracy:.4f} | F1 (weighted): {f1_weighted:.4f}")
    print(f"Samples  : {len(y_true)}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Per-class Report:\n{report}")


def load_classifier(model_name: str) -> pipeline:
    """Load a standard text-classification pipeline on CPU."""
    print(f"  Loading {model_name}...")
    return pipeline(
        "text-classification",
        model=model_name,
        device=-1,
        truncation=True,
    )


def load_zero_shot(model_name: str) -> pipeline:
    """Load a zero-shot classification pipeline on CPU."""
    print(f"  Loading {model_name}...")
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1,
    )


def run_inference(pipe, texts: list, batch_size: int = 32) -> list:
    """Run batched text-classification inference, return label strings."""
    results = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        outputs = pipe(batch)
        results.extend([o["label"] for o in outputs])
        done = min(i + batch_size, total)
        if done % 100 == 0 or done == total:
            print(f"    {done}/{total}")
    return results


def run_zero_shot(
    pipe,
    texts: list,
    candidate_labels: list,
    batch_size: int = 16,
) -> list:
    """Run batched zero-shot inference, return top label per text."""
    results = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        outputs = pipe(batch, candidate_labels=candidate_labels)
        if isinstance(outputs, dict):
            outputs = [outputs]
        results.extend([o["labels"][0] for o in outputs])
        done = min(i + batch_size, total)
        if done % 100 == 0 or done == total:
            print(f"    {done}/{total}")
    return results


# ── Eval dataset — balanced 15 per class ─────────────────────────────────────

# fmt: off
EVAL_DATA: list[tuple[str, int]] = [
    # LEFT (0)
    ("The Republican war on voting rights is suppressing turnout.", 0),
    ("Working families are crushed by a system rigged for the wealthy.", 0),
    ("The fossil fuel industry is bankrolling climate denial in Congress.", 0),
    ("Universal basic income would lift millions out of poverty.", 0),
    ("The GOP is criminalising protest to silence dissent.", 0),
    ("Reproductive justice is inseparable from racial and economic justice.", 0),
    ("The wealth gap between Black and white Americans is a legacy of slavery.", 0),
    ("Democrats fight to protect Medicaid from Republican budget cuts.", 0),
    ("The prison industrial complex must be dismantled.", 0),
    ("Expanding the child tax credit is the fastest way to reduce child poverty.", 0),
    ("Corporate greed is driving inflation while workers suffer.", 0),
    ("Systemic racism oppresses Black Americans across every sector.", 0),
    ("Medicare for All would save lives and trillions of dollars.", 0),
    ("The minimum wage must be raised to address the cost-of-living crisis.", 0),
    ("Billionaires doubled their wealth during the pandemic.", 0),
    # CENTER (1)
    ("The Federal Reserve held rates steady at its latest policy meeting.", 1),
    ("Congress is debating a supplemental spending package for foreign aid.", 1),
    ("The administration released a statement on the latest jobs report.", 1),
    ("A bipartisan senate caucus is working on a border security bill.", 1),
    ("The Pentagon released its annual report on global threat assessments.", 1),
    ("The surgeon general issued a new advisory on adolescent mental health.", 1),
    ("Federal prosecutors filed charges in a lobbying investigation.", 1),
    ("The prime minister met with the president at the White House.", 1),
    ("A census report shows population growth concentrated in southern states.", 1),
    ("The energy department released its annual fossil fuel outlook.", 1),
    ("The Federal Reserve raised interest rates to combat inflation.", 1),
    ("Both parties reached a budget compromise averting a shutdown.", 1),
    ("The unemployment rate fell to 3.7 percent per the Bureau of Labor.", 1),
    ("The Supreme Court will hear a case on immigration policy.", 1),
    ("GDP grew by 2.1 percent in the third quarter.", 1),
    # RIGHT (2)
    ("The radical left is determined to transform America into a socialist state.", 2),
    ("Illegal immigration is costing American taxpayers billions every year.", 2),
    ("The mainstream media is actively working to destroy conservatism.", 2),
    ("America's military is being hollowed out by diversity mandates.", 2),
    ("The Democrat party has been taken over by anti-American extremists.", 2),
    ("The globalist agenda is selling out American workers and sovereignty.", 2),
    ("Teachers unions are indoctrinating children with left-wing propaganda.", 2),
    ("The climate agenda is a trojan horse for government control.", 2),
    ("Free market capitalism is the only path to prosperity.", 2),
    ("Biden's open border policy is fuelling a surge in illegal immigration.", 2),
    ("Radical Democrats are pushing socialist policies destroying the economy.", 2),
    ("Critical race theory is indoctrinating children in public schools.", 2),
    ("Government overreach is strangling small businesses.", 2),
    ("Woke ideology is undermining military readiness.", 2),
    ("Tax cuts unleash growth and let Americans keep their money.", 2),
]
# fmt: on


# ── 1. Sentiment — SST-2 GLUE ────────────────────────────────────────────────

print("\n1. SENTIMENT — SST-2 GLUE")
print("-" * 70)

try:
    from datasets import load_dataset

    sst2 = load_dataset("glue", "sst2", split="validation")
    texts_sst2 = list(sst2["sentence"])[:N_SAMPLES]
    true_sst2 = list(sst2["label"])[:N_SAMPLES]

    senti_pipe = load_classifier(SENTIMENT_MODEL)
    raw_sst2 = run_inference(senti_pipe, texts_sst2)
    pred_sst2 = [
        {"NEGATIVE": 0, "POSITIVE": 1}.get(p.upper(), 0)
        for p in raw_sst2
    ]

    print_results(
        "SENTIMENT — SST-2 GLUE",
        SENTIMENT_MODEL,
        "Expected F1 ~0.91-0.93 | target >= 0.79",
        true_sst2,
        pred_sst2,
    )

except Exception as err:
    print(f"  Sentiment benchmark failed: {err}")


# ── 2. Political bias — bart-large-mnli zero-shot ────────────────────────────

print("\n2. POLITICAL BIAS — bart-large-mnli zero-shot (PRODUCTION MODEL)")
print("-" * 70)

try:
    eval_texts = [t for t, _ in EVAL_DATA]
    eval_labels = [lb for _, lb in EVAL_DATA]

    print(
        f"  Eval set : {len(eval_texts)} samples "
        f"{dict(Counter(eval_labels))}"
    )

    bias_pipe = load_zero_shot(BIAS_MODEL)
    raw_bias = run_zero_shot(bias_pipe, eval_texts, CANDIDATE_LABELS)

    print(f"  Unique predicted labels : {list(set(raw_bias))}")
    pred_bias = [ZERO_SHOT_MAP.get(p, 1) for p in raw_bias]
    print(f"  Predicted distribution  : {dict(Counter(pred_bias))}")

    print_results(
        "POLITICAL BIAS — bart-large-mnli zero-shot (3-CLASS)",
        BIAS_MODEL,
        "0=Left, 1=Center, 2=Right | zero-shot NLI | target >= 0.60",
        eval_labels,
        pred_bias,
    )

except Exception as err:
    print(f"  Political bias benchmark failed: {err}")


# ── 3. General bias — BABE ───────────────────────────────────────────────────

print("\n3. GENERAL BIAS — BABE (Expert-Annotated)")
print("-" * 70)

try:
    from datasets import load_dataset

    babe = load_dataset("mediabiasgroup/BABE", split="test")
    texts_babe = list(babe["text"])[:N_SAMPLES]
    raw_babe_labels = list(babe["label"])[:N_SAMPLES]

    def parse_babe_label(val) -> int:
        """Return 1 for biased, 0 for unbiased/neutral."""
        if isinstance(val, int):
            return val
        return (
            1 if str(val).strip().lower() in ("1", "biased") else 0
        )

    true_babe = [parse_babe_label(v) for v in raw_babe_labels]
    print(f"  Samples           : {len(true_babe)}")
    print(f"  Label distribution: {dict(Counter(true_babe))}")

    general_pipe = load_classifier(GENERAL_BIAS_MODEL)
    raw_babe_preds = run_inference(general_pipe, texts_babe)
    print(f"  Unique predicted labels: {list(set(raw_babe_preds))}")

    def remap_babe(label: str) -> int:
        """Return 1 for BIASED, 0 for NEUTRAL/UNBIASED."""
        upper = label.strip().upper()
        if upper in ("NEUTRAL", "UNBIASED"):
            return 0
        if "BIASED" in upper and "UN" not in upper and "NON" not in upper:
            return 1
        return 0

    pred_babe = [remap_babe(p) for p in raw_babe_preds]
    print(f"  Predicted distribution: {dict(Counter(pred_babe))}")

    print_results(
        "GENERAL BIAS — BABE (Expert-Annotated)",
        GENERAL_BIAS_MODEL,
        "0=NEUTRAL, 1=BIASED | target >= 0.69",
        true_babe,
        pred_babe,
    )

except Exception as err:
    print(f"  General bias benchmark failed: {err}")


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("BENCHMARK SUMMARY — NFR6 targets")
print(f"{'=' * 70}")
print(
    f"  Sentiment : SST-2 GLUE "
    f"({N_SAMPLES or 'full'} samples) — target F1 >= 0.79"
)
print(
    "  Political : custom balanced L/C/R eval (45 samples)"
    " — target F1 >= 0.60 (zero-shot)"
)
print(
    f"  General   : BABE expert-annotated "
    f"({N_SAMPLES or 'full'} samples) — target F1 >= 0.69"
)
print(f"{'=' * 70}\n")
