"""
NewsScope Bias Explainability Service.

Uses LIME (Local Interpretable Model-Agnostic Explanations) to identify
the words most responsible for an article's political bias classification.

Unlike the standard LIME setup, the classifier here is a remote
HuggingFace Space called via _spaces_call — not a local pipeline.
_NUM_SAMPLES is set to 50 (vs typical 300) to keep the ~50 HTTP calls
to the Space within a 10–20s window acceptable for background scoring.

This runs after the main _score_article call in analysis.py so it
never blocks the primary sentiment/bias pipeline.

Flake8: 0 errors/warnings.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from lime.lime_text import LimeTextExplainer

from app.core.config import settings


_POLITICAL_BIAS_SPACE = settings.HF_POLITICAL_BIAS_SPACE.strip()

_LABEL_MAP: Dict[str, int] = {
    "left": 0,
    "center": 1,
    "centre": 1,
    "right": 2,
}
_CLASS_NAMES: List[str] = ["left", "center", "right"]

# Reduced from 300 — each sample is a remote HTTP call to an HF Space.
# 50 samples gives stable top-word attribution in ~10–20s.
_NUM_SAMPLES = 50
_NUM_FEATURES = 6

_explainer = LimeTextExplainer(class_names=_CLASS_NAMES)


def _spaces_call_simple(text: str) -> Optional[Dict[str, Any]]:
    """
    Minimal inline Space caller for LIME — avoids circular import
    by not importing from analysis.py. Uses requests directly.
    Identical protocol to _spaces_call in analysis.py.
    """
    import json
    import requests

    base = _POLITICAL_BIAS_SPACE.rstrip("/")
    try:
        r1 = requests.post(
            f"{base}/gradio_api/call/classify_bias",
            headers={"Content-Type": "application/json"},
            json={"data": [text]},
        )
        r1.raise_for_status()
        event_id = r1.json().get("event_id")
        if not event_id:
            return None

        r2 = requests.get(
            f"{base}/gradio_api/call/classify_bias/{event_id}",
            stream=True,
        )
        r2.raise_for_status()
        for raw_line in r2.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            payload = json.loads(raw_line[len("data:"):].strip())
            r2.close()
            result = payload[0]
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    pass
            return result
    except Exception as exc:
        print(f"LIME Space call error: {exc}")
    return None


def _predict_proba(texts: List[str]) -> np.ndarray:
    """
    Maps the political bias Space output to a (n_texts, 3) probability
    matrix required by LimeTextExplainer.
    Falls back to uniform [0.33, 0.33, 0.33] on any error so LIME
    doesn't crash the whole scoring run.
    """
    probs = []
    for text in texts:
        row = [0.33, 0.33, 0.33]
        result = _spaces_call_simple(text[:512])
        if result:
            try:
                label = result["label"].lower()
                score = float(result["score"])
                idx = _LABEL_MAP.get(label)
                if idx is not None:
                    row = [0.0, 0.0, 0.0]
                    row[idx] = score
            except Exception:
                pass
        probs.append(row)
    return np.array(probs)


def explain_bias(
    text: str,
    predicted_label: str,
) -> List[Dict[str, Any]]:
    """
    Run LIME on `text` and return the top words driving the
    `predicted_label` classification.

    Returns a list of dicts sorted by absolute weight descending:
      [
        {"word": "government", "weight": 0.42, "direction": "towards"},
        {"word": "welfare",    "weight": 0.31, "direction": "towards"},
        {"word": "spending",   "weight": 0.18, "direction": "against"},
      ]

    direction "towards" → word pushes toward the predicted label.
    direction "against" → word pushes away from the predicted label.

    Returns [] if text is too short, label is unrecognised, or
    the Space raises an exception — never raises itself.
    """
    if not text or len(text.split()) < 10:
        return []

    label_idx = _LABEL_MAP.get(predicted_label.lower())
    if label_idx is None:
        return []

    try:
        explanation = _explainer.explain_instance(
            text[:1000],
            _predict_proba,
            num_features=_NUM_FEATURES,
            num_samples=_NUM_SAMPLES,
            labels=[label_idx],
        )
        raw = explanation.as_list(label=label_idx)
        return [
            {
                "word": word,
                "weight": round(abs(weight), 4),
                "direction": "towards" if weight > 0 else "against",
            }
            for word, weight in sorted(
                raw, key=lambda x: abs(x[1]), reverse=True
            )
        ]
    except Exception as exc:
        print(f"LIME explanation failed: {exc}")
        return []
