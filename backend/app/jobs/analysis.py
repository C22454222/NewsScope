"""
NewsScope Analysis Service.

All three models run via HuggingFace Serverless Inference — zero local
RAM footprint. No transformers/torch loaded on Render.

Sentiment      : distilbert/distilbert-base-uncased-finetuned-sst-2-english
Political bias : matous-volf/political-leaning-politics
General bias   : valurank/distilroberta-bias

Flake8: 0 errors/warnings.
"""

import os
import time
import asyncio
from typing import Optional, Tuple

import requests

from app.db.supabase import supabase

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

SENTIMENT_MODEL = os.getenv(
    "HF_SENTIMENT_MODEL",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
).strip()

BIAS_MODEL = os.getenv(
    "HF_BIAS_MODEL",
    "facebook/bart-large-mnli",
).strip()

GENERAL_BIAS_MODEL = os.getenv(
    "HF_GENERAL_BIAS_MODEL",
    "valurank/distilroberta-bias",
).strip()

# HuggingFace Serverless Inference — replaces deprecated api-inference endpoint
_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"

# ── Concurrency guard — prevents overlapping analysis runs ───────────────────
_analysis_running = False

# ── Batch size: 1 article per trigger — survives Render rotations ─────────────
_BATCH_SIZE = 1

# ── Source fallback map ───────────────────────────────────────────────────────
_SOURCE_BIAS_MAP = {
    "CNN": -1.0,
    "The Guardian": -1.0,
    "NPR": -0.5,
    "The Independent": -0.5,
    "BBC News": 0.0,
    "RTÉ News": 0.0,
    "The Irish Times": 0.0,
    "Euronews": 0.0,
    "Sky News": 0.5,
    "Politico Europe": 0.5,
    "Fox News": 1.0,
    "GB News": 1.0,
}

_POLITICAL_LABEL_MAP = {
    "LABEL_0": -1.0,
    "LABEL_1": 0.0,
    "LABEL_2": 1.0,
    "LEFT": -1.0,
    "CENTER": 0.0,
    "RIGHT": 1.0,
    "0": -1.0,
    "1": 0.0,
    "2": 1.0,
}


# ── Inference API helper ──────────────────────────────────────────────────────


def _inference_api_call(
    model: str,
    text: str,
    retries: int = 3,
) -> Optional[list]:
    """
    POST to HuggingFace Serverless Inference with retry + warm-up handling.
    Returns raw list of label/score dicts, or None on failure.
    """
    if not HF_API_TOKEN:
        print("HF_API_TOKEN not set — skipping Inference API call.")
        return None

    url = f"{_HF_API_BASE}/{model}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text[:512]}

    for attempt in range(retries):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=30
            )

            if response.status_code == 503:
                wait = 10 + (attempt * 5)
                print(
                    f"Model {model} loading, "
                    f"waiting {wait}s (attempt {attempt + 1})..."
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and result:
                if isinstance(result[0], list):
                    return result[0]
                return result

        except Exception as exc:
            print(
                f"Inference API error [{model}] "
                f"attempt {attempt + 1}: {exc}"
            )
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return None


# ── Source fallback ───────────────────────────────────────────────────────────


def _get_source_political_leaning(source_name: str) -> float:
    """Retrieve source bias from DB, fall back to hardcoded map."""
    try:
        response = (
            supabase.table("sources")
            .select("bias_rating")
            .eq("name", source_name)
            .single()
            .execute()
        )
        rating = (response.data or {}).get("bias_rating")
        db_map = {
            "Left": -1.0,
            "Center-Left": -0.5,
            "Center": 0.0,
            "Center-Right": 0.5,
            "Right": 1.0,
        }
        if rating and rating in db_map:
            return db_map[rating]
    except Exception:
        pass
    return _SOURCE_BIAS_MAP.get(source_name, 0.0)


# ── Sentiment — HuggingFace Serverless Inference ─────────────────────────────


def _sentiment_score(text: str) -> Optional[float]:
    """
    Run sentiment analysis via HuggingFace Serverless Inference.
    Returns float from -1.0 (negative) to +1.0 (positive).
    Returns None on failure — article retried on next cycle.
    """
    items = _inference_api_call(SENTIMENT_MODEL, text)
    if not items:
        print("Sentiment API call failed — will retry next cycle.")
        return None

    try:
        sentiment_map: dict = {}
        for item in items:
            label = item["label"].lower()
            score = item["score"]
            if "negative" in label:
                sentiment_map["negative"] = score
            elif "neutral" in label:
                sentiment_map["neutral"] = score
            elif "positive" in label:
                sentiment_map["positive"] = score

        neg = sentiment_map.get("negative", 0.0)
        pos = sentiment_map.get("positive", 0.0)
        return round(pos - neg, 4)

    except Exception as exc:
        print(f"Sentiment parse error: {exc}")
        return None


# ── Political bias — HuggingFace Serverless Inference ────────────────────────


def _detect_political_bias_ai(
    text: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Run political bias classification via HuggingFace Serverless Inference.
    Returns (bias_score, confidence): -1.0=Left, 0=Center, 1=Right.
    """
    items = _inference_api_call(BIAS_MODEL, text)
    if not items:
        print("Political bias API call failed — will retry next cycle.")
        return None, None

    try:
        predictions = [(item["label"], item["score"]) for item in items]
        top_label, confidence = max(predictions, key=lambda x: x[1])

        top_norm = top_label.strip()
        bias_score = _POLITICAL_LABEL_MAP.get(
            top_norm, _POLITICAL_LABEL_MAP.get(top_norm.upper(), 0.0)
        )

        if bias_score < -0.3:
            label_str = "LEFT"
        elif bias_score > 0.3:
            label_str = "RIGHT"
        else:
            label_str = "CENTER"

        print(
            f"Political bias: {label_str} "
            f"(score={bias_score:.2f}, confidence={confidence:.2%})"
        )
        return round(bias_score, 4), round(confidence, 4)

    except Exception as exc:
        print(f"Political bias parse error: {exc}")
        return None, None


def _hybrid_bias_analysis(
    text: str,
    source_name: str,
) -> Tuple[float, float]:
    """
    Political bias with graceful source fallback.
    Returns (bias_score, bias_intensity).
    """
    ai_bias, ai_confidence = _detect_political_bias_ai(text)

    if ai_bias is not None and ai_confidence is not None:
        if ai_bias < -0.3:
            label = "LEFT"
        elif ai_bias > 0.3:
            label = "RIGHT"
        else:
            label = "CENTER"

        print(
            f"Article AI result: {label} "
            f"(score={ai_bias:.2f}, confidence={ai_confidence:.2%}, "
            f"intensity={ai_confidence:.2f})"
        )
        return ai_bias, ai_confidence

    source_bias = _get_source_political_leaning(source_name)
    print(
        f"AI failed -> fallback to source: "
        f"{source_name} = {source_bias:.2f}"
    )
    return source_bias, 0.5


# ── General bias — HuggingFace Serverless Inference ──────────────────────────


def _detect_general_bias(
    text: str,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Run general bias classification via HuggingFace Serverless Inference.
    Returns (label, confidence): label is 'BIASED' or 'UNBIASED'.
    """
    items = _inference_api_call(GENERAL_BIAS_MODEL, text)
    if not items:
        print("General bias API call failed — will retry next cycle.")
        return None, None

    try:
        print(f"  General bias raw results: {items}")
        predictions = [(item["label"], item["score"]) for item in items]
        top_label, confidence = max(predictions, key=lambda x: x[1])
        top_upper = top_label.upper()

        if "BIASED" in top_upper and "UN" not in top_upper:
            label = "BIASED"
        elif (
            "UNBIASED" in top_upper or "NON" in top_upper or "NEUTRAL" in top_upper
        ):
            label = "UNBIASED"
        elif top_upper in ("LABEL_1", "1"):
            label = "BIASED"
        elif top_upper in ("LABEL_0", "0"):
            label = "UNBIASED"
        else:
            print(
                f"Unknown general bias label: '{top_label}' "
                f"— defaulting to UNBIASED"
            )
            label = "UNBIASED"

        print(f"General bias: {label} (confidence={confidence:.2%})")
        return label, round(confidence, 4)

    except Exception as exc:
        print(f"General bias parse error: {exc}")
        return None, None


# ── Sync worker — runs in thread pool, never blocks the event loop ────────────


def _score_article(article: dict) -> dict:
    """Score a single article synchronously. Called from thread pool."""
    content = article.get("content") or ""
    title = article.get("title") or ""
    source = article.get("source") or ""
    text = f"{title}. {content}".strip()

    if len(text) < 10:
        print(f"Skipping article {article['id']} — no content")
        return {}

    sentiment = _sentiment_score(text)
    time.sleep(0.3)

    bias_score, bias_intensity = _hybrid_bias_analysis(text, source)
    time.sleep(0.3)

    general_bias_label, general_bias_score = _detect_general_bias(text)

    sent_str = f"{sentiment:.3f}" if sentiment is not None else "N/A"
    bias_str = f"{bias_score:.2f}" if bias_score is not None else "N/A"
    intensity_str = (
        f"{bias_intensity:.2f}" if bias_intensity is not None else "N/A"
    )

    print(
        f"Article {article['id']}: "
        f"Sentiment={sent_str}, "
        f"Bias={bias_str}, "
        f"Intensity={intensity_str}, "
        f"GeneralBias={general_bias_label} "
        f"({source})"
    )

    update_data: dict = {}
    if sentiment is not None:
        update_data["sentiment_score"] = sentiment
    if bias_score is not None:
        update_data["bias_score"] = bias_score
    if bias_intensity is not None:
        update_data["bias_intensity"] = bias_intensity
    if general_bias_label is not None:
        update_data["general_bias"] = general_bias_label
    if general_bias_score is not None:
        update_data["general_bias_score"] = general_bias_score

    return {"id": article["id"], "update": update_data}


def _run_analysis_sync() -> None:
    """
    Blocking analysis worker — must be called via asyncio.to_thread().
    Fetches 1 unscored article per call — fully resumable across
    Render instance rotations. State persisted in Supabase after
    every article so no work is lost on rotation.
    """
    print("Starting analysis job...")

    response = (
        supabase.table("articles")
        .select("id, content, title, source")
        .is_("sentiment_score", "null")
        .limit(_BATCH_SIZE)
        .execute()
    )
    articles = response.data

    if not articles:
        print("No unscored articles found.")
        return

    print(
        f"Analyzing {len(articles)} articles "
        f"(hybrid methodology + general bias)..."
    )

    for batch_start in range(0, len(articles), _BATCH_SIZE):
        batch = articles[batch_start: batch_start + _BATCH_SIZE]
        for article in batch:
            result = _score_article(article)
            if result.get("update"):
                supabase.table("articles").update(
                    result["update"]
                ).eq("id", result["id"]).execute()

    print("Analysis job complete.")


# ── Public entry point ────────────────────────────────────────────────────────


async def analyze_unscored_articles() -> None:
    """
    Async entry point called by APScheduler and debug routes.
    Offloads all CPU/IO-bound work to a thread so the event loop
    (and health checks) are never blocked.
    Skips silently if a run is already in progress.
    """
    global _analysis_running

    if _analysis_running:
        print("Analysis already running — skipping duplicate trigger.")
        return

    _analysis_running = True
    try:
        await asyncio.to_thread(_run_analysis_sync)
    finally:
        _analysis_running = False
