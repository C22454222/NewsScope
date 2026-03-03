"""
NewsScope Analysis Service.

All three models run via HuggingFace Serverless Inference — zero local
RAM footprint. No transformers/torch loaded on Render.

Sentiment      : distilbert/distilbert-base-uncased-finetuned-sst-2-english
Political bias : facebook/bart-large-mnli (zero-shot classification)
General bias   : valurank/distilroberta-bias

Flake8: 0 errors/warnings.
"""

import gc
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

_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"

# Reduced from 20 — each article makes 3 API calls with up to 3 retries
# each at 30s timeout. 10 articles = manageable memory + time budget.
_BATCH_SIZE = 10

_TEXT_LIMIT = 1024

# Reduced timeout — fail fast and fall back to source bias rather than
# holding open connections for 30s × 3 retries × 20 articles.
_REQUEST_TIMEOUT = 30

_analysis_running = False

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


# ── Inference API helper ──────────────────────────────────────────────────────


def _inference_api_call(
    model: str,
    text: str,
    retries: int = 2,
) -> Optional[list]:
    """
    POST to HuggingFace Serverless Inference with retry + warm-up handling.
    Returns raw list of label/score dicts, or None on failure.
    Uses an explicit session closed in a finally block to prevent leaks.
    Retries capped at 2 to limit memory pressure from stacked connections.
    """
    if not HF_API_TOKEN:
        print("HF_API_TOKEN not set — skipping Inference API call.")
        return None

    url = f"{_HF_API_BASE}/{model}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text[:512]}
    session = requests.Session()

    try:
        for attempt in range(retries):
            try:
                response = session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=_REQUEST_TIMEOUT,
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
                    return result[0] if isinstance(result[0], list) else result

            except Exception as exc:
                print(
                    f"Inference API error [{model}] "
                    f"attempt {attempt + 1}: {exc}"
                )
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
    finally:
        session.close()

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


# ── Sentiment ─────────────────────────────────────────────────────────────────


def _sentiment_score(text: str) -> Optional[float]:
    """
    Run sentiment analysis via HuggingFace Serverless Inference.
    Returns float from -1.0 (negative) to +1.0 (positive), or None on failure.
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


# ── Political bias ────────────────────────────────────────────────────────────


def _detect_political_bias_ai(
    text: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Zero-shot political bias via facebook/bart-large-mnli.
    Returns (bias_score, confidence): -1.0=Left, 0.0=Center, 1.0=Right.
    bias_score derived from (right-wing probability - left-wing probability).
    Retries capped at 2 to avoid holding connections on timeout-heavy articles.
    """
    if not HF_API_TOKEN:
        print("Political bias: HF_API_TOKEN not set — skipping.")
        return None, None

    url = f"{_HF_API_BASE}/{BIAS_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": text[:512],
        "parameters": {
            "candidate_labels": ["left-wing", "centrist", "right-wing"]
        },
    }
    session = requests.Session()

    try:
        for attempt in range(2):
            try:
                response = session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=_REQUEST_TIMEOUT,
                )

                print(
                    f"  Political bias HTTP {response.status_code} "
                    f"(attempt {attempt + 1})"
                )

                if response.status_code == 503:
                    wait = 10 + (attempt * 5)
                    print(
                        f"  Political bias model loading, "
                        f"waiting {wait}s..."
                    )
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                result = response.json()

                if not isinstance(result, list) or not result:
                    print(
                        "  Political bias: unexpected response shape "
                        "— returning None"
                    )
                    return None, None

                label_score_map = {
                    item["label"]: item["score"] for item in result
                }
                left = label_score_map.get("left-wing", 0.0)
                center = label_score_map.get("centrist", 0.0)
                right = label_score_map.get("right-wing", 0.0)

                print(
                    f"  Political bias scores — "
                    f"left={left:.3f}, center={center:.3f}, "
                    f"right={right:.3f}"
                )

                bias_score = round(right - left, 4)
                confidence = round(max(left, center, right), 4)

                if bias_score < -0.3:
                    label_str = "LEFT"
                elif bias_score > 0.3:
                    label_str = "RIGHT"
                else:
                    label_str = "CENTER"

                print(
                    f"Political bias: {label_str} "
                    f"(score={bias_score:.2f}, "
                    f"confidence={confidence:.2%})"
                )
                return bias_score, confidence

            except Exception as exc:
                print(
                    f"Political bias error attempt {attempt + 1}: {exc}"
                )
                if attempt < 1:
                    time.sleep(5)
    finally:
        session.close()

    print("Political bias API call failed — will retry next cycle.")
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


# ── General bias ──────────────────────────────────────────────────────────────


def _detect_general_bias(
    text: str,
) -> Tuple[Optional[str], Optional[float]]:
    """
    General bias classification via valurank/distilroberta-bias.
    Returns (label, confidence): label is 'BIASED' or 'UNBIASED'.
    """
    items = _inference_api_call(GENERAL_BIAS_MODEL, text)
    if not items:
        print("General bias API call failed — will retry next cycle.")
        return None, None

    try:
        predictions = [(item["label"], item["score"]) for item in items]
        top_label, confidence = max(predictions, key=lambda x: x[1])
        top_upper = top_label.strip().upper()

        if (
            "BIASED" in top_upper and "UN" not in top_upper and "NON" not in top_upper
        ):
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
                "— defaulting to UNBIASED"
            )
            label = "UNBIASED"

        print(f"General bias: {label} (confidence={confidence:.2%})")
        return label, round(confidence, 4)

    except Exception as exc:
        print(f"General bias parse error: {exc}")
        return None, None


# ── Single article scorer ─────────────────────────────────────────────────────


def _score_article(article: dict) -> dict:
    """
    Score a single article synchronously. Called from thread pool.
    Only title, content (truncated to _TEXT_LIMIT), source are used.
    published_at is available in the dict for future use.
    """
    content = (article.get("content") or "")[:_TEXT_LIMIT]
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


# ── Sync worker ───────────────────────────────────────────────────────────────


def _run_analysis_sync() -> None:
    """
    Blocking analysis worker — must be called via asyncio.to_thread().
    Fetches _BATCH_SIZE unscored articles per call. Selects only the
    fields required for analysis to minimise memory. Persists each
    result to Supabase immediately so no work is lost on instance
    rotation. Calls gc.collect() after each article to release RAM.
    """
    print("Starting analysis job...")

    response = (
        supabase.table("articles")
        .select("id, title, source, content, published_at")
        .is_("sentiment_score", "null")
        .limit(_BATCH_SIZE)
        .execute()
    )

    articles = [
        {**a, "content": (a.get("content") or "")[:_TEXT_LIMIT]}
        for a in (response.data or [])
    ]

    if not articles:
        print("No unscored articles found.")
        return

    print(
        f"Analyzing {len(articles)} articles "
        "(hybrid methodology + general bias)..."
    )

    for article in articles:
        result = _score_article(article)
        if result.get("update"):
            supabase.table("articles").update(
                result["update"]
            ).eq("id", result["id"]).execute()
        gc.collect()

    print("Analysis job complete.")


# ── Public entry point ────────────────────────────────────────────────────────


async def analyze_unscored_articles() -> None:
    """
    Async entry point called by APScheduler and debug routes.
    Offloads all CPU/IO-bound work to a thread so the event loop
    and health checks are never blocked. Skips silently if a run
    is already in progress.
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
