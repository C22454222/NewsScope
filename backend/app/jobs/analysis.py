"""
NewsScope Analysis Service.

All three models run via HuggingFace Serverless Inference — zero local
RAM footprint. No transformers/torch loaded on Render.

Sentiment      : distilbert/distilbert-base-uncased-finetuned-sst-2-english
                 Confirmed on HF Inference API. 3.63M downloads.
Political bias : facebook/bart-large-mnli
                 Zero-shot NLI — confirmed on HF Inference API. 10M+ downloads.
                 Payload: inputs + parameters.candidate_labels.
                 Response: {"label": "left-wing", "score": 0.86}
                 Single top-prediction dict after list unwrap.
                 Runs on HF servers — zero RAM cost on Render free tier.
General bias   : valurank/distilroberta-bias
                 Confirmed on HF Inference API. 3.79k downloads.

Flake8: 0 errors/warnings.
"""

import asyncio
import gc
import time
from typing import Optional, Tuple

import requests

from app.core.config import settings
from app.db.supabase import supabase


HF_API_TOKEN = settings.HF_API_TOKEN
SENTIMENT_MODEL = settings.HF_SENTIMENT_MODEL.strip()
BIAS_MODEL = settings.HF_BIAS_MODEL.strip()
GENERAL_BIAS_MODEL = settings.HF_GENERAL_BIAS_MODEL.strip()

_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"

# Reduced from 10 — each article holds 3 concurrent HF responses in RAM.
# 3 articles × ~40MB response overhead stays well under 512MB free tier.
_BATCH_SIZE = 3
_TEXT_LIMIT = 512
_REQUEST_TIMEOUT = 60

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

# Module-level session — created once, reused across all calls,
# avoids opening two sessions simultaneously per article.
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Return the module-level requests session, creating it if needed."""
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


# ── Inference API helper ──────────────────────────────────────────────────────


def _inference_api_call(
    model: str,
    text: str,
    retries: int = 2,
) -> Optional[list]:
    """
    POST to HuggingFace Serverless Inference with retry + warm-up handling.
    Returns raw list of label/score dicts, or None on failure.
    Uses the module-level session to avoid per-call session overhead.
    NOTE: Not used for political bias — that model returns a dict, not list.
    """
    if not HF_API_TOKEN:
        print("HF_API_TOKEN not set — skipping Inference API call.")
        return None

    url = f"{_HF_API_BASE}/{model}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text[:_TEXT_LIMIT]}
    session = _get_session()

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
    Returns float from -1.0 (negative) to +1.0 (positive), or None.
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
    Political bias via facebook/bart-large-mnli (zero-shot NLI).

    Confirmed on HF Inference API — 10M+ downloads.
    Runs entirely on HF servers — zero RAM cost on Render free tier.

    Payload: inputs (str) + parameters.candidate_labels (list).
    Response after list unwrap: {"label": "left-wing", "score": 0.86}
    Single top-prediction dict — label mapped to bias_score float.

    Returns (bias_score, confidence): -1.0=Left, 0.0=Center, 1.0=Right.
    """
    if not HF_API_TOKEN:
        print("Political bias: HF_API_TOKEN not set — skipping.")
        return None, None

    url = f"{_HF_API_BASE}/{BIAS_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": text[:_TEXT_LIMIT],
        "parameters": {
            "candidate_labels": ["left-wing", "centrist", "right-wing"],
        },
    }
    session = _get_session()

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

            # HF router wraps response in a list — unwrap to single dict.
            if isinstance(result, list) and result:
                result = result[0]

            if not isinstance(result, dict):
                print(
                    "  Political bias: unexpected response type "
                    f"({type(result).__name__}) — returning None"
                )
                return None, None

            # Response shape: {"label": "left-wing", "score": 0.86}
            label_raw = result.get("label", "")
            score_raw = result.get("score", 0.0)

            if not label_raw:
                print(
                    "  Political bias: empty label — returning None"
                )
                return None, None

            label_lower = label_raw.lower()
            if "left" in label_lower:
                bias_score = round(-score_raw, 4)
            elif "right" in label_lower:
                bias_score = round(score_raw, 4)
            else:
                bias_score = 0.0

            confidence = round(score_raw, 4)

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
            print(f"Political bias error attempt {attempt + 1}: {exc}")
            if attempt < 1:
                time.sleep(5)

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
    Confirmed on HF Inference API. 3.79k downloads.
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
