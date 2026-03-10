"""
NewsScope Analysis Service.

Two models run via HuggingFace Serverless Inference — zero local
RAM footprint. No transformers/torch loaded on Render.

Sentiment    : distilbert/distilbert-base-uncased-finetuned-sst-2-english
               Confirmed on HF Inference API. 3.63M downloads.
General bias : valurank/distilroberta-bias
               Confirmed on HF Inference API. 3.79k downloads.

Political bias : SOURCE-LEVEL ONLY (no AI model call).
                 mDeBERTa-v3-base-mnli-xnli was called on every article
                 but returned confidence below 60% threshold on 100% of
                 observed cycles, falling back to source every time while
                 burning a 45s socket dwell and HF response buffer RAM
                 per article. Removed entirely until a better model is
                 found. Bias scores are still written to DB — sourced
                 from _SOURCE_BIAS_MAP or the sources table as before.

MEMORY FIX SUMMARY (v3):
- AI political bias call removed: saves 1 HF API call + up to 45s
  socket dwell per article. Was contributing 0% useful results.
- HF calls per article: 3 → 2 (sentiment + general bias only).
- _REQUEST_TIMEOUT 90→30s: faster timeout on remaining two models.
- _session reset on connection errors: flushes stale socket buffers.
- _BATCH_SIZE 3→2: halves peak concurrent HF response RAM per cycle.

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
GENERAL_BIAS_MODEL = settings.HF_GENERAL_BIAS_MODEL.strip()

_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"

# 2 articles per batch — each holds 2 concurrent HF responses in RAM.
_BATCH_SIZE = 2

_TEXT_LIMIT = 512

# 30s timeout — sufficient for warm distilbert/distilroberta calls.
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

# Module-level session — created once, reused across calls.
# Recreated on connection errors to flush stale socket buffers.
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Return the module-level requests session, creating it if needed."""
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _reset_session() -> None:
    """
    Close and discard the current session.
    Called after connection errors to flush stale socket buffers
    that would otherwise accumulate in RAM across retries.
    """
    global _session
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
        _session = None


# ── Inference API helper ──────────────────────────────────────────────────────


def _inference_api_call(
    model: str,
    text: str,
    retries: int = 2,
    timeout: int = _REQUEST_TIMEOUT,
) -> Optional[list]:
    """
    POST to HuggingFace Serverless Inference with retry + warm-up handling.
    Returns raw list of label/score dicts, or None on failure.

    Session is reset on connection errors (ReadTimeout, ConnectionError)
    to prevent stale socket buffers from accumulating in RAM.
    """
    if not HF_API_TOKEN:
        print("HF_API_TOKEN not set — skipping Inference API call.")
        return None

    url = f"{_HF_API_BASE}/{model}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text[:_TEXT_LIMIT]}

    for attempt in range(retries):
        session = _get_session()
        try:
            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if response.status_code == 503:
                wait = 10 + (attempt * 5)
                print(
                    f"Model {model} loading, "
                    f"waiting {wait}s (attempt {attempt + 1})..."
                )
                response.close()
                time.sleep(wait)
                continue

            response.raise_for_status()
            result = response.json()
            response.close()

            if isinstance(result, list) and result:
                return result[0] if isinstance(result[0], list) else result

        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as exc:
            print(
                f"Inference API connection error [{model}] "
                f"attempt {attempt + 1}: {exc}"
            )
            _reset_session()
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

        except Exception as exc:
            print(
                f"Inference API error [{model}] "
                f"attempt {attempt + 1}: {exc}"
            )
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return None


# ── Source bias ───────────────────────────────────────────────────────────────


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


# ── Political bias (source-level) ─────────────────────────────────────────────


def _get_political_bias(source_name: str) -> Tuple[float, float]:
    """
    Return political bias from source-level data only. No HF API call.

    mDeBERTa-v3-base-mnli-xnli was removed after logging showed it
    returned confidence below 60% on every observed article, burning
    a 45s socket dwell per article while contributing nothing.

    Returns (bias_score, bias_intensity):
      bias_score    : -1.0 (Left) to +1.0 (Right)
      bias_intensity: fixed 0.5 — reflects source-level certainty
    """
    bias_score = _get_source_political_leaning(source_name)

    if bias_score < -0.3:
        label = "LEFT"
    elif bias_score > 0.3:
        label = "RIGHT"
    else:
        label = "CENTER"

    print(
        f"Political bias (source): {label} "
        f"({source_name} = {bias_score:.2f})"
    )
    return bias_score, 0.5


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
    Makes 2 HF API calls: sentiment + general bias.
    Political bias is sourced locally — no network call.
    """
    content = (article.get("content") or "")[:_TEXT_LIMIT]
    title = article.get("title") or ""
    source = article.get("source") or ""
    text = f"{title}. {content}".strip()

    if len(text) < 10:
        print(f"Skipping article {article['id']} — no content")
        return {}

    sentiment = _sentiment_score(text)
    time.sleep(0.1)

    bias_score, bias_intensity = _get_political_bias(source)
    time.sleep(0.1)

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
    1s sleep between articles prevents HF rate limiting across batch.
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
        "(sentiment + source bias + general bias)..."
    )

    for article in articles:
        result = _score_article(article)
        if result.get("update"):
            supabase.table("articles").update(
                result["update"]
            ).eq("id", result["id"]).execute()
        gc.collect()
        time.sleep(1.0)

    del articles
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
