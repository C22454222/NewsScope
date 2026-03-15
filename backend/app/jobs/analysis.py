"""
NewsScope Analysis Service.

Two models run via HuggingFace Serverless Inference — zero local
RAM footprint. No transformers/torch loaded on Render.

Sentiment    : distilbert/distilbert-base-uncased-finetuned-sst-2-english
               Confirmed on HF Inference API. 3.63M downloads.
General bias : valurank/distilroberta-bias
               Confirmed on HF Inference API. 3.79k downloads.

Political bias : SOURCE-LEVEL ONLY (no AI model call).
                 mDeBERTa-v3-base-mnli-xnli removed — returned confidence
                 below 60% on 100% of observed cycles while burning a 45s
                 socket dwell per article. Bias scores are still written
                 to DB from _SOURCE_BIAS_MAP or the sources table.

MEMORY / THROUGHPUT (v4 — spaCy removed):
- _BATCH_SIZE removed entirely: all unscored articles fetched and
  processed per cycle instead of 2 at a time. With spaCy gone, HF
  Serverless Inference is the only RAM cost — each call holds one
  JSON response buffer (~2-10KB). No meaningful RAM pressure.
- _ANALYSIS_CONCURRENCY = 5: five articles processed simultaneously
  via asyncio.gather + Semaphore. Each article still makes its 2 HF
  calls sequentially inside a thread to respect rate limits, but 5
  articles overlap in parallel — ~5x throughput vs serial processing.
- _score_article runs in asyncio.to_thread per article so blocking
  requests calls never touch the event loop.
- Supabase update per article also runs in to_thread — immediate
  persistence means no work is lost on instance rotation.
- _REQUEST_TIMEOUT remains 30s — sufficient for warm HF calls.
- Session reset on connection errors retained — flushes stale buffers.

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

# Five articles processed concurrently — each runs 2 sequential HF calls
# inside its own thread. Semaphore prevents unbounded parallelism that
# would exhaust HF free-tier rate limits.
_ANALYSIS_CONCURRENCY = 5

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

        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
        ) as exc:
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
    Score a single article synchronously. Called via asyncio.to_thread.
    Makes 2 HF API calls: sentiment + general bias.
    Political bias is sourced locally — no network call.

    0.1s sleep between HF calls respects per-article rate limits.
    1.0s sleep at end staggers concurrent threads slightly to prevent
    all 5 workers from hitting HF simultaneously on the same tick.
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

    gc.collect()
    time.sleep(1.0)
    return {"id": article["id"], "update": update_data}


# ── Public entry point ────────────────────────────────────────────────────────


async def analyze_unscored_articles() -> None:
    """
    Async entry point called by APScheduler and debug routes.

    Fetches ALL unscored articles in one query — no batch cap.
    Processes up to _ANALYSIS_CONCURRENCY=5 articles simultaneously
    via asyncio.gather + Semaphore. Each article's _score_article call
    runs in its own thread via asyncio.to_thread so blocking requests
    calls never touch the event loop. Supabase update also runs in
    to_thread for the same reason.

    Skips silently if a run is already in progress.
    """
    global _analysis_running

    if _analysis_running:
        print("Analysis already running — skipping duplicate trigger.")
        return

    _analysis_running = True
    try:
        response = (
            supabase.table("articles")
            .select("id, title, source, content, published_at")
            .is_("sentiment_score", "null")
            .limit(10000)
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
            f"Analyzing {len(articles)} unscored articles "
            f"(concurrency={_ANALYSIS_CONCURRENCY})..."
        )

        semaphore = asyncio.Semaphore(_ANALYSIS_CONCURRENCY)

        async def _process_one(article: dict) -> None:
            async with semaphore:
                result = await asyncio.to_thread(_score_article, article)
                if result.get("update"):
                    await asyncio.to_thread(
                        lambda r=result: (
                            supabase.table("articles")
                            .update(r["update"])
                            .eq("id", r["id"])
                            .execute()
                        )
                    )

        await asyncio.gather(*[_process_one(a) for a in articles])

        del articles
        gc.collect()
        print("Analysis cycle complete.")

    except Exception as exc:
        print(f"Analysis cycle failed: {exc}")
    finally:
        _analysis_running = False
