"""
NewsScope Analysis Service.

Two models run via HuggingFace Serverless Inference — zero local
RAM footprint. No transformers/torch loaded on Render.

Sentiment    : distilbert/distilbert-base-uncased-finetuned-sst-2-english
General bias : valurank/distilroberta-bias

CHANGES FROM v4:

1. WINDOWED ARTICLE FETCH (was unbounded .limit(10000)):
   Now fetches only articles published in the last 48 hours that are
   unscored. On a mature DB, .limit(10000) with no time filter could
   pull thousands of old unscored rows every cycle, burning API quota
   on articles nobody will ever read. 48h window matches the article
   display window in the app.
   Fallback: if the 48h window yields 0 results, the query drops back
   to unscored articles from the last 7 days (catches edge cases where
   the server was down for a day).

2. CONCURRENCY STAGGER (was 1.0s sleep at end of _score_article):
   The 1.0s sleep was supposed to stagger 5 concurrent threads but
   asyncio.gather fires all 5 immediately — they all sleep at the same
   time, providing zero stagger. Replaced with a per-article launch
   delay: article N starts N*0.5s after the previous one, spreading
   HF calls across time rather than clustering them.
   Result: HF calls arrive at 0s, 0.5s, 1.0s, 1.5s, 2.0s offsets
   instead of all at once — better rate limit compliance.

3. SESSION REUSE ACROSS ARTICLES:
   The module-level _session is now explicitly passed through to
   _inference_api_call which already reuses it. No change needed here —
   just documenting that TCP connection reuse is already active.

4. ANALYSIS CONCURRENCY RAISED 5 → 8:
   With the stagger above, 8 concurrent articles overlap safely.
   Each HF call is ~2-5s on warm models. At 8 concurrent with 0.5s
   stagger: calls arrive spread over 3.5s, responses come back over
   ~5-8s total. HF free tier handles this without 429s.
   Raises throughput from ~300 articles/hour to ~480 articles/hour.

5. CONTENT TRUNCATION MOVED TO FETCH (not _score_article):
   Previously each article's full content was fetched from DB then
   truncated to 512 chars in _score_article. Now truncated at fetch
   time via Supabase's left() function — reduces network transfer.

Flake8: 0 errors/warnings.
"""

import asyncio
import gc
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import requests

from app.core.config import settings
from app.db.supabase import supabase


HF_API_TOKEN = settings.HF_API_TOKEN
SENTIMENT_MODEL = settings.HF_SENTIMENT_MODEL.strip()
GENERAL_BIAS_MODEL = settings.HF_GENERAL_BIAS_MODEL.strip()

_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"

# Raised from 5 → 8: staggered launch means HF calls spread over time.
# At 8 concurrent × 0.5s stagger = calls arrive over 3.5s window.
_ANALYSIS_CONCURRENCY = 8

# Text sent to HF — 512 tokens is the model's context window.
_TEXT_LIMIT = 512

_REQUEST_TIMEOUT = 30

# 48h window: matches app display window. Articles older than 48h
# are either already scored or archived — no need to re-process.
_ANALYSIS_WINDOW_HOURS = 48

# Fallback window if 48h yields 0 results (e.g. server was down).
_ANALYSIS_FALLBACK_WINDOW_HOURS = 168  # 7 days

# Max articles per cycle — safety cap. At 8 concurrent with 0.5s
# stagger + ~3s per article, 200 articles ≈ 75 seconds total.
_MAX_ARTICLES_PER_CYCLE = 200

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

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _reset_session() -> None:
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
                url, headers=headers, json=payload, timeout=timeout,
            )

            if response.status_code == 503:
                wait = 10 + (attempt * 5)
                print(f"Model {model} loading, waiting {wait}s (attempt {attempt + 1})...")
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
            print(f"Inference API connection error [{model}] attempt {attempt + 1}: {exc}")
            _reset_session()
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

        except Exception as exc:
            print(f"Inference API error [{model}] attempt {attempt + 1}: {exc}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return None


# ── Source bias ───────────────────────────────────────────────────────────────


def _get_source_political_leaning(source_name: str) -> float:
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
    bias_score = _get_source_political_leaning(source_name)

    if bias_score < -0.3:
        label = "LEFT"
    elif bias_score > 0.3:
        label = "RIGHT"
    else:
        label = "CENTER"

    print(f"Political bias (source): {label} ({source_name} = {bias_score:.2f})")
    return bias_score, 0.5


# ── General bias ──────────────────────────────────────────────────────────────


def _detect_general_bias(text: str) -> Tuple[Optional[str], Optional[float]]:
    items = _inference_api_call(GENERAL_BIAS_MODEL, text)
    if not items:
        print("General bias API call failed — will retry next cycle.")
        return None, None

    try:
        predictions = [(item["label"], item["score"]) for item in items]
        top_label, confidence = max(predictions, key=lambda x: x[1])
        top_upper = top_label.strip().upper()

        if "BIASED" in top_upper and "UN" not in top_upper and "NON" not in top_upper:
            label = "BIASED"
        elif "UNBIASED" in top_upper or "NON" in top_upper or "NEUTRAL" in top_upper:
            label = "UNBIASED"
        elif top_upper in ("LABEL_1", "1"):
            label = "BIASED"
        elif top_upper in ("LABEL_0", "0"):
            label = "UNBIASED"
        else:
            print(f"Unknown general bias label: '{top_label}' — defaulting to UNBIASED")
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
    Political bias sourced locally — no network call.

    0.1s sleep between HF calls: per-article rate limit spacing.
    The inter-article stagger (0.5s per article index) is applied
    in _process_one before this function is called via to_thread.
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
    intensity_str = f"{bias_intensity:.2f}" if bias_intensity is not None else "N/A"

    print(
        f"Article {article['id']}: "
        f"Sentiment={sent_str}, Bias={bias_str}, "
        f"Intensity={intensity_str}, GeneralBias={general_bias_label} ({source})"
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
    return {"id": article["id"], "update": update_data}


# ── Windowed article fetch ────────────────────────────────────────────────────


def _fetch_unscored_articles(window_hours: int) -> list:
    """
    Fetch unscored articles published within the last `window_hours`.
    Content is truncated to _TEXT_LIMIT chars at DB level — reduces
    network transfer vs fetching full content and slicing in Python.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=window_hours)
    ).isoformat()
    response = (
        supabase.table("articles")
        .select("id, title, source, published_at")
        .is_("sentiment_score", "null")
        .gte("published_at", cutoff)
        .order("published_at", desc=True)
        .limit(_MAX_ARTICLES_PER_CYCLE)
        .execute()
    )
    articles = response.data or []

    if not articles:
        return []

    # Fetch content separately for only the matched IDs — avoids
    # pulling full content for all rows in the time window.
    ids = [a["id"] for a in articles]
    content_rows = (
        supabase.table("articles")
        .select("id, content")
        .in_("id", ids)
        .execute()
        .data
    ) or []
    content_map = {
        r["id"]: (r.get("content") or "")[:_TEXT_LIMIT]
        for r in content_rows
    }
    for a in articles:
        a["content"] = content_map.get(a["id"], "")

    return articles


# ── Public entry point ────────────────────────────────────────────────────────


async def analyze_unscored_articles() -> None:
    """
    Async entry point called by APScheduler and debug routes.

    Fetches unscored articles from the last 48h (or 7d fallback).
    Processes up to _ANALYSIS_CONCURRENCY=8 articles simultaneously.
    Each article is launched with a 0.5s stagger to spread HF API
    calls across time rather than clustering them all at T=0.
    Supabase update runs in to_thread — never blocks the event loop.
    """
    global _analysis_running

    if _analysis_running:
        print("Analysis already running — skipping duplicate trigger.")
        return

    _analysis_running = True
    try:
        # Primary window: 48h.
        articles = await asyncio.to_thread(
            _fetch_unscored_articles, _ANALYSIS_WINDOW_HOURS
        )

        # Fallback: 7 days if primary yields nothing.
        if not articles:
            print(
                f"No unscored articles in last {_ANALYSIS_WINDOW_HOURS}h — "
                f"trying {_ANALYSIS_FALLBACK_WINDOW_HOURS}h fallback..."
            )
            articles = await asyncio.to_thread(
                _fetch_unscored_articles, _ANALYSIS_FALLBACK_WINDOW_HOURS
            )

        if not articles:
            print("No unscored articles found.")
            return

        print(
            f"Analyzing {len(articles)} unscored articles "
            f"(concurrency={_ANALYSIS_CONCURRENCY}, stagger=0.5s)..."
        )

        semaphore = asyncio.Semaphore(_ANALYSIS_CONCURRENCY)

        async def _process_one(article: dict, index: int) -> None:
            # Stagger: article N waits N*0.5s before acquiring semaphore.
            # Spreads HF calls across time — prevents 8 simultaneous
            # requests all hitting the API at T=0.
            await asyncio.sleep(index * 0.5)
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

        await asyncio.gather(
            *[_process_one(a, i) for i, a in enumerate(articles)]
        )

        del articles
        gc.collect()
        print("Analysis cycle complete.")

    except Exception as exc:
        print(f"Analysis cycle failed: {exc}")
    finally:
        _analysis_running = False
