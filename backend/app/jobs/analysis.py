"""
NewsScope Analysis Service.

Three models score every article — all run on free HuggingFace Spaces,
zero Inference API credits consumed:

  Sentiment      : distilbert-base-uncased-finetuned-sst-2-english
                   Space: c22454222-sentiment.hf.space
                   Endpoint: /gradio_api/call/lambda
                   Returns POSITIVE / NEGATIVE scores → mapped to [-1, +1]

  General bias   : valurank/distilroberta-bias
                   Space: c22454222-general-bias.hf.space
                   Endpoint: /gradio_api/call/lambda
                   Returns BIASED / UNBIASED

  Political bias : C22454222/political-bias-roberta (fine-tuned RoBERTa)
                   Space: c22454222-political-bias-api.hf.space
                   Endpoint: /gradio_api/call/classify_bias
                   Returns LEFT / CENTER / RIGHT + confidence score
                   Trained on 37,554 AllSides articles. 87.3% acc / F1.

All three use the Gradio 5.x two-step SSE protocol:
  Step 1 — POST {base}/gradio_api/call/{fn}   → {"event_id": "<uuid>"}
  Step 2 — GET  {base}/gradio_api/call/{fn}/{event_id}  → SSE stream

No timeouts set — Render free tier does not impose request timeouts and
HF Spaces cold starts can take 30-90s. Full article text is sent to
every Space; each applies truncation=True internally so the underlying
model only sees its token limit (512 for all three).

Flake8: 0 errors/warnings.
"""

import asyncio
import gc
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

import requests

from app.core.config import settings
from app.db.supabase import supabase


POLITICAL_BIAS_SPACE = settings.HF_POLITICAL_BIAS_SPACE.strip()
SENTIMENT_SPACE = settings.HF_SENTIMENT_SPACE.strip()
GENERAL_BIAS_SPACE = settings.HF_GENERAL_BIAS_SPACE.strip()

# 8 concurrent articles × 0.5s stagger = calls spread over 3.5s window.
_ANALYSIS_CONCURRENCY = 8

# Full article text cap — Spaces apply truncation=True internally.
_TEXT_LIMIT_ARTICLE = 8000

# 48h primary window. 7d fallback if primary yields nothing.
_ANALYSIS_WINDOW_HOURS = 48
_ANALYSIS_FALLBACK_WINDOW_HOURS = 168

# Safety cap per cycle.
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


# ── Generic Gradio Space caller ───────────────────────────────────────────────


def _spaces_call(
    space_base: str,
    fn_name: str,
    text: str,
) -> Optional[Any]:
    """
    Call any Gradio 5.x Space via the two-step SSE protocol.

    Step 1: POST {base}/gradio_api/call/{fn_name}
            Body: {"data": ["<text>"]}
            Returns: {"event_id": "<uuid>"}

    Step 2: GET {base}/gradio_api/call/{fn_name}/{event_id}
            Streams SSE lines. The "data:" line holds a JSON list —
            payload[0] is what the Space function returned.

    No timeout set — HF Spaces cold starts can legitimately take
    30-90s and Render does not impose request time limits.
    Returns payload[0] on success, None on any error.
    """
    if not space_base:
        print(f"Space URL not configured for {fn_name} — skipping.")
        return None

    base = space_base.rstrip("/")

    try:
        session = _get_session()

        # Step 1: submit job, receive event_id.
        r1 = session.post(
            f"{base}/gradio_api/call/{fn_name}",
            headers={"Content-Type": "application/json"},
            json={"data": [text]},
        )
        r1.raise_for_status()
        event_id = r1.json()["event_id"]
        r1.close()

        # Step 2: stream SSE until result line.
        r2 = session.get(
            f"{base}/gradio_api/call/{fn_name}/{event_id}",
            stream=True,
        )
        r2.raise_for_status()

        for raw_line in r2.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            payload = json.loads(raw_line[len("data:"):].strip())
            r2.close()
            return payload[0]

    except requests.exceptions.ConnectionError as exc:
        print(f"Space connection error [{fn_name}]: {exc}")
        _reset_session()
    except Exception as exc:
        print(f"Space call error [{fn_name}]: {exc}")

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
    """
    Score sentiment via the sentiment Space (/gradio_api/call/lambda).
    Returns float in [-1.0, +1.0]: positive - negative scores.
    Returns None if the Space call fails — retried next cycle.
    """
    items = _spaces_call(SENTIMENT_SPACE, "lambda", text)
    if not items:
        print("Sentiment Space call failed — will retry next cycle.")
        return None

    try:
        sentiment_map: dict = {}
        for item in items:
            label = item["label"].lower()
            score = item["score"]
            if "negative" in label:
                sentiment_map["negative"] = score
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
    Classify BIASED / UNBIASED via the general bias Space
    (/gradio_api/call/lambda).
    Returns (label, confidence) or (None, None) on failure.
    """
    items = _spaces_call(GENERAL_BIAS_SPACE, "lambda", text)
    if not items:
        print("General bias Space call failed — will retry next cycle.")
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
            "UNBIASED" in top_upper or "UN" in top_upper or "NON" in top_upper or "NEUTRAL" in top_upper
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


# ── Article-level political bias (fine-tuned RoBERTa via HF Space) ────────────


def _detect_political_bias(
    title: str,
    content: str,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Classify LEFT / CENTER / RIGHT via the political bias Space
    (/gradio_api/call/classify_bias).
    Full title + content sent — Space truncates to 512 tokens internally.
    Returns (label, confidence) or (None, None) on failure.
    """
    text = f"{title}. {content}".strip()

    if len(text) < 10:
        print("Political bias: text too short — skipping.")
        return None, None

    result = _spaces_call(POLITICAL_BIAS_SPACE, "classify_bias", text)
    if not result:
        print("Political bias Space call failed — will retry next cycle.")
        return None, None

    try:
        label = result["label"].upper()
        score = round(result["score"], 4)
        print(
            f"Political bias (article): {label} "
            f"(confidence={score:.2%})"
        )
        return label, score

    except Exception as exc:
        print(f"Political bias parse error: {exc}")
        return None, None


# ── Single article scorer ─────────────────────────────────────────────────────


def _score_article(article: dict) -> dict:
    """
    Score a single article synchronously. Called via asyncio.to_thread.

    Full article text (title + content, capped at _TEXT_LIMIT_ARTICLE)
    sent to all three Spaces. Each Space applies truncation=True so
    the model only sees its token limit (512 for all three).

    Three remote Space calls: sentiment, general bias, political bias.
    Source-level political leaning is a local DB lookup — no network.
    0.1s sleep between Space calls to avoid bursting the same Space.
    """
    content = (article.get("content") or "")
    title = article.get("title") or ""
    source = article.get("source") or ""

    full_text = f"{title}. {content}".strip()[:_TEXT_LIMIT_ARTICLE]

    if len(full_text) < 10:
        print(f"Skipping article {article['id']} — no content")
        return {}

    sentiment = _sentiment_score(full_text)
    time.sleep(0.1)

    bias_score, bias_intensity = _get_political_bias(source)
    time.sleep(0.1)

    general_bias_label, general_bias_score = _detect_general_bias(full_text)
    time.sleep(0.1)

    political_bias_label, political_bias_score = _detect_political_bias(
        title, content
    )

    sent_str = f"{sentiment:.3f}" if sentiment is not None else "N/A"
    bias_str = f"{bias_score:.2f}" if bias_score is not None else "N/A"
    intensity_str = (
        f"{bias_intensity:.2f}" if bias_intensity is not None else "N/A"
    )

    print(
        f"Article {article['id']}: "
        f"Sentiment={sent_str}, Bias={bias_str}, "
        f"Intensity={intensity_str}, "
        f"GeneralBias={general_bias_label}, "
        f"PoliticalBias={political_bias_label} ({source})"
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
    if political_bias_label is not None:
        update_data["political_bias"] = political_bias_label
    if political_bias_score is not None:
        update_data["political_bias_score"] = political_bias_score

    gc.collect()
    return {"id": article["id"], "update": update_data}


# ── Windowed article fetch ────────────────────────────────────────────────────


def _fetch_unscored_articles(window_hours: int) -> list:
    """
    Fetch unscored articles published within the last `window_hours`.
    Full content fetched — all three models need the article body.
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

    ids = [a["id"] for a in articles]
    content_rows = (
        supabase.table("articles")
        .select("id, content")
        .in_("id", ids)
        .execute()
        .data
    ) or []
    content_map = {
        r["id"]: (r.get("content") or "")
        for r in content_rows
    }
    for a in articles:
        a["content"] = content_map.get(a["id"], "")

    return articles


# ── Public entry point ────────────────────────────────────────────────────────


async def analyze_unscored_articles() -> None:
    """
    Async entry point called by APScheduler and debug routes.

    Fetches unscored articles from the last 48h (7d fallback).
    Processes up to _ANALYSIS_CONCURRENCY=8 articles simultaneously.
    Each article staggered by 0.5s × index to spread Space calls.
    Supabase updates run via to_thread — event loop never blocked.
    """
    global _analysis_running

    if _analysis_running:
        print("Analysis already running — skipping duplicate trigger.")
        return

    _analysis_running = True
    try:
        articles = await asyncio.to_thread(
            _fetch_unscored_articles, _ANALYSIS_WINDOW_HOURS
        )

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
