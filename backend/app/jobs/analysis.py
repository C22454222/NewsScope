"""
NewsScope analysis service.

Scores every article across three dimensions using free HuggingFace
Spaces -- zero Inference API credits consumed.

Sentiment        : distilbert-base-uncased-finetuned-sst-2-english
                   Space    : c22454222-sentiment.hf.space
                   Endpoint : /gradio_api/call/classify_sentiment
                   Output   : POSITIVE / NEGATIVE scores mapped to [-1, +1]

General bias     : valurank/distilroberta-bias
                   Space    : c22454222-general-bias.hf.space
                   Endpoint : /gradio_api/call/classify_general_bias
                   Output   : BIASED / UNBIASED

Political bias   : C22454222/political-bias-roberta (fine-tuned RoBERTa)
                   Space    : c22454222-political-bias-api.hf.space
                   Endpoint : /gradio_api/call/classify_bias
                   Output   : LEFT / CENTER / RIGHT + confidence score
                   Training : 37,554 AllSides articles, 87.3% acc / F1

All three Spaces use the Gradio 5.x two-step SSE protocol:
  Step 1 -- POST {base}/gradio_api/call/{fn}  returns {"event_id": ""}
  Step 2 -- GET  {base}/gradio_api/call/{fn}/{event_id}  SSE stream

No request timeouts are set because HF Spaces cold starts can take
30-90 seconds and Render free tier does not impose request time limits.

Analysis window is keyed on created_at (ingestion time) rather than
published_at so articles ingested today are always scored regardless of
their original publish date.
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
from app.services.explainability import explain_bias

POLITICAL_BIAS_SPACE = settings.HF_POLITICAL_BIAS_SPACE.strip()
SENTIMENT_SPACE = settings.HF_SENTIMENT_SPACE.strip()
GENERAL_BIAS_SPACE = settings.HF_GENERAL_BIAS_SPACE.strip()

# 8 concurrent articles x 0.5s stagger spreads calls over a 3.5s window.
_ANALYSIS_CONCURRENCY = 8

# Character cap before sending to a Space. Each Space applies
# truncation=True internally so only 512 tokens reach the model.
_TEXT_LIMIT_ARTICLE = 8000

# 7 day window keyed on created_at -- covers ingestion lag where
# articles published weeks ago are ingested today.
_ANALYSIS_WINDOW_HOURS = 168

# Safety cap per regular scheduled cycle. Backfill has no cap.
_MAX_ARTICLES_PER_CYCLE = 200

# Political bias confidence must reach this threshold before LIME
# explainability is run. Below it the classification is too uncertain
# to produce reliable word-level attribution.
_LIME_CONFIDENCE_THRESHOLD = 0.6

# Guard flag prevents overlapping analysis cycles triggered by the scheduler.
_analysis_running = False

# Fallback source-level political leaning used when the sources table
# has no bias_rating entry for a given source name.
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

# Lazily initialised session reused across all Space calls.
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """
    Return the shared requests session, creating it on first call.

    Forces HTTP/1.1 via the Connection header to prevent
    ConnectionTerminated errors from HF Spaces that can drop
    HTTP/2 connections mid-stream.
    """
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Connection": "keep-alive"})
    return _session


def _reset_session() -> None:
    """Close and discard the shared session so the next call creates a fresh one."""
    global _session
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
    _session = None


def _spaces_call(
    space_base: str,
    fn_name: str,
    text: str,
) -> Optional[Any]:
    """
    Call a Gradio 5.x Space using the two-step SSE protocol.

    Step 1: POST {base}/gradio_api/call/{fn_name}
            Body    : {"data": [""]}
            Returns : {"event_id": ""}

    Step 2: GET {base}/gradio_api/call/{fn_name}/{event_id}
            Streams SSE lines. The "data:" line contains a JSON list
            where payload[0] is the value returned by the Space function.

    Handles two payload shapes seen across Gradio versions:
      payload[0] is a list/dict  -- returned as-is
      payload[0] is a JSON string -- parsed before returning

    No timeout is set because HF Spaces cold starts can take 30-90s.
    Returns payload[0] on success or None on any error.
    """
    if not space_base:
        print(f"Space URL not configured for {fn_name} -- skipping.")
        return None

    base = space_base.rstrip("/")

    try:
        session = _get_session()

        # Step 1: submit the job and receive an event_id.
        r1 = session.post(
            f"{base}/gradio_api/call/{fn_name}",
            headers={"Content-Type": "application/json"},
            json={"data": [text]},
        )
        r1.raise_for_status()

        r1_json = r1.json()
        r1.close()

        if not r1_json or "event_id" not in r1_json:
            print(
                f"Space [{fn_name}]: unexpected response shape "
                f"-- {r1_json} -- Space may still be building."
            )
            return None

        event_id = r1_json["event_id"]

        # Step 2: stream SSE lines until the result data line arrives.
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

            result = payload[0]

            # Unwrap JSON strings that Gradio sometimes returns instead
            # of a parsed object -- seen in some Gradio version responses.
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    pass

            return result

    except requests.exceptions.ConnectionError as exc:
        print(f"Space connection error [{fn_name}]: {exc}")
        _reset_session()
    except Exception as exc:
        print(f"Space call error [{fn_name}]: {exc}")

    return None


def _get_source_political_leaning(source_name: str) -> float:
    """
    Look up source political leaning from the Supabase sources table.

    Falls back to _SOURCE_BIAS_MAP if the table has no entry for the
    given source name, and to 0.0 (Centre) if neither matches.
    """
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


def _sentiment_score(text: str) -> Optional[float]:
    """
    Score text sentiment via the sentiment Space.

    Returns a float in [-1.0, +1.0] computed as positive_score minus
    negative_score. Returns None if the Space call fails so the
    article can be retried on the next scheduled cycle.
    """
    items = _spaces_call(SENTIMENT_SPACE, "classify_sentiment", text)
    if not items:
        print("Sentiment Space call failed -- will retry next cycle.")
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


def _get_political_bias(source_name: str) -> Tuple[float, float]:
    """
    Derive source-level political bias from the sources table or fallback map.

    Returns (bias_score, confidence) where bias_score is in [-1.0, +1.0]
    and confidence is fixed at 0.5 for source-level lookups.
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


def _detect_general_bias(
    text: str,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Classify BIASED / UNBIASED via the general bias Space.

    Handles multiple label formats returned by different Gradio versions
    including LABEL_0/LABEL_1 numeric fallbacks. Returns (None, None) if
    the Space call fails so the article can be retried next cycle.
    """
    items = _spaces_call(
        GENERAL_BIAS_SPACE, "classify_general_bias", text
    )
    if not items:
        print("General bias Space call failed -- will retry next cycle.")
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
                "-- defaulting to UNBIASED"
            )
            label = "UNBIASED"

        print(f"General bias: {label} (confidence={confidence:.2%})")
        return label, round(confidence, 4)

    except Exception as exc:
        print(f"General bias parse error: {exc}")
        return None, None


def _detect_political_bias(
    title: str,
    content: str,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Classify LEFT / CENTER / RIGHT via the political bias Space.

    Full title and content are sent; the Space applies truncation=True
    internally so the underlying RoBERTa model only sees 512 tokens.
    Returns (None, None) if the Space call fails.
    """
    text = f"{title}. {content}".strip()

    if len(text) < 10:
        print("Political bias: text too short -- skipping.")
        return None, None

    result = _spaces_call(POLITICAL_BIAS_SPACE, "classify_bias", text)
    if not result:
        print("Political bias Space call failed -- will retry next cycle.")
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


def _score_article(article: dict) -> dict:
    """
    Score a single article synchronously. Called via asyncio.to_thread.

    Runs three remote Space calls in order: sentiment, general bias,
    political bias. Source-level political leaning is a local DB lookup.
    A 0.1 second sleep between Space calls avoids bursting one Space.

    LIME explainability runs only when political bias confidence is at or
    above _LIME_CONFIDENCE_THRESHOLD (0.6). Below that threshold the
    classification is too uncertain to produce reliable attribution.
    LIME makes approximately 50 additional Space calls per article (10-20s).
    """
    content = (article.get("content") or "")
    title = article.get("title") or ""
    source = article.get("source") or ""

    full_text = f"{title}. {content}".strip()[:_TEXT_LIMIT_ARTICLE]

    if len(full_text) < 10:
        print(f"Skipping article {article['id']} -- no content")
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

    # Run LIME only when confidence is high enough for reliable attribution.
    if (
        political_bias_label and political_bias_score is not None and political_bias_score >= _LIME_CONFIDENCE_THRESHOLD
    ):
        bias_explanation = explain_bias(
            text=full_text,
            predicted_label=political_bias_label,
        )
        if bias_explanation:
            update_data["bias_explanation"] = bias_explanation
            print(
                f"LIME explanation: {len(bias_explanation)} words "
                f"for {political_bias_label} "
                f"(confidence={political_bias_score:.2%})"
            )

    gc.collect()
    return {"id": article["id"], "update": update_data}


def _fetch_unscored_articles(window_hours: int) -> list:
    """
    Fetch unscored articles ingested within the last `window_hours`.

    Uses created_at (ingestion time) not published_at so articles
    published weeks ago but ingested today are always included.
    Content is fetched in a second query and merged in to avoid
    Supabase row size limits on the initial select.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=window_hours)
    ).isoformat()
    response = (
        supabase.table("articles")
        .select("id, title, source, published_at")
        .is_("sentiment_score", "null")
        .gte("created_at", cutoff)
        .order("created_at", desc=True)
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


def _fetch_all_unscored_articles() -> list:
    """
    Fetch all unscored articles regardless of age for one-time backfill.

    Pages through the database in batches of 1000 to stay within
    Supabase row limits. Called by backfill_all_articles.
    """
    all_articles: list = []
    page_size = 1000
    offset = 0

    while True:
        response = (
            supabase.table("articles")
            .select("id, title, source, published_at")
            .is_("sentiment_score", "null")
            .order("created_at", desc=True)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = response.data or []
        if not batch:
            break

        ids = [a["id"] for a in batch]
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
        for a in batch:
            a["content"] = content_map.get(a["id"], "")

        all_articles.extend(batch)
        offset += page_size

        if len(batch) < page_size:
            break

    print(f"Backfill: found {len(all_articles)} unscored articles total.")
    return all_articles


async def _run_analysis(articles: list, label: str) -> None:
    """
    Score and persist a list of articles concurrently.

    Shared by analyze_unscored_articles and backfill_all_articles.
    Staggered 0.5s per article to spread Space calls across the window
    defined by _ANALYSIS_CONCURRENCY (8 articles x 0.5s = 3.5s spread).
    """
    print(
        f"{label}: analyzing {len(articles)} articles "
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
    print(f"{label}: complete.")


async def analyze_unscored_articles() -> None:
    """
    Score articles ingested in the last 7 days. Called by APScheduler
    every 5 minutes.

    Uses the created_at window to cover ingestion lag where articles
    published weeks ago are ingested and scored today. A running guard
    prevents overlapping cycles if the previous one has not finished.
    """
    global _analysis_running

    if _analysis_running:
        print("Analysis already running -- skipping duplicate trigger.")
        return

    _analysis_running = True
    try:
        articles = await asyncio.to_thread(
            _fetch_unscored_articles, _ANALYSIS_WINDOW_HOURS
        )

        if not articles:
            print("No unscored articles found in last 7 days.")
            return

        await _run_analysis(articles, "Analysis")

    except Exception as exc:
        print(f"Analysis cycle failed: {exc}")
    finally:
        _analysis_running = False


async def backfill_all_articles() -> None:
    """
    Score all unscored articles in the database regardless of age.

    One-time backfill called via POST /debug/backfill. Pages through
    in batches of 1000 with no article count cap. A running guard
    prevents the backfill from launching during an active cycle.
    """
    global _analysis_running

    if _analysis_running:
        print("Analysis already running -- skipping backfill.")
        return

    _analysis_running = True
    try:
        articles = await asyncio.to_thread(_fetch_all_unscored_articles)

        if not articles:
            print("Backfill: no unscored articles found.")
            return

        await _run_analysis(articles, "Backfill")

    except Exception as exc:
        print(f"Backfill failed: {exc}")
    finally:
        _analysis_running = False
