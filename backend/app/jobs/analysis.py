"""
NewsScope Analysis Service.

Three models run for article analysis:
  Sentiment      : distilbert/distilbert-base-uncased-finetuned-sst-2-english
                   via HuggingFace Serverless Inference API
  General bias   : valurank/distilroberta-bias
                   via HuggingFace Serverless Inference API
  Political bias : C22454222/political-bias-roberta
                   via HuggingFace Spaces (fine-tuned RoBERTa)
                   Trained on ramybaly/Article-Bias-Prediction (37,554
                   AllSides articles). 87.3% accuracy, 87.3% macro F1.
                   Returns LEFT / CENTER / RIGHT with confidence score.

Zero local RAM footprint on Render — all inference runs remotely.
No transformers/torch loaded on Render.

Political bias input: title + full article body concatenated, truncated
to 512 tokens. RoBERTa tokenizer handles subword tokenisation so 512
tokens ≈ 350-400 words of actual article content — enough to capture
the political framing of any news article.

CHANGES FROM v5:

1. GRADIO 5 TWO-STEP SSE API (replaces single POST /run/predict):
   Gradio 5.x removed the synchronous POST /run/predict endpoint.
   Posting to the Space root or /run/predict now hits the SvelteKit
   SSR layer which returns 405 "No form actions exist for this page".
   The correct protocol is:
     Step 1 — POST /call/predict  →  {"event_id": "abc123"}
     Step 2 — GET  /call/predict/{event_id}  →  SSE stream
   The SSE stream emits lines; the result is on the "data:" line that
   follows the "event: complete" line. Parsed as JSON list — [0] gives
   the {"label": ..., "score": ...} dict returned by classify_bias().
   No new dependencies — uses the existing requests.Session.

2. CONFIG URL FIXED (base URL, no /run/predict suffix):
   HF_POLITICAL_BIAS_SPACE is now the base URL only. _detect_political_bias
   appends /call/predict and /call/predict/{event_id} itself. Previously
   the default in config.py included /run/predict which is now dead.

3. import json ADDED:
   Required to parse the SSE data line from the GET stream.

Flake8: 0 errors/warnings.
"""

import asyncio
import gc
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import requests

from app.core.config import settings
from app.db.supabase import supabase


HF_API_TOKEN = settings.HF_API_TOKEN
SENTIMENT_MODEL = settings.HF_SENTIMENT_MODEL.strip()
GENERAL_BIAS_MODEL = settings.HF_GENERAL_BIAS_MODEL.strip()
POLITICAL_BIAS_SPACE = settings.HF_POLITICAL_BIAS_SPACE.strip()

_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"

# Raised from 5 → 8: staggered launch means HF calls spread over time.
# At 8 concurrent × 0.5s stagger = calls arrive over 3.5s window.
_ANALYSIS_CONCURRENCY = 8

# Text limit for sentiment and general bias models — 512 chars.
_TEXT_LIMIT = 512

# Text limit for political bias model — 512 tokens (RoBERTa tokenizer).
# Title + body concatenated. 512 tokens ≈ 350-400 words of article text.
_TEXT_LIMIT_POLITICAL = 1500

_REQUEST_TIMEOUT = 30

# Longer timeout for HF Spaces — cold start can take 30-60s.
_SPACES_TIMEOUT = 60

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
                print(
                    f"Model {model} loading, waiting {wait}s "
                    f"(attempt {attempt + 1})..."
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
                f"Inference API error [{model}] attempt {attempt + 1}: {exc}"
            )
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
                f"— defaulting to UNBIASED"
            )
            label = "UNBIASED"

        print(f"General bias: {label} (confidence={confidence:.2%})")
        return label, round(confidence, 4)

    except Exception as exc:
        print(f"General bias parse error: {exc}")
        return None, None


# ── Article-level political bias (fine-tuned RoBERTa via HF Spaces) ───────────


def _detect_political_bias(
    title: str,
    content: str,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Classify political bias at article level using fine-tuned RoBERTa.

    Model: C22454222/political-bias-roberta
    Hosted: HuggingFace Spaces (c22454222-political-bias-api.hf.space)
    Trained on: ramybaly/Article-Bias-Prediction (37,554 AllSides articles)
    Performance: 87.3% accuracy, 87.3% macro F1 (LEFT/CENTER/RIGHT)

    Gradio 5.x API protocol (two-step SSE):
      Step 1 — POST {base}/call/predict
               Body: {"data": ["<article text>"]}
               Returns: {"event_id": "<uuid>"}
      Step 2 — GET  {base}/call/predict/{event_id}
               Streams SSE lines until "event: complete".
               The "data:" line after complete contains a JSON list;
               index [0] is the dict returned by classify_bias():
               {"label": "LEFT"|"CENTER"|"RIGHT", "score": 0.xx}

    No new dependencies — uses the existing requests.Session.
    Timeout _SPACES_TIMEOUT (60s) covers HF Spaces cold starts.
    No HF_API_TOKEN required — Space is public.
    """
    if not POLITICAL_BIAS_SPACE:
        print("HF_POLITICAL_BIAS_SPACE not set — skipping political bias.")
        return None, None

    text = f"{title}. {content}".strip()
    text = text[:_TEXT_LIMIT_POLITICAL]

    if len(text) < 10:
        print("Political bias: text too short — skipping.")
        return None, None

    base = POLITICAL_BIAS_SPACE.rstrip("/")

    try:
        session = _get_session()

        # Step 1: POST to /call/predict → receive event_id.
        r1 = session.post(
            f"{base}/call/predict",
            headers={"Content-Type": "application/json"},
            json={"data": [text]},
            timeout=_SPACES_TIMEOUT,
        )
        r1.raise_for_status()
        event_id = r1.json()["event_id"]

        # Step 2: GET SSE stream → read until result line.
        r2 = session.get(
            f"{base}/call/predict/{event_id}",
            stream=True,
            timeout=_SPACES_TIMEOUT,
        )
        r2.raise_for_status()

        for raw_line in r2.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            payload = json.loads(raw_line[len("data:"):].strip())
            # payload is a list: [{"label": "RIGHT", "score": 0.91}]
            result = payload[0]
            label = result["label"].upper()
            score = round(result["score"], 4)
            print(
                f"Political bias (article): {label} "
                f"(confidence={score:.2%})"
            )
            return label, score

    except requests.exceptions.Timeout:
        print("Political bias: HF Space timed out — will retry next cycle.")
        return None, None
    except Exception as exc:
        print(f"Political bias error: {exc}")
        return None, None

    return None, None


# ── Single article scorer ─────────────────────────────────────────────────────


def _score_article(article: dict) -> dict:
    """
    Score a single article synchronously. Called via asyncio.to_thread.
    Makes 3 remote calls: sentiment + general bias (HF Inference API)
    + political bias (HF Spaces via two-step SSE).
    Source-level political leaning sourced locally — no network call.

    0.1s sleep between HF calls: per-article rate limit spacing.
    The inter-article stagger (0.5s per article index) is applied
    in _process_one before this function is called via to_thread.
    """
    content = (article.get("content") or "")
    title = article.get("title") or ""
    source = article.get("source") or ""

    # Short combined text for sentiment + general bias (512 char limit).
    text_short = f"{title}. {content[:_TEXT_LIMIT]}".strip()

    if len(text_short) < 10:
        print(f"Skipping article {article['id']} — no content")
        return {}

    sentiment = _sentiment_score(text_short)
    time.sleep(0.1)

    bias_score, bias_intensity = _get_political_bias(source)
    time.sleep(0.1)

    general_bias_label, general_bias_score = _detect_general_bias(text_short)
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
    Content fetched in full — political bias model needs full body.
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

    # Fetch full content for matched IDs.
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
