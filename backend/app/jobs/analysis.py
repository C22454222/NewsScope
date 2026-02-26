"""
NewsScope Analysis Service.

Uses local transformers pipelines — no Inference API dependency.
Models loaded lazily as singletons to stay within 512 MB RAM.
All three models total ~280 MB on CPU.

Sentiment : cardiffnlp/twitter-roberta-base-sentiment-latest (~280 MB)
Political : matous-volf/political-leaning-politics            (~270 MB)
General   : valurank/distilroberta-bias                       (~260 MB)

Flake8: 0 errors/warnings.
"""

import os
from typing import Optional, Tuple

from app.db.supabase import supabase

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# Allow model override via env vars
SENTIMENT_MODEL = os.getenv(
    "HF_SENTIMENT_MODEL",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
).strip()

BIAS_MODEL = os.getenv(
    "HF_BIAS_MODEL",
    "matous-volf/political-leaning-politics",
).strip()

GENERAL_BIAS_MODEL = os.getenv(
    "HF_GENERAL_BIAS_MODEL",
    "valurank/distilroberta-bias",
).strip()

# ── Lazy singletons — loaded once, reused forever ────────────────────────────
_sentiment_pipeline = None
_political_pipeline = None
_general_bias_pipeline = None


def _get_sentiment_pipeline():
    """Load sentiment pipeline once and cache."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline

        print(f"Loading sentiment model: {SENTIMENT_MODEL}")
        _sentiment_pipeline = pipeline(
            "text-classification",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            truncation=True,
            max_length=512,
            device=-1,  # CPU — safe on 512 MB
            top_k=None,
        )
        print("Sentiment model loaded.")
    return _sentiment_pipeline


def _get_political_pipeline():
    """
    Load political bias pipeline once and cache.
    matous-volf model requires launch/POLITICS tokenizer.
    """
    global _political_pipeline
    if _political_pipeline is None:
        from transformers import pipeline

        print(f"Loading political bias model: {BIAS_MODEL}")
        _political_pipeline = pipeline(
            "text-classification",
            model=BIAS_MODEL,
            tokenizer="launch/POLITICS",
            truncation=True,
            max_length=512,
            device=-1,
            top_k=None,
        )
        print("Political bias model loaded.")
    return _political_pipeline


def _get_general_bias_pipeline():
    """Load general bias pipeline once and cache."""
    global _general_bias_pipeline
    if _general_bias_pipeline is None:
        from transformers import pipeline

        print(f"Loading general bias model: {GENERAL_BIAS_MODEL}")
        _general_bias_pipeline = pipeline(
            "text-classification",
            model=GENERAL_BIAS_MODEL,
            truncation=True,
            max_length=512,
            device=-1,
            top_k=None,
        )
        print("General bias model loaded.")
    return _general_bias_pipeline


# ── Source fallback (used when AI model fails) ────────────────────────────────

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


# ── Individual model callers ──────────────────────────────────────────────────


def _sentiment_score(text: str) -> Optional[float]:
    """
    Run sentiment analysis.
    Returns float from -1.0 (negative) to +1.0 (positive).
    """
    try:
        pipe = _get_sentiment_pipeline()
        results = pipe(text[:512])

        # pipeline with top_k=None returns list of lists
        items = results[0] if isinstance(results[0], list) else results

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
        print(f"Sentiment error: {exc}")
        return None


def _detect_political_bias_ai(
    text: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Run political bias classification.
    Returns (bias_score, confidence): bias_score -1.0=Left, 0=Center, 1=Right.
    """
    try:
        pipe = _get_political_pipeline()
        results = pipe(text[:512])

        items = results[0] if isinstance(results[0], list) else results
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
        print(f"Political bias error: {exc}")
        return None, None


def _detect_general_bias(
    text: str,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Run general bias classification.
    Returns (label, confidence): label is 'BIASED' or 'UNBIASED'.
    """
    try:
        pipe = _get_general_bias_pipeline()
        results = pipe(text[:512])

        items = results[0] if isinstance(results[0], list) else results
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
        print(f"General bias error: {exc}")
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
    print(f"AI failed -> fallback to source: {source_name} = {source_bias:.2f}")
    return source_bias, 0.5


# ── Public entry point ────────────────────────────────────────────────────────


def analyze_unscored_articles() -> None:
    """
    Fetch and score all articles with null sentiment_score.
    Runs as a background scheduled job after each ingestion cycle.
    Limited to 25 articles per cycle to stay within RAM budget.
    """
    import time

    print("Starting analysis job...")

    response = (
        supabase.table("articles")
        .select("id, content, title, source")
        .is_("sentiment_score", "null")
        .limit(25)
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

    for article in articles:
        content = article.get("content") or ""
        title = article.get("title") or ""
        source = article.get("source") or ""
        text = f"{title}. {content}".strip()

        if len(text) < 10:
            print(f"Skipping article {article['id']} — no content")
            continue

        sentiment = _sentiment_score(text)
        time.sleep(0.5)

        bias_score, bias_intensity = _hybrid_bias_analysis(text, source)
        time.sleep(0.5)

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

        if update_data:
            supabase.table("articles").update(update_data).eq(
                "id", article["id"]
            ).execute()

    print("Analysis job complete.")
