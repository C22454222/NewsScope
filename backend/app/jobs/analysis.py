# app/jobs/analysis.py
import os
import time
from app.db.supabase import supabase
from huggingface_hub import InferenceClient


HF_API_TOKEN = os.getenv("HF_API_TOKEN")

SENTIMENT_MODEL = os.getenv(
    "HF_SENTIMENT_MODEL",
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
).strip()

BIAS_MODEL = os.getenv(
    "HF_BIAS_MODEL",
    "d4data/bias-detection-model"
).strip()

client = InferenceClient(token=HF_API_TOKEN)


def _call_classification(model: str, text: str):
    """
    Safe wrapper around InferenceClient.text_classification.
    Retries on model loading errors (503).
    """
    truncated_text = text[:512]

    for attempt in range(3):
        try:
            return client.text_classification(
                truncated_text,
                model=model
            )

        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "loading" in error_str.lower():
                print(
                    f"Model {model} loading... "
                    f"waiting 10s (attempt {attempt + 1}/3)"
                )
                time.sleep(10)
                continue

            print(f"Analysis error for {model}: {e}")
            return None

    print(f"Model {model} failed after 3 retries")
    return None


def _sentiment_score(text: str):
    """
    Analyze sentiment using DistilBERT (Sanh et al., 2019).
    Returns float: -1 (negative) to +1 (positive).
    """
    try:
        results = _call_classification(SENTIMENT_MODEL, text)
        if not results:
            return None

        sentiment_map = {}
        for item in results:
            label = item.label.lower()
            score = item.score

            if 'negative' in label:
                sentiment_map['negative'] = score
            elif 'neutral' in label:
                sentiment_map['neutral'] = score
            elif 'positive' in label:
                sentiment_map['positive'] = score

        neg = sentiment_map.get('negative', 0)
        pos = sentiment_map.get('positive', 0)

        return pos - neg

    except Exception as e:
        print(f"Sentiment parsing error: {e}")
        return None


def _get_source_political_leaning(source_name: str):
    """
    Retrieve source's political leaning from database.
    Uses AllSides/Ground News methodology.
    Returns float: -1 (Left) to +1 (Right).
    """
    try:
        response = (
            supabase.table("sources")
            .select("bias_rating")
            .eq("name", source_name)
            .single()
            .execute()
        )

        rating = response.data.get("bias_rating")

        bias_map = {
            "Left": -1.0,
            "Center-Left": -0.5,
            "Center": 0.0,
            "Center-Right": 0.5,
            "Right": 1.0
        }

        return bias_map.get(rating, 0.0)

    except Exception:
        return 0.0


def _detect_article_bias(text: str):
    """
    Article-level bias detection using RoBERTa-based model.
    Returns tuple: (is_biased: bool, intensity: float)

    - is_biased: True if article contains bias, False if neutral
    - intensity: 0.0 (unbiased) to 1.0 (highly biased)
    """
    try:
        results = _call_classification(BIAS_MODEL, text)
        if not results:
            return (None, None)

        biased_score = 0.0
        neutral_score = 0.0

        for item in results:
            label = item.label.upper()
            score = item.score

            if 'BIASED' in label and 'NON' not in label:
                biased_score = score
            elif 'NON-BIASED' in label or 'NEUTRAL' in label:
                neutral_score = score

        # Article is biased if confidence > 60%
        is_biased = biased_score > 0.6

        # Intensity = biased confidence score
        intensity = biased_score if is_biased else (1.0 - neutral_score)

        return (is_biased, intensity)

    except Exception as e:
        print(f"Bias detection error: {e}")
        return (None, None)


def _hybrid_bias_analysis(text: str, source_name: str):
    """
    Hybrid bias detection methodology combining:
    1. Article-level AI bias detection (RoBERTa)
    2. Source-level political leaning (AllSides methodology)

    Returns tuple: (political_direction, bias_intensity)

    Logic:
    - If article is UNBIASED: direction=0.0 (Center), low intensity
    - If article is BIASED: direction=source_leaning, AI intensity
    - If AI fails: fallback to source rating, moderate intensity
    """
    # Step 1: Detect if article is biased (article-level)
    is_biased, intensity = _detect_article_bias(text)

    # Fallback if AI fails
    if is_biased is None:
        direction = _get_source_political_leaning(source_name)
        print(
            f"   AI failed → fallback to source: "
            f"{source_name} = {direction}"
        )
        return (direction, 0.5)

    # Step 2: Determine political direction
    if is_biased:
        # Article IS biased → map to source's political leaning
        direction = _get_source_political_leaning(source_name)
        print(
            f"   BIASED (intensity={intensity:.2f}) → "
            f"{source_name} leaning = {direction}"
        )
        return (direction, intensity)
    else:
        # Article is UNBIASED → center regardless of source
        print(
            f"   UNBIASED (intensity={intensity:.2f}) → "
            f"Center (0.0)"
        )
        return (0.0, intensity)


def analyze_unscored_articles():
    """
    Analyze articles using hybrid transformer-based approach.

    Sentiment Analysis:
      - Model: DistilBERT (Sanh et al., 2019)
      - Method: Article-level sentiment classification
      - Output: sentiment_score (-1 to +1)

    Bias Detection:
      - Model: RoBERTa-based (Liu et al., 2019)
      - Method: Hybrid article-level + source-level
      - Output: bias_score (-1 to +1), bias_intensity (0 to 1)

    Runs every hour at :15 (15 minutes after ingestion).
    """
    print("Starting analysis job...")

    response = (
        supabase.table("articles")
        .select("id, content, title, source")
        .is_("sentiment_score", "null")
        .limit(500)
        .execute()
    )
    articles = response.data

    if not articles:
        print("No unscored articles found.")
        return

    print(
        f"Analyzing {len(articles)} articles "
        f"(hybrid methodology)..."
    )

    for article in articles:
        content = article.get("content") or ""
        title = article.get("title") or ""
        source = article.get("source") or ""

        text = f"{title}. {content}".strip()

        if len(text) < 10:
            print(f"Skipping article {article['id']} - no content")
            continue

        # Sentiment analysis (AI)
        sentiment = _sentiment_score(text)
        time.sleep(1.0)

        # Hybrid bias detection (AI + source mapping)
        direction, intensity = _hybrid_bias_analysis(text, source)

        print(
            f"Article {article['id']}: "
            f"Sentiment={sentiment:.3f}, "
            f"Bias={direction:.2f}, "
            f"Intensity={intensity:.2f} "
            f"({source})"
        )

        # Update database
        update_data = {}
        if sentiment is not None:
            update_data["sentiment_score"] = sentiment
        if direction is not None:
            update_data["bias_score"] = direction
        if intensity is not None:
            update_data["bias_intensity"] = intensity

        if update_data:
            supabase.table("articles").update(
                update_data
            ).eq("id", article["id"]).execute()

    print("Analysis job complete.")
