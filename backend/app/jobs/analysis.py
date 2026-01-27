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


def _detect_if_biased(text: str):
    """
    Step 1: Use AI to detect IF article is biased.
    Returns: True (biased), False (unbiased), or None (error).
    """
    try:
        results = _call_classification(BIAS_MODEL, text)
        if not results:
            return None

        for item in results:
            label = item.label.upper()
            score = item.score

            # Check if model says "Biased" with high confidence
            if 'BIASED' in label and 'NON' not in label:
                # If >60% confident it's biased, return True
                return score > 0.6
            elif 'NON-BIASED' in label or 'NEUTRAL' in label:
                # If >60% confident it's unbiased, return False
                return score < 0.4

        return None

    except Exception as e:
        print(f"Bias detection error: {e}")
        return None


def _get_source_political_leaning(source_name: str):
    """
    Get source's known political leaning from database.
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


def _hybrid_bias_score(text: str, source_name: str):
    """
    Hybrid bias detection (article-level + source-level).

    Step 1: AI analyzes article text to detect IF it's biased
    Step 2a: If biased → map to source's political leaning
    Step 2b: If unbiased → return Center (0.0)

    This combines RoBERTa-based bias detection with known
    source ratings (AllSides methodology).
    """
    # Step 1: Check if THIS specific article is biased
    is_biased = _detect_if_biased(text)

    if is_biased is None:
        # AI failed, fallback to source rating
        print(f"   AI failed, using source rating for {source_name}")
        return _get_source_political_leaning(source_name)

    if is_biased:
        # Article IS biased → use source's known leaning
        leaning = _get_source_political_leaning(source_name)
        print(
            f"   Article is BIASED → "
            f"mapped to {source_name} leaning: {leaning}"
        )
        return leaning
    else:
        # Article is NOT biased → return Center
        print("   Article is UNBIASED → returning Center (0.0)")
        return 0.0


def analyze_unscored_articles():
    """
    Analyze articles using hybrid approach:
    - Sentiment: DistilBERT AI analysis (Sanh et al., 2019)
    - Bias: Article-level detection + source mapping

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

    print(f"Analyzing {len(articles)} articles (hybrid method)...")

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
        bias = _hybrid_bias_score(text, source)

        print(
            f"Article {article['id']}: "
            f"Sentiment={sentiment}, Bias={bias} ({source})"
        )

        update_data = {}
        if sentiment is not None:
            update_data["sentiment_score"] = sentiment
        if bias is not None:
            update_data["bias_score"] = bias

        if update_data:
            supabase.table("articles").update(
                update_data
            ).eq("id", article["id"]).execute()

    print("Analysis job complete.")
