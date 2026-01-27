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


def _get_source_bias(source_name: str):
    """
    Get political bias from source rating.
    Uses source-level classification similar to AllSides/Ground
    News (as referenced in interim report).
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

        # Map to numerical scale (-1 to +1)
        bias_map = {
            "Left": -1.0,
            "Center-Left": -0.5,
            "Center": 0.0,
            "Center-Right": 0.5,
            "Right": 1.0
        }

        return bias_map.get(rating, 0.0)

    except Exception:
        # Default to center if source not found
        return 0.0


def analyze_unscored_articles():
    """
    Analyze articles using transformer models.
    Sentiment: DistilBERT (Sanh et al., 2019)
    Bias: Source-level ratings (AllSides methodology)
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

    print(f"Analyzing {len(articles)} articles...")

    for article in articles:
        content = article.get("content") or ""
        title = article.get("title") or ""
        source = article.get("source") or ""

        text = f"{title}. {content}".strip()

        if len(text) < 10:
            print(f"Skipping article {article['id']} - no content")
            continue

        # Get sentiment from AI (DistilBERT)
        sentiment = _sentiment_score(text)
        time.sleep(0.5)

        # Get bias from source rating
        bias = _get_source_bias(source)

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
