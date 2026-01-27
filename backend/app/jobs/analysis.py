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
    "valurank/distilroberta-bias"
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
    Analyze sentiment using CardiffNLP RoBERTa (Liu et al., 2019).
    Returns float: -1 (negative) to +1 (positive).
    """
    try:
        results = _call_classification(SENTIMENT_MODEL, text)
        if not results:
            return None

        # CardiffNLP returns: negative, neutral, positive
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

        # Calculate weighted sentiment
        neg = sentiment_map.get('negative', 0)
        pos = sentiment_map.get('positive', 0)

        return pos - neg

    except Exception as e:
        print(f"Sentiment parsing error: {e}")
        return None


def _bias_score(text: str):
    """
    Analyze political bias using RoBERTa (Liu et al., 2019).
    Article-level bias detection as per interim report.
    Returns float: -1 (Left) to +1 (Right).
    """
    try:
        results = _call_classification(BIAS_MODEL, text)
        if not results:
            return None

        # DEBUG: Print what the model actually returns
        print(f"🔍 Bias model output: {results}")

        # Parse bias labels
        bias_map = {}
        for item in results:
            label = item.label.upper()
            score = item.score

            print(f"   Label: {label}, Score: {score}")

            # Handle various label formats
            if 'LEFT' in label or 'LIBERAL' in label:
                bias_map['left'] = score
            elif (
                'CENTER' in label or 'NEUTRAL' in label or 'MODERATE' in label
            ):
                bias_map['center'] = score
            elif 'RIGHT' in label or 'CONSERVATIVE' in label:
                bias_map['right'] = score
            elif 'BIAS' in label:
                # Some models return "Biased" vs "Unbiased"
                bias_map['biased'] = score

        # Convert to -1 (left) to +1 (right) scale
        left = bias_map.get('left', 0)
        center = bias_map.get('center', 0)
        right = bias_map.get('right', 0)

        # If we have left/center/right scores
        if left > 0 or center > 0 or right > 0:
            # Weighted average approach
            total = left + center + right
            if total > 0:
                # Map to -1 to +1 scale
                bias_value = (right - left) / total
                print(f"   Final bias: {bias_value}")
                return bias_value

        # Fallback: return 0 (neutral) if unclear
        print("   No bias labels found, returning 0.0")
        return 0.0

    except Exception as e:
        print(f"Bias parsing error: {e}")
        return None


def analyze_unscored_articles():
    """
    Analyze articles using RoBERTa (Liu et al., 2019).
    Article-level bias and sentiment detection.
    Runs every hour at :15 (15 minutes after ingestion).
    """
    print("Starting analysis job...")

    response = (
        supabase.table("articles")
        .select("id, content, title")
        .is_("sentiment_score", "null")
        .limit(500)
        .execute()
    )
    articles = response.data

    if not articles:
        print("No unscored articles found.")
        return

    print(f"Analyzing {len(articles)} articles with RoBERTa...")

    for article in articles:
        content = article.get("content") or ""
        title = article.get("title") or ""

        # Combine title and content for analysis
        text = f"{title}. {content}".strip()

        if len(text) < 10:
            print(
                f"Skipping article {article['id']} - no content"
            )
            continue

        # Run sentiment analysis
        sentiment = _sentiment_score(text)

        # Small delay to avoid rate limits
        time.sleep(1.0)

        # Run bias analysis (article-level)
        bias = _bias_score(text)

        print(
            f"Article {article['id']}: "
            f"Sentiment={sentiment}, Bias={bias}"
        )

        # Update database
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
