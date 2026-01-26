# app/jobs/analysis.py
import os
import time
from app.db.supabase import supabase
from huggingface_hub import InferenceClient


HF_API_TOKEN = os.getenv("HF_API_TOKEN")

SENTIMENT_MODEL = os.getenv(
    "HF_SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)
BIAS_MODEL = os.getenv(
    "HF_BIAS_MODEL",
    "bucketresearch/politicalBiasBERT"
)

client = InferenceClient(token=HF_API_TOKEN)


def _call_classification(model: str, text: str):
    """
    Safe wrapper around InferenceClient.text_classification.
    Retries on model loading errors (503).
    """
    truncated_text = text[:512]

    for _ in range(3):
        try:
            return client.text_classification(
                truncated_text,
                model=model
            )

        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "loading" in error_str.lower():
                print(f"Model {model} loading... waiting 5s")
                time.sleep(5)
                continue

            print(f"Analysis error for {model}: {e}")
            return None
    return None


def _sentiment_score(text: str):
    """
    Analyze sentiment using Hugging Face model.
    Returns float: positive (0 to 1), negative (-1 to 0), or None.
    """
    try:
        results = _call_classification(SENTIMENT_MODEL, text)
        if not results:
            return None

        top = max(results, key=lambda x: x.score)

        label = top.label.upper()
        score = top.score

        if "POS" in label:
            return score
        elif "NEG" in label:
            return -score
        else:
            return 0.0
    except Exception as e:
        print(f"Sentiment parsing error: {e}")
        return None


def _bias_score(text: str):
    """
    Analyze political bias using Hugging Face model.
    Returns float: left (-1) to right (+1), or None.
    """
    try:
        results = _call_classification(BIAS_MODEL, text)
        if not results:
            return None

        left_score = 0.0
        right_score = 0.0

        for item in results:
            lbl = item.label.upper()
            scr = item.score
            if 'LEFT' in lbl:
                left_score = scr
            if 'RIGHT' in lbl:
                right_score = scr

        return right_score - left_score

    except Exception as e:
        print(f"Bias parsing error: {e}")
        return None


def analyze_unscored_articles():
    """
    Analyze articles missing sentiment or bias scores.
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

    print(f"Analyzing {len(articles)} articles...")

    for article in articles:
        # Use content if available, fallback to title
        content = article.get("content") or ""
        title = article.get("title") or ""

        # Combine title and content for better analysis
        text = f"{title}. {content}".strip()

        if len(text) < 10:
            print(
                f"Skipping article {article['id']} - "
                f"no title or content"
            )
            continue

        sentiment = _sentiment_score(text)

        # Small delay to avoid rate limits
        time.sleep(0.5)

        bias = _bias_score(text)

        print(f"Article {article['id']}: S={sentiment}, B={bias}")

        # Save even if only one score is available
        if sentiment is not None or bias is not None:
            update_data = {}
            if sentiment is not None:
                update_data["sentiment_score"] = sentiment
            if bias is not None:
                update_data["bias_score"] = bias

            supabase.table("articles").update(
                update_data
            ).eq("id", article["id"]).execute()

    print("Analysis job complete.")
