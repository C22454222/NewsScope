# app/jobs/analysis.py (Hugging Face integration)
import os
import time
from app.db.supabase import supabase
from huggingface_hub import InferenceClient

# Hugging Face API credentials
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Models
SENTIMENT_MODEL = os.getenv(
    "HF_SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)
BIAS_MODEL = os.getenv("HF_BIAS_MODEL", "bucketresearch/politicalBiasBERT")

# Initialize Client
client = InferenceClient(token=HF_API_TOKEN)


def _call_classification(model: str, text: str):
    """
    Safe wrapper around InferenceClient.text_classification with retries.
    """
    # Truncate to 512 chars to avoid model errors
    truncated_text = text[:512]

    for _ in range(3):  # Retry loop
        try:
            # This method returns a list of ClassificationOutput objects
            # e.g. [TextClassificationOutput(label='POSITIVE', score=0.99), ...]
            return client.text_classification(truncated_text, model=model)

        except Exception as e:
            error_str = str(e)
            # Check if model is loading (503 error hidden in exception)
            if "503" in error_str or "loading" in error_str.lower():
                print(f"Model {model} loading... waiting 5s")
                time.sleep(5)
                continue

            print(f"Analysis error for {model}: {e}")
            return None
    return None


def _sentiment_score(text: str):
    try:
        results = _call_classification(SENTIMENT_MODEL, text)
        if not results:
            return None

        # Find the top result (highest score)
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
    try:
        results = _call_classification(BIAS_MODEL, text)
        if not results:
            return None

        # Parse list of objects: [Output(label='LEFT', score=0.9), ...]
        left_score = 0.0
        right_score = 0.0

        for item in results:
            lbl = item.label.upper()
            scr = item.score
            if 'LEFT' in lbl:
                left_score = scr
            if 'RIGHT' in lbl:
                right_score = scr

        # Heuristic: Right - Left = -1.0 (Left) to 1.0 (Right)
        return right_score - left_score

    except Exception as e:
        print(f"Bias parsing error: {e}")
        return None


def analyze_unscored_articles():
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
        content = article.get("content") or article.get("title") or ""

        if len(content) < 50:
            print(f"Skipping article {article['id']} - content too short")
            continue

        sentiment = _sentiment_score(content)
        bias = _bias_score(content)

        print(f"Article {article['id']}: Sentiment={sentiment}, Bias={bias}")

        if sentiment is not None and bias is not None:
            supabase.table("articles").update(
                {
                    "sentiment_score": sentiment,
                    "bias_score": bias,
                }
            ).eq("id", article["id"]).execute()

    print("Analysis job complete.")
