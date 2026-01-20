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
    Safe wrapper around InferenceClient.text_classification with retries.
    """
    truncated_text = text[:512]

    for _ in range(3):
        try:
            return client.text_classification(truncated_text, model=model)

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
            print(f"Skipping article {article['id']} - too short")
            continue

        sentiment = _sentiment_score(content)
        bias = _bias_score(content)

        print(f"Article {article['id']}: S={sentiment}, B={bias}")

        if sentiment is not None and bias is not None:
            supabase.table("articles").update(
                {
                    "sentiment_score": sentiment,
                    "bias_score": bias,
                }
            ).eq("id", article["id"]).execute()

    print("Analysis job complete.")
