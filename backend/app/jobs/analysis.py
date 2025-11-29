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


def _call_model(model: str, text: str):
    """
    Safe wrapper around InferenceClient with retries for loading models.
    """
    # Truncate to 512 chars to avoid model errors
    truncated_text = text[:512]

    for _ in range(3):  # Retry loop
        try:
            # We use the specific HTTP post method to get raw JSON
            # This allows us to manually parse the weird list-of-lists format
            response = client.post(
                json={"inputs": truncated_text},
                model=model
            )
            # client.post returns bytes, we need json
            import json
            return json.loads(response.decode("utf-8"))

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
        out = _call_model(SENTIMENT_MODEL, text)
        if not out or isinstance(out, dict) and "error" in out:
            return None

        # Output format: [[{'label': 'POSITIVE', 'score': 0.9}]]
        top = out[0]
        if isinstance(top, list):
            top = sorted(top, key=lambda x: x['score'], reverse=True)[0]

        label = top['label'].upper()
        score = top['score']

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
        result = _call_model(BIAS_MODEL, text)
        if not result or isinstance(result, dict) and "error" in result:
            return None

        # Output format: [[{'label': 'LEFT', 'score': 0.9}, ...]]
        scores = result[0] if isinstance(result[0], list) else result

        left_score = 0.0
        right_score = 0.0

        for item in scores:
            lbl = item['label'].upper()
            scr = item['score']
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
        .limit(5)
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
