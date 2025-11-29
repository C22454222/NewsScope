# app/jobs/analysis.py (Hugging Face integration)
import os
import requests
import time
from app.db.supabase import supabase

# Hugging Face API credentials and model names are configured via environment
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Sentiment: Positive/Negative
SENTIMENT_MODEL = os.getenv(
    "HF_SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)

# Politics: Left/Center/Right
# We use a specific political model (defaults to politicalBiasBERT if not set in env)
BIAS_MODEL = os.getenv("HF_BIAS_MODEL", "bucketresearch/politicalBiasBERT")


def _hf_post(model: str, inputs: str):
    """
    Call a Hugging Face hosted model with the given input text.

    Handles wait-on-load errors (503) by retrying automatically.

    Raises:
        requests.HTTPError: if the Hugging Face API returns a non-2xx response after retries.
    """
    # UPDATED URL HERE: using 'router' instead of 'api-inference'
    url = f"https://router.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    # Truncate text to 512 chars to fit BERT model limits (crucial for performance)
    payload = {"inputs": inputs[:512]}

    for _ in range(3):  # Retry up to 3 times
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)

            if response.status_code == 503:
                # Model is loading (cold start), wait and retry
                print(f"Model {model} loading... waiting 10s")
                time.sleep(10)
                continue

            if response.status_code == 200:
                return response.json()

            # Log error but don't crash immediately, return None to handle gracefully
            print(f"HF Error {response.status_code}: {response.text}")
            return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    return None


def _sentiment_score(text: str):
    """
    Run sentiment analysis on a block of text.

    Returns:
        float | None: Positive scores for positive sentiment,
        negative scores for negative sentiment, 0.0 for neutral,
        or None if the model call fails.
    """
    try:
        out = _hf_post(SENTIMENT_MODEL, text)
        if not out or isinstance(out, dict) and "error" in out:
            return None

        # Hugging Face returns a list of lists of label/score dicts: [[{'label': 'POSITIVE', 'score': 0.9}]]
        # Handle flattening if needed
        top = out[0]
        if isinstance(top, list):
            # Sort by score descending to get the most confident label
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
        # Analysis failures are tolerated; the article remains unscored
        print(f"Sentiment analysis error: {e}")
        return None


def _bias_score(text: str):
    """
    Run political bias analysis on a block of text.

    Returns:
        float | None: -1.0 (Left) to 1.0 (Right), 0.0 (Center),
        or None if analysis fails.
    """
    try:
        result = _hf_post(BIAS_MODEL, text)
        if not result or isinstance(result, dict) and "error" in result:
            return None

        # The model returns labels: LEFT, CENTER, RIGHT
        # Example output: [[{'label': 'LEFT', 'score': 0.9}, {'label': 'RIGHT', 'score': 0.1}]]

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
            # We ignore CENTER score for the simplified -1 to 1 axis calc

        # Heuristic: Right confidence minus Left confidence
        # Result ranges from -1 (Strong Left) to +1 (Strong Right)
        return right_score - left_score

    except Exception as e:
        print(f"Bias analysis error: {e}")
        return None


def analyze_unscored_articles():
    """
    Fetch articles without sentiment/bias scores and enrich them using NLP.

    This job is designed to be called periodically by the scheduler so
    that analysis is decoupled from ingestion and can run in the background.
    """
    print("Starting analysis job...")

    # Select a batch of articles that do not yet have a sentiment score
    # Limit to 5 per run to avoid hitting API rate limits
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
        # Prefer content, fallback to title if content is missing or empty
        content = article.get("content") or article.get("title") or ""

        if len(content) < 50:
            print(f"Skipping article {article['id']} - content too short")
            continue

        # Run analysis
        sentiment = _sentiment_score(content)
        bias = _bias_score(content)

        print(f"Article {article['id']}: Sentiment={sentiment}, Bias={bias}")

        # Persist the new scores for this article
        # Only update if we got results to avoid setting nulls repeatedly
        if sentiment is not None and bias is not None:
            supabase.table("articles").update(
                {
                    "sentiment_score": sentiment,
                    "bias_score": bias,
                }
            ).eq("id", article["id"]).execute()

    print("Analysis job complete.")
