# app/jobs/analysis.py (Hugging Face integration)
import os
import requests
from app.db.supabase import supabase

# Hugging Face API credentials and model names are configured via environment
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
SENTIMENT_MODEL = os.getenv(
    "HF_SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)
# Bias classifier is a placeholder and can be swapped out later
BIAS_MODEL = os.getenv("HF_BIAS_MODEL", "cardiffnlp/twitter-roberta-base")


def _hf_post(model: str, inputs: str):
    """
    Call a Hugging Face hosted model with the given input text.

    Raises:
        requests.HTTPError: if the Hugging Face API returns a non-2xx response.
    """
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    r = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers=headers,
        json={"inputs": inputs},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


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
        # Hugging Face returns a list of lists of label/score dicts
        label = out[0][0]["label"].lower()
        score = out[0][0]["score"]
        if "pos" in label:
            return score
        elif "neg" in label:
            return -score
        else:
            return 0.0
    except Exception:
        # Analysis failures are tolerated; the article remains unscored
        return None


def _bias_score(text: str):
    """
    Placeholder for a future political bias classifier.

    For now, this returns None so the database schema and
    calling code are ready when a real model is integrated.
    """
    return None


def analyze_unscored_articles():
    """
    Fetch articles without sentiment/bias scores and enrich them using NLP.

    This job is designed to be called periodically by the scheduler so
    that analysis is decoupled from ingestion and can run in the background.
    """
    # Select a batch of articles that do not yet have a sentiment score
    articles = (
        supabase.table("articles")
        .select("*")
        .is_("sentiment_score", "null")
        .limit(50)
        .execute()
        .data
    )

    for article in articles:
        content = article.get("content")
        if content:
            sentiment = _sentiment_score(content)
            bias = _bias_score(content)

            # Persist the new scores for this article
            supabase.table("articles").update(
                {
                    "sentiment_score": sentiment,
                    "bias_score": bias,
                }
            ).eq("id", article["id"]).execute()
