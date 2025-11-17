import os
import requests
from app.db.supabase import supabase


HF_API_TOKEN = os.getenv("HF_API_TOKEN")
SENTIMENT_MODEL = os.getenv("HF_SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
BIAS_MODEL = os.getenv("HF_BIAS_MODEL", "cardiffnlp/twitter-roberta-base")  # placeholder


def _hf_post(model: str, inputs: str):
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
    try:
        out = _hf_post(SENTIMENT_MODEL, text)
        # Map to a simple score: positive=+score, negative=-score, neutral=0
        label = out[0][0]["label"].lower()
        score = out[0][0]["score"]
        if "pos" in label:
            return score
        elif "neg" in label:
            return -score
        else:
            return 0.0
    except Exception:
        return None


def _bias_score(text: str):
    # Placeholder until you integrate a real bias classifier
    # For now, return None or a dummy value
    return None


def analyze_unscored_articles():
    # Fetch articles that don't yet have sentiment_score
    articles = supabase.table("articles").select("*").is_("sentiment_score", "null").limit(50).execute().data
    for article in articles:
        content = article.get("content")
        if content:
            sentiment = _sentiment_score(content)
            bias = _bias_score(content)
            supabase.table("articles").update({
                "sentiment_score": sentiment,
                "bias_score": bias
            }).eq("id", article["id"]).execute()
