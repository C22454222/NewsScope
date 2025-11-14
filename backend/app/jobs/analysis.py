# app/jobs/analysis.py
import os
import requests
from app.db.supabase import supabase


HF_API_TOKEN = os.getenv("HF_API_TOKEN")
SENTIMENT_MODEL = os.getenv("HF_SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
BIAS_MODEL = os.getenv("HF_BIAS_MODEL", "cardiffnlp/twitter-roberta-base")  # placeholder


def _hf_post(model: str, inputs: str):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    r = requests.post(f"https://api-inference.huggingface.co/models/{model}",
                      headers=headers, json={"inputs": inputs}, timeout=30)
    r.raise_for_status()
    return r.json()


def _sentiment_score(text: str):
    try:
        out = _hf_post(SENTIMENT_MODEL, text)
        # Map to a simple score: positive=1, negative=-1, neutral=0
        label = out[0][0]["label"].lower()
        score = out[0][0]["score"]
        return  score if "pos" in label else (-score if "neg" in label else 0.0)
    except Exception:
        return None


def _bias_score(text: str):
    # Placeholder: until you pick a concrete bias classifier, return None or a dummy
    # You can later replace with a political spectrum classifier
    return None


def analyze_unscored_articles():
    rows = supabase.table("articles").select("*").is_("sentiment_score", "null").limit(50).execute().data
    for row in rows:
        # You may need to fetch content separately; for now, use URL as text placeholder
        text = row["url"]
        sentiment = _sentiment_score(text)
        bias = _bias_score(text)
        supabase.table("articles").update({
            "sentiment_score": sentiment,
            "bias_score": bias
        }).eq("id", row["id"]).execute()
