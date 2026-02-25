import os
import requests
from urllib.parse import urlparse

# Canonical category labels
CATEGORIES = [
    "politics", "world", "business", "tech",
    "sport", "entertainment", "health", "science", "general"
]

# HuggingFace Inference API — free tier, no local model loaded
# Add HF_TOKEN to Render env vars (free at huggingface.co/settings/tokens)
_HF_TOKEN = os.getenv("HF_API_TOKEN")
_HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "facebook/bart-large-mnli"
)


def _match_from_path(path: str) -> str | None:
    path = path.lower()
    if any(seg in path for seg in ["politics", "election", "government"]):
        return "politics"
    if any(seg in path for seg in ["world", "international", "europe", "africa", "asia"]):
        return "world"
    if any(seg in path for seg in ["business", "markets", "economy", "finance"]):
        return "business"
    if any(seg in path for seg in ["tech", "technology", "digital", "science/tech"]):
        return "tech"
    if any(seg in path for seg in ["sport", "sports", "football", "rugby", "tennis"]):
        return "sport"
    if any(seg in path for seg in ["entertainment", "culture", "arts", "tv", "film"]):
        return "entertainment"
    if any(seg in path for seg in ["health", "wellbeing", "covid", "medicine"]):
        return "health"
    if "science" in path:
        return "science"
    return None


def _match_from_title(title: str) -> str | None:
    t = title.lower()
    if any(w in t for w in ["election", "minister", "government", "parliament", "taoiseach", "senate", "congress", "brexit", "treaty", "legislation", "referendum", "political", "politician"]):
        return "politics"
    if any(w in t for w in ["stocks", "market", "economy", "inflation", "company", "bank", "gdp", "trade", "revenue", "profit", "investment", "financial", "shares", "nasdaq", "ftse"]):
        return "business"
    if any(w in t for w in ["ai", "app", "software", "startup", "technology", "cyber", "hack", "data breach", "robot", "drone", "smartphone", "chip", "gpu", "openai", "google", "apple", "microsoft"]):
        return "tech"
    if any(w in t for w in ["wins", "defeat", "draw", "tournament", "league", "cup", "match", "goal", "score", "player", "manager", "transfer", "premier league", "champions league", "fifa", "gaa", "cricket", "rugby"]):
        return "sport"
    if any(w in t for w in ["film", "movie", "series", "album", "festival", "celebrity", "actor", "actress", "director", "netflix", "disney", "spotify", "grammy", "oscar", "bafta"]):
        return "entertainment"
    if any(w in t for w in ["covid", "hospital", "vaccine", "health", "nhs", "hse", "cancer", "mental health", "drug", "medical", "disease", "pandemic", "obesity", "surgery", "patient"]):
        return "health"
    if any(w in t for w in ["climate", "planet", "environment", "research", "study", "scientists", "nasa", "space", "species", "carbon", "emissions", "biodiversity", "fossil", "renewable", "solar"]):
        return "science"
    if any(w in t for w in ["war", "conflict", "troops", "ukraine", "russia", "israel", "gaza", "nato", "un ", "united nations", "foreign", "diplomacy", "sanctions", "refugee", "ambassador"]):
        return "world"
    return None


def _classify_with_api(title: str, content: str | None) -> str:
    """
    Tier 3: HuggingFace Inference API — zero local memory cost.
    Falls back to 'general' if token missing or request fails.
    """
    if not _HF_TOKEN:
        return "general"

    text = f"{title}. {(content or '')[:400]}"

    try:
        response = requests.post(
            _HF_API_URL,
            headers={"Authorization": f"Bearer {_HF_TOKEN}"},
            json={
                "inputs": text,
                "parameters": {"candidate_labels": CATEGORIES},
            },
            timeout=8,  # don't block ingestion if API is slow
        )

        if response.status_code == 503:
            # Model loading on HF side — acceptable silent failure
            return "general"

        response.raise_for_status()
        result = response.json()
        return result["labels"][0]

    except Exception:
        return "general"


def infer_category(
    url: str | None,
    title: str | None,
    content: str | None = None,
) -> str:
    """
    3-tier category inference — zero local ML model loaded.

      Tier 1: URL path keyword match  (instant, no network)
      Tier 2: Title keyword match     (instant, no network, expanded)
      Tier 3: HuggingFace Inference API (remote, free, ~0 local RAM)

    Tier 3 only fires when tiers 1+2 both return None.
    Gracefully falls back to 'general' if HF_TOKEN is unset or API fails.
    """
    category = None

    if url:
        try:
            category = _match_from_path(urlparse(url).path or "")
        except Exception:
            pass

    if not category and title:
        category = _match_from_title(title)

    if not category and title:
        category = _classify_with_api(title, content)

    return category or "general"
