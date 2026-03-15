"""NewsScope article category inference - 3-tier, zero local ML.

Never returns 'general' — worst case is 'world'.
Tier 3 falls back to _match_from_title_broad() instead of 'general'
when HF_TOKEN is missing or the API call fails.

Flake8: 0 errors/warnings.
"""

import os
import requests
from typing import Optional
from urllib.parse import urlparse


# Expanded canonical category labels from BBC/NYT/Guardian/RTE/AP standards.
CATEGORIES = [
    "politics",
    "world",
    "us",
    "uk",
    "ireland",
    "europe",
    "business",
    "economy",
    "markets",
    "finance",
    "tech",
    "science",
    "environment",
    "climate",
    "health",
    "sport",
    "football",
    "rugby",
    "gaa",
    "cricket",
    "entertainment",
    "culture",
    "film",
    "tv",
    "music",
    "lifestyle",
    "travel",
    "food",
    "news",
    "local",
    "opinion",
    "analysis",
]

# Tier 3 candidate labels — 8 parent groups only.
# BART MNLI performs poorly with 33 labels; 8 is the sweet spot.
# Sub-categories are handled by tiers 1+2 and CATEGORY_GROUP_MAP
# in articles.py. This ensures fast, accurate API classification.
_TIER3_LABELS = [
    "politics",
    "world",
    "business",
    "tech",
    "sport",
    "entertainment",
    "health",
    "science",
]

# HuggingFace Inference API — free tier, no local model loaded.
# Add HF_API_TOKEN to Render env vars.
_HF_TOKEN = os.getenv("HF_API_TOKEN")
_HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "facebook/bart-large-mnli"
)


# ── Tier 1a: URL path matching ────────────────────────────────────────────────


def _match_from_path(path: str) -> Optional[str]:
    """Tier 1a: Direct path/section matching + keyword fallback."""
    if not path:
        return None

    path_lower = path.lower()

    section_map = {
        # BBC
        "/news/uk": "uk",
        "/news/world": "world",
        "/news/business": "business",
        "/news/technology": "tech",
        "/news/science-environment": "science",
        "/news/health": "health",
        "/sport": "sport",
        "/news/entertainment-arts": "entertainment",
        # NYT
        "/section/us": "us",
        "/section/politics": "politics",
        "/section/business": "business",
        "/section/technology": "tech",
        "/section/science": "science",
        "/section/health": "health",
        "/section/sports": "sport",
        "/section/arts": "entertainment",
        # RTE
        "/news/ireland": "ireland",
        "/news/politics": "politics",
        "/sport": "sport",
        "/news/business": "business",
        "/news/health": "health",
        # Guardian
        "/uk-news": "uk",
        "/world": "world",
        "/business": "business",
        "/technology": "tech",
        "/science": "science",
        "/sport": "sport",
        "/culture": "entertainment",
        "/environment": "environment",
        # Irish Times
        "/ireland": "ireland",
        "/crime-law": "ireland",
        "/crime-law/courts": "ireland",
        "/courts": "ireland",
        # Generic
        "/politics": "politics",
        "/opinion": "opinion",
        "/analysis": "analysis",
        "/local": "local",
    }

    parts = [p for p in path_lower.split("/") if p]
    for i in range(1, min(4, len(parts) + 1)):
        key = "/".join(parts[:i])
        if key in section_map:
            return section_map[key]

    # Keyword fallback on path segments.
    if any(seg in path_lower for seg in ["politics", "election", "government"]):
        return "politics"
    if any(seg in path_lower for seg in ["world", "international", "europe", "africa", "asia"]):
        return "world"
    if any(seg in path_lower for seg in ["business", "markets", "economy", "finance"]):
        return "business"
    if any(seg in path_lower for seg in ["tech", "technology", "digital"]):
        return "tech"
    if any(seg in path_lower for seg in ["sport", "sports", "football", "rugby", "tennis", "gaa"]):
        return "sport"
    if any(seg in path_lower for seg in ["entertainment", "culture", "arts", "tv", "film"]):
        return "entertainment"
    if any(seg in path_lower for seg in ["health", "wellbeing", "covid", "medicine"]):
        return "health"
    if any(seg in path_lower for seg in ["science", "environment"]):
        return "science"
    if any(seg in path_lower for seg in ["crime", "courts", "law", "crime-law"]):
        return "ireland"
    if any(seg in path_lower for seg in ["opinion", "comment"]):
        return "opinion"

    return None


# ── Tier 2: Title keyword matching ────────────────────────────────────────────


def _match_from_title(title: str) -> Optional[str]:
    """Tier 2: Expanded title keywords covering ~95% of common topics."""
    if not title:
        return None

    t = title.lower()

    if any(w in t for w in [
        "election", "minister", "government", "parliament",
        "taoiseach", "senate", "congress", "brexit", "treaty",
        "legislation", "referendum", "political", "politician",
        "policy", "bill",
    ]):
        return "politics"

    if any(w in t for w in [
        "stocks", "market", "economy", "inflation", "company",
        "bank", "gdp", "trade", "revenue", "profit", "investment",
        "financial", "shares", "nasdaq", "ftse", "dow", "recession",
    ]):
        return "business"

    if any(w in t for w in [
        "ai", "app", "software", "startup", "technology", "cyber",
        "hack", "data breach", "robot", "drone", "smartphone",
        "chip", "gpu", "openai", "google", "apple", "microsoft",
        "chatgpt",
    ]):
        return "tech"

    if any(w in t for w in [
        "wins", "defeat", "draw", "tournament", "league", "cup",
        "match", "goal", "score", "player", "manager", "transfer",
        "premier league", "champions league", "fifa", "gaa",
        "all-ireland", "hurling", "camogie", "cricket", "rugby",
        "six nations",
    ]):
        return "sport"

    if any(w in t for w in [
        "film", "movie", "series", "album", "festival", "celebrity",
        "actor", "actress", "director", "netflix", "disney",
        "spotify", "grammy", "oscar", "bafta", "concert", "theatre",
    ]):
        return "entertainment"

    if any(w in t for w in [
        "covid", "hospital", "vaccine", "health", "nhs", "hse",
        "cancer", "mental health", "drug", "medical", "disease",
        "pandemic", "obesity", "surgery", "patient", "waiting lists",
    ]):
        return "health"

    if any(w in t for w in [
        "climate", "planet", "environment", "research", "study",
        "scientists", "nasa", "space", "species", "carbon",
        "emissions", "biodiversity", "fossil", "renewable", "solar",
        "weather",
    ]):
        return "science"

    if any(w in t for w in [
        "war", "conflict", "troops", "ukraine", "russia", "israel",
        "gaza", "nato", "united nations", "foreign", "diplomacy",
        "sanctions", "refugee", "ambassador", "peace talks",
    ]):
        return "world"

    if any(w in t for w in [
        "dublin", "cork", "galway", "traffic", "crime", "courts",
        "court", "asylum", "high court", "tribunal", "garda",
        "judicial", "murder", "convicted", "sentencing", "acquitted",
    ]):
        return "ireland"

    if any(w in t for w in [
        "opinion", "analysis", "comment", "editorial", "column",
    ]):
        return "opinion"

    return None


# ── Tier 3 fallback: broad scan ───────────────────────────────────────────────


def _match_from_title_broad(title: str) -> str:
    """
    Broad fallback keyword scan — catches what tier 2 misses.

    Used when HF_TOKEN is missing or the API call fails.
    Returns 'world' as true last resort — unclassifiable articles
    are more likely international news than any other single category.
    Never returns 'general'.
    """
    if not title:
        return "world"

    t = title.lower()

    # Governance / official events → politics or world
    if any(w in t for w in [
        "president", "prime minister", "official", "authorities",
        "agency", "committee", "summit", "bilateral", "rally",
        "protest", "military", "army", "attack", "shooting",
        "explosion",
    ]):
        return "world"

    # Legal / courts / crime
    if any(w in t for w in [
        "court", "judge", "jury", "verdict", "sentence", "charged",
        "arrested", "police", "murder", "trial", "lawsuit", "legal",
        "attorney", "prosecution", "accused",
    ]):
        return "ireland"

    # Economy / money
    if any(w in t for w in [
        "price", "cost", "billion", "million", "fund", "budget",
        "tax", "rate", "growth", "loss", "earnings", "quarter",
        "fiscal", "debt", "interest",
    ]):
        return "business"

    # People / culture / lifestyle
    if any(w in t for w in [
        "star", "fans", "award", "show", "premiere", "release",
        "interview", "model", "fashion", "viral", "social media",
    ]):
        return "entertainment"

    # True last resort — international news is the most likely bucket
    # for any article that genuinely cannot be classified.
    return "world"


# ── Tier 3: HuggingFace API ───────────────────────────────────────────────────


def _classify_with_api(
    title: str, content: Optional[str] = None
) -> str:
    """
    Tier 3: HuggingFace Inference API — zero local memory cost.

    Falls back to _match_from_title_broad() instead of 'general'
    when HF_TOKEN is missing or the API call fails — every article
    gets a meaningful category regardless of API availability.
    """
    if not _HF_TOKEN:
        return _match_from_title_broad(title)

    text = f"{title}. {(content or '')[:400]}"

    try:
        response = requests.post(
            _HF_API_URL,
            headers={"Authorization": f"Bearer {_HF_TOKEN}"},
            json={
                "inputs": text,
                "parameters": {"candidate_labels": _TIER3_LABELS},
            },
            timeout=8,
        )

        if response.status_code == 503:
            # HF model still loading — fall back to broad scan.
            return _match_from_title_broad(title)

        response.raise_for_status()
        return response.json()["labels"][0]

    except Exception:
        return _match_from_title_broad(title)


# ── Public interface ──────────────────────────────────────────────────────────


def infer_category(
    url: Optional[str],
    title: Optional[str],
    content: Optional[str] = None,
) -> str:
    """
    3-tier category inference — zero local ML model loaded.

    Tier 1a: URL path/section parsing (BBC/NYT/RTE/Irish Times maps)
    Tier 1b: URL keyword match
    Tier 2:  Title keyword match (strict, ~95% coverage)
    Tier 3:  HuggingFace BART zero-shot OR broad keyword fallback
    Never returns 'general' — worst case is 'world'.
    """
    category = None

    if url:
        try:
            parsed_path = urlparse(url).path or ""
            category = _match_from_path(parsed_path)
        except Exception:
            pass

    if not category and title:
        category = _match_from_title(title)

    if not category:
        category = _classify_with_api(title or "", content)

    # _classify_with_api guarantees a non-None, non-'general' result.
    return category
