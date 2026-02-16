# app/core/categorization.py
from urllib.parse import urlparse

# Canonical category labels
CATEGORIES = {
    "politics",
    "world",
    "business",
    "tech",
    "sport",
    "entertainment",
    "health",
    "science",
    "general",
}


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
    # Very lightweight keyword rules
    if any(w in t for w in ["election", "minister", "government", "parliament"]):
        return "politics"
    if any(w in t for w in ["stocks", "market", "economy", "inflation", "company"]):
        return "business"
    if any(w in t for w in ["ai", "app", "software", "startup", "technology"]):
        return "tech"
    if any(w in t for w in ["wins", "defeat", "draw", "championship", "tournament"]):
        return "sport"
    if any(w in t for w in ["film", "movie", "series", "album", "festival"]):
        return "entertainment"
    if any(w in t for w in ["covid", "hospital", "vaccine", "health", "nhs"]):
        return "health"
    if any(w in t for w in ["climate", "planet", "environment", "research", "study"]):
        return "science"
    return None


def infer_category(url: str | None, title: str | None) -> str:
    # Default
    category = None

    if url:
        try:
            parsed = urlparse(url)
            path = parsed.path or ""
            category = _match_from_path(path)
        except Exception:
            category = None

    if not category and title:
        category = _match_from_title(title)

    return category or "general"
