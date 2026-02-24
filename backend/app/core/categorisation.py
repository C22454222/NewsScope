from urllib.parse import urlparse
from transformers import pipeline


# Canonical category labels — used by zero-shot model
CATEGORIES = [
    "politics", "world", "business", "tech",
    "sport", "entertainment", "health", "science", "general"
]

# Lazy-loaded — only initialised on first zero-shot call
# Avoids memory cost on cold starts when keyword matching is sufficient
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-large-zeroshot-v2",
            device=-1,  # CPU; set to 0 if GPU available
        )
    return _classifier


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


def _classify_with_model(title: str, content: str | None) -> str:
    text = f"{title}. {(content or '')[:512]}"
    result = _get_classifier()(text, CATEGORIES, multi_label=False)
    return result["labels"][0]


def infer_category(
    url: str | None,
    title: str | None,
    content: str | None = None,
) -> str:
    """
    3-tier category inference:
      1. URL path keyword match  (fastest, no model)
      2. Title keyword match     (fast, no model)
      3. Zero-shot NLP model     (only fires when both above return None)

    content is only used by tier 3 to improve accuracy.
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
        category = _classify_with_model(title, content)

    return category or "general"
