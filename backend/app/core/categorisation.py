import os
import time
from typing import Dict, Optional
from urllib.parse import urlparse

import requests


CATEGORIES = [
    "politics", "world", "us", "uk", "ireland", "europe",
    "business", "economy", "markets", "finance",
    "tech", "science", "environment", "climate", "health",
    "sport", "football", "rugby", "gaa", "cricket",
    "entertainment", "culture", "film", "tv", "music",
    "crime",
    "opinion", "analysis",
]


_TIER3_LABELS = [
    "politics", "world", "business", "tech",
    "sport", "entertainment", "health", "science",
    "crime",
]


_HF_TOKEN = os.getenv("HF_API_TOKEN")
_HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "facebook/bart-large-mnli"
)


_hf_session: Optional[requests.Session] = None


def _get_hf_session() -> requests.Session:
    global _hf_session
    if _hf_session is None:
        _hf_session = requests.Session()
    return _hf_session


_SOURCE_URL_PREFIX_MAP: Dict[str, str] = {
    # RTÉ News
    "rte.ie/news/politics": "politics",
    "rte.ie/news/ireland": "ireland",
    "rte.ie/news/business": "business",
    "rte.ie/news/world": "world",
    "rte.ie/news/health": "health",
    "rte.ie/news/science": "science",
    "rte.ie/sport": "sport",
    "rte.ie/news": "ireland",
    # BBC
    "bbc.com/news/uk": "uk",
    "bbc.com/news/world": "world",
    "bbc.com/news/business": "business",
    "bbc.com/news/technology": "tech",
    "bbc.com/news/science-environment": "science",
    "bbc.com/news/health": "health",
    "bbc.com/news/entertainment-arts": "entertainment",
    "bbc.com/news/politics": "politics",
    "bbc.com/news/us-canada": "us",
    "bbc.com/news/europe": "europe",
    "bbc.com/sport": "sport",
    "bbc.com/news": "world",
    "bbc.co.uk/news/uk": "uk",
    "bbc.co.uk/news/world": "world",
    "bbc.co.uk/news/business": "business",
    "bbc.co.uk/news/technology": "tech",
    "bbc.co.uk/news/health": "health",
    "bbc.co.uk/news/us-canada": "us",
    "bbc.co.uk/sport": "sport",
    "bbc.co.uk/news": "world",
    # The Guardian
    "theguardian.com/politics": "politics",
    "theguardian.com/uk-news": "uk",
    "theguardian.com/world": "world",
    "theguardian.com/us-news": "us",
    "theguardian.com/europe-news": "europe",
    "theguardian.com/business": "business",
    "theguardian.com/technology": "tech",
    "theguardian.com/science": "science",
    "theguardian.com/environment": "environment",
    "theguardian.com/society": "health",
    "theguardian.com/health": "health",
    "theguardian.com/sport": "sport",
    "theguardian.com/football": "sport",
    "theguardian.com/culture": "entertainment",
    "theguardian.com/film": "entertainment",
    "theguardian.com/music": "entertainment",
    "theguardian.com/commentisfree": "opinion",
    # Irish Times
    "irishtimes.com/politics": "politics",
    "irishtimes.com/ireland": "ireland",
    "irishtimes.com/world": "world",
    "irishtimes.com/business": "business",
    "irishtimes.com/technology": "tech",
    "irishtimes.com/health": "health",
    "irishtimes.com/science": "science",
    "irishtimes.com/environment": "environment",
    "irishtimes.com/sport": "sport",
    "irishtimes.com/culture": "entertainment",
    "irishtimes.com/crime-law": "crime",
    "irishtimes.com/courts": "crime",
    "irishtimes.com/opinion": "opinion",
    # The Independent (Ireland edition)
    "independent.ie/irish-news": "ireland",
    "independent.ie/world-news": "world",
    "independent.ie/business": "business",
    "independent.ie/entertainment": "entertainment",
    "independent.ie/sport": "sport",
    "independent.ie/health": "health",
    "independent.ie/regionals": "ireland",
    "independent.co.uk/news/uk": "uk",
    "independent.co.uk/news/world": "world",
    "independent.co.uk/news/business": "business",
    "independent.co.uk/sport": "sport",
    # Sky News
    "news.sky.com/story": "world",
    "news.sky.com/uk": "uk",
    "news.sky.com/world": "world",
    "news.sky.com/politics": "politics",
    "news.sky.com/business": "business",
    "news.sky.com/science-tech": "tech",
    "news.sky.com/health": "health",
    "news.sky.com/entertainment": "entertainment",
    "news.sky.com/sport": "sport",
    # Deutsche Welle
    "dw.com/en/politics": "politics",
    "dw.com/en/business": "business",
    "dw.com/en/science": "science",
    "dw.com/en/environment": "environment",
    "dw.com/en/health": "health",
    "dw.com/en/sport": "sport",
    "dw.com/en/culture": "entertainment",
    "dw.com/en/europe": "europe",
    "dw.com/en": "world",
    # AP News
    "apnews.com/hub/politics": "politics",
    "apnews.com/hub/world-news": "world",
    "apnews.com/hub/business": "business",
    "apnews.com/hub/technology": "tech",
    "apnews.com/hub/sports": "sport",
    "apnews.com/hub/health": "health",
    "apnews.com/hub/science": "science",
    "apnews.com/hub/entertainment": "entertainment",
    "apnews.com/hub/us-news": "us",
    "apnews.com/hub/europe": "europe",
    # CNN
    "cnn.com/politics": "politics",
    "cnn.com/world": "world",
    "cnn.com/us": "us",
    "cnn.com/business": "business",
    "cnn.com/health": "health",
    "cnn.com/entertainment": "entertainment",
    "cnn.com/sport": "sport",
    "cnn.com/tech": "tech",
    "cnn.com/science": "science",
    # NPR
    "npr.org/sections/politics": "politics",
    "npr.org/sections/business": "business",
    "npr.org/sections/health": "health",
    "npr.org/sections/science": "science",
    "npr.org/sections/technology": "tech",
    "npr.org/sections/arts": "entertainment",
    "npr.org/sections/world": "world",
    "npr.org/sections/national": "us",
    # Fox News
    "foxnews.com/politics": "politics",
    "foxnews.com/us": "us",
    "foxnews.com/world": "world",
    "foxnews.com/health": "health",
    "foxnews.com/entertainment": "entertainment",
    "foxnews.com/sports": "sport",
    "foxnews.com/tech": "tech",
    "foxnews.com/opinion": "opinion",
    # GB News
    "gbnews.com/politics": "politics",
    "gbnews.com/business": "business",
    "gbnews.com/sport": "sport",
    "gbnews.com/entertainment": "entertainment",
    "gbnews.com/science-tech": "tech",
    "gbnews.com/health": "health",
}


def _match_from_known_source(url: str) -> Optional[str]:
    """
    Tier 0: strip scheme + www, then check against known source prefixes.
    More specific paths checked before broader ones via sorted order.
    """
    if not url:
        return None

    url_lower = url.lower()

    for scheme in ("https://", "http://"):
        if url_lower.startswith(scheme):
            url_lower = url_lower[len(scheme):]
            break

    if url_lower.startswith("www."):
        url_lower = url_lower[4:]

    for prefix in sorted(_SOURCE_URL_PREFIX_MAP, key=len, reverse=True):
        if url_lower.startswith(prefix):
            return _SOURCE_URL_PREFIX_MAP[prefix]

    return None


def _match_from_path(path: str) -> Optional[str]:
    if not path:
        return None

    path_lower = path.lower()

    section_map = {
        "/news/uk": "uk",
        "/news/world": "world",
        "/news/business": "business",
        "/news/technology": "tech",
        "/news/science-environment": "science",
        "/news/health": "health",
        "/sport": "sport",
        "/news/entertainment-arts": "entertainment",
        "/section/us": "us",
        "/section/politics": "politics",
        "/section/business": "business",
        "/section/technology": "tech",
        "/section/science": "science",
        "/section/health": "health",
        "/section/sports": "sport",
        "/section/arts": "entertainment",
        "/news/ireland": "ireland",
        "/news/politics": "politics",
        "/uk-news": "uk",
        "/world": "world",
        "/business": "business",
        "/technology": "tech",
        "/science": "science",
        "/sport": "sport",
        "/culture": "entertainment",
        "/environment": "environment",
        "/ireland": "ireland",
        "/crime-law": "crime",
        "/crime-law/courts": "crime",
        "/courts": "crime",
        "/politics": "politics",
        "/opinion": "opinion",
        "/analysis": "analysis",
        "/us-news": "us",
        "/us": "us",
        "/uk": "uk",
        "/europe": "europe",
        "/health": "health",
    }

    parts = [p for p in path_lower.split("/") if p]
    for i in range(min(4, len(parts)), 0, -1):
        key = "/" + "/".join(parts[:i])
        if key in section_map:
            return section_map[key]

    if any(seg in path_lower for seg in ["politics", "election", "government"]):
        return "politics"
    if any(
        seg in path_lower
        for seg in ["world", "international", "europe", "africa", "asia"]
    ):
        return "world"
    if any(seg in path_lower for seg in ["business", "markets", "economy", "finance"]):
        return "business"
    if any(seg in path_lower for seg in ["tech", "technology", "digital"]):
        return "tech"
    if any(
        seg in path_lower
        for seg in ["sport", "sports", "football", "rugby", "tennis", "gaa"]
    ):
        return "sport"
    if any(seg in path_lower for seg in ["entertainment", "culture", "arts", "tv", "film"]):
        return "entertainment"
    if any(seg in path_lower for seg in ["health", "wellbeing", "covid", "medicine"]):
        return "health"
    if any(seg in path_lower for seg in ["science", "environment"]):
        return "science"
    if any(seg in path_lower for seg in ["crime", "courts", "law", "crime-law"]):
        return "crime"
    if any(seg in path_lower for seg in ["opinion", "comment"]):
        return "opinion"
    if any(seg in path_lower for seg in ["us-news", "/us/"]):
        return "us"
    if any(seg in path_lower for seg in ["uk-news", "/uk/"]):
        return "uk"

    return None


def _match_from_title(title: str) -> Optional[str]:
    if not title:
        return None

    t = title.lower()

    if any(
        w in t
        for w in [
            "election", "minister", "government", "parliament",
            "taoiseach", "senate", "congress", "brexit", "treaty",
            "legislation", "referendum", "political", "politician",
            "policy", "bill", "mp ", "td ", "tánaiste", "dáil",
            "seanad", "westminster",
        ]
    ):
        return "politics"

    if any(
        w in t
        for w in [
            "murder", "manslaughter", "convicted", "sentenced", "acquitted",
            "verdict", "trial", "garda", "detective", "arrested", "charged",
            "inquest", "tribunal", "high court", "circuit court", "criminal",
            "prison sentence", "jail", "robbery", "assault", "homicide",
            "gardaí", "court heard", "pleaded guilty",
        ]
    ):
        return "crime"

    if any(
        w in t
        for w in [
            "stocks", "market", "economy", "inflation", "company",
            "bank", "gdp", "trade", "revenue", "profit", "investment",
            "financial", "shares", "nasdaq", "ftse", "dow", "recession",
            "housing", "mortgage", "rent", "property prices", "interest rate",
            "ecb", "federal reserve", "budget", "cost of living",
        ]
    ):
        return "business"

    if any(
        w in t
        for w in [
            "ai", "app", "software", "startup", "technology", "cyber",
            "hack", "data breach", "robot", "drone", "smartphone",
            "chip", "gpu", "openai", "google", "apple", "microsoft",
            "chatgpt", "artificial intelligence", "machine learning",
        ]
    ):
        return "tech"

    if any(
        w in t
        for w in [
            "wins", "defeat", "draw", "tournament", "league", "cup",
            "match", "goal", "score", "player", "manager", "transfer",
            "premier league", "champions league", "fifa", "gaa",
            "all-ireland", "hurling", "camogie", "cricket", "rugby",
            "six nations", "formula 1", "grand prix", "athletics",
            "olympic", "world cup", "championship",
        ]
    ):
        return "sport"

    if any(
        w in t
        for w in [
            "film", "movie", "series", "album", "festival", "celebrity",
            "actor", "actress", "director", "netflix", "disney",
            "spotify", "grammy", "oscar", "bafta", "concert", "theatre",
            "tv show", "box office",
        ]
    ):
        return "entertainment"

    if any(
        w in t
        for w in [
            "covid", "hospital", "vaccine", "health", "nhs", "hse",
            "cancer", "mental health", "drug", "medical", "disease",
            "pandemic", "obesity", "surgery", "patient", "waiting lists",
            "gp shortage", "nursing", "ambulance",
        ]
    ):
        return "health"

    if any(
        w in t
        for w in [
            "climate", "planet", "environment", "research", "study",
            "scientists", "nasa", "space", "species", "carbon",
            "emissions", "biodiversity", "fossil", "renewable", "solar",
            "weather", "earthquake", "volcano",
        ]
    ):
        return "science"

    if any(
        w in t
        for w in [
            "war", "conflict", "troops", "ukraine", "russia", "israel",
            "gaza", "nato", "united nations", "foreign", "diplomacy",
            "sanctions", "refugee", "ambassador", "peace talks",
            "ceasefire", "airstrike", "invasion",
        ]
    ):
        return "world"

    if any(
        w in t
        for w in [
            "white house", "pentagon", "supreme court", "congress",
            "senate", "american", "washington dc", "new york",
            "los angeles", "california", "texas", "florida",
        ]
    ):
        return "us"

    if any(
        w in t
        for w in [
            "downing street", "labour", "conservative", "tory",
            "keir starmer", "rishi sunak", "england", "scotland",
            "wales", "northern ireland", "london mayor",
        ]
    ):
        return "uk"

    if any(
        w in t
        for w in [
            "dublin", "cork", "galway", "traffic",
            "asylum", "direct provision", "ireland",
            "irish water", "luas", "dart",
        ]
    ):
        return "ireland"

    if any(w in t for w in ["opinion", "analysis", "comment", "editorial", "column"]):
        return "opinion"

    return None


def _match_from_title_broad(title: str) -> str:
    """
    Broad fallback keyword scan — catches what tier 2 misses.
    Never returns 'general'. Worst case returns 'world'.
    """
    if not title:
        return "world"

    t = title.lower()

    if any(
        w in t
        for w in [
            "united", "city fc", "match", "final", "champion",
            "olympic", "athlete", "trophy", "fixture", "league cup",
            "transfer", "goal", "wicket", "innings", "formula",
            "race circuit", "penalty", "qualifier",
        ]
    ):
        return "sport"

    if any(
        w in t
        for w in [
            "court", "judge", "jury", "verdict", "sentence", "charged",
            "arrested", "murder", "trial", "lawsuit", "legal",
            "prosecution", "accused", "convicted", "jailed", "bail",
        ]
    ):
        return "crime"

    if any(
        w in t
        for w in [
            "price", "cost", "billion", "million", "fund", "budget",
            "tax", "rate", "growth", "loss", "earnings", "quarter",
            "fiscal", "debt", "interest", "inflation", "gdp",
            "housing", "mortgage", "rent",
        ]
    ):
        return "business"

    if any(
        w in t
        for w in [
            "hospital", "patient", "treatment", "diagnosis", "therapy",
            "nhs", "hse", "gp", "surgery", "clinical", "outbreak",
            "infection", "virus", "illness",
        ]
    ):
        return "health"

    if any(
        w in t
        for w in [
            "app", "tech", "digital", "online", "cyber",
            "software", "hardware", "data", "ai ", "robot",
        ]
    ):
        return "tech"

    if any(
        w in t
        for w in [
            "star", "fans", "award", "show", "premiere", "release",
            "interview", "fashion", "viral", "celebrity",
        ]
    ):
        return "entertainment"

    if any(
        w in t
        for w in [
            "president", "prime minister", "official", "authorities",
            "agency", "committee", "summit", "rally",
            "protest", "military", "army", "attack", "explosion",
            "sanctions", "treaty", "diplomatic",
        ]
    ):
        return "world"

    return "world"


def _classify_with_api(title: str, content: Optional[str] = None) -> str:
    """
    Tier 3: HuggingFace Inference API — zero local memory cost.

    Timeout raised 8s → 20s: bart-large-mnli warm calls complete in
    ~3-8s; cold-start takes 20-40s. 20s catches warm models reliably.

    503 warm-up retry: waits 15s and retries once on model-loading
    503 responses.

    Session reuse: TCP connection reused across calls via _hf_session.
    """
    if not _HF_TOKEN:
        return _match_from_title_broad(title)

    text = f"{title}. {(content or '')[:400]}"

    try:
        session = _get_hf_session()
        response = session.post(
            _HF_API_URL,
            headers={"Authorization": f"Bearer {_HF_TOKEN}"},
            json={
                "inputs": text,
                "parameters": {"candidate_labels": _TIER3_LABELS},
            },
            timeout=20,
        )

        if response.status_code == 503:
            print("bart-large-mnli loading, waiting 15s before retry...")
            response.close()
            time.sleep(15)
            response = session.post(
                _HF_API_URL,
                headers={"Authorization": f"Bearer {_HF_TOKEN}"},
                json={
                    "inputs": text,
                    "parameters": {"candidate_labels": _TIER3_LABELS},
                },
                timeout=20,
            )
            if response.status_code == 503:
                response.close()
                return _match_from_title_broad(title)

        response.raise_for_status()
        return response.json()["labels"][0]

    except Exception:
        return _match_from_title_broad(title)


def infer_category(
    url: Optional[str],
    title: Optional[str],
    content: Optional[str] = None,
) -> str:
    """
    4-tier category inference — zero local ML model loaded.

    Tier 0: Known source URL prefix (most reliable — source-specific)
    Tier 1a: Generic URL path/section parsing (leading-slash fix applied)
    Tier 2: Title keyword match — crime category added, US/UK split
    Tier 3: HuggingFace BART zero-shot OR broad keyword fallback
    Never returns 'general' — worst case is 'world'.
    """
    if url:
        category = _match_from_known_source(url)
        if category:
            return category

    if url:
        try:
            parsed_path = urlparse(url).path or ""
            category = _match_from_path(parsed_path)
            if category:
                return category
        except Exception:
            pass

    if title:
        category = _match_from_title(title)
        if category:
            return category

    return _classify_with_api(title or "", content)

# Test helpers (used by tests/unit/test_categorisation.py)


CATEGORY_GROUP_MAP = {
    "football": "sport",
    "rugby": "sport",
    "tennis": "sport",
    "climate": "science",
    "environment": "science",
    "election": "politics",
    "policy": "politics",
}
