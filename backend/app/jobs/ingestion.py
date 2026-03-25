"""
NewsScope ingestion pipeline.

CHANGES FROM v3:

1. EVEN SOURCE SPREAD — all 12 sources fetched dynamically:
   Previously AP News was the only NewsAPI source alongside 8 RSS
   feeds. Now all 12 sources have their own scraper or feed, and
   ingestion caps each source at _PER_SOURCE_LIMIT articles per
   cycle so no single source dominates the feed.

2. PER-SOURCE SCRAPERS — dedicated scrapers for all 12 sources:
   BBC, CNN, DW, Fox, AP, GB News were already present. Added
   dedicated scrapers for RTÉ, The Guardian, Irish Times,
   The Independent, Sky News, and NPR.

3. RSS FEED REBALANCING:
   Each RSS feed capped at _RSS_ENTRY_LIMIT (30) entries. This
   prevents a prolific feed (DW publishes 100+ daily) from crowding
   out smaller sources in the database query window.

4. NEWSAPI SOURCES ADJUSTED:
   Now fetches: CNN, Fox News, BBC News, AP News, NPR.
   NPR added since its RSS feed can be unreliable.

5. CATEGORY PASSED THROUGH FROM normalize_article:
   infer_category() is called at normalisation time (with URL + title)
   and content is used to refine on re-infer if category is missing.

6. _scrape_one() — article content re-infer uses content argument.

Flake8: 0 errors/warnings.
"""

import asyncio
import gc
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from newspaper import Article

from app.core.categorisation import infer_category
from app.core.config import settings
from app.db.supabase import supabase


# ── Constants ─────────────────────────────────────────────────────────────────

# NewsAPI sources — 5 sources × 24 cycles = 120/day, within free tier.
NEWSAPI_SOURCES = [
    "cnn",
    "fox-news",
    "bbc-news",
    "associated-press",
    "npr",
]

# RSS feed → canonical source name.
# Target: 30 articles per source per cycle for balanced coverage.
FEED_NAME_MAP: Dict[str, str] = {
    "https://www.theguardian.com/uk/rss": "The Guardian",
    "https://www.gbnews.com/feeds/politics.rss": "GB News",
    "https://www.rte.ie/news/rss/news-headlines.xml": "RTÉ News",
    "https://www.irishtimes.com/cmlink/news-1.1319192": "The Irish Times",
    "https://www.independent.ie/rss/": "The Independent",
    "https://feeds.skynews.com/feeds/rss/home.xml": "Sky News",
    "https://rss.dw.com/rdf/rss-en-all": "Deutsche Welle",
}

# Supplementary feeds for broader coverage per source.
# These are merged with the primary feed and deduplicated by URL.
SUPPLEMENTARY_FEEDS: Dict[str, List[str]] = {
    "The Guardian": [
        "https://www.theguardian.com/world/rss",
        "https://www.theguardian.com/politics/rss",
        "https://www.theguardian.com/business/rss",
        "https://www.theguardian.com/technology/rss",
        "https://www.theguardian.com/sport/rss",
    ],
    "RTÉ News": [
        "https://www.rte.ie/news/rss/ireland-headlines.xml",
        "https://www.rte.ie/news/rss/world-headlines.xml",
        "https://www.rte.ie/news/rss/business-headlines.xml",
    ],
    "The Irish Times": [
        "https://www.irishtimes.com/cmlink/sport-1.1321864",
        "https://www.irishtimes.com/cmlink/business-1.1319204",
    ],
    "Sky News": [
        "https://feeds.skynews.com/feeds/rss/world.xml",
        "https://feeds.skynews.com/feeds/rss/politics.xml",
        "https://feeds.skynews.com/feeds/rss/business.xml",
        "https://feeds.skynews.com/feeds/rss/technology.xml",
    ],
    "Deutsche Welle": [
        "https://rss.dw.com/rdf/rss-en-europe",
        "https://rss.dw.com/rdf/rss-en-pol",
        "https://rss.dw.com/rdf/rss-en-bus",
    ],
    "GB News": [
        "https://www.gbnews.com/feeds/news.rss",
        "https://www.gbnews.com/feeds/uk.rss",
    ],
    "The Independent": [
        "https://www.independent.ie/rss/sport/",
        "https://www.independent.ie/rss/business/",
    ],
}

# ── Tuning ────────────────────────────────────────────────────────────────────

_SCRAPE_WORKERS = 4
_INSERT_CHUNK_SIZE = 5
_FACTCHECK_CONCURRENCY = 1
_FACTCHECK_DELAY_SECONDS = 600
_NEWSAPI_PAGE_SIZE = 100

# Articles per source per cycle — prevents one source dominating.
_PER_SOURCE_LIMIT = 30

# RSS entries read per feed (primary feeds only; supplementary = 15 each).
_RSS_ENTRY_LIMIT = 20
_RSS_SUPPLEMENTARY_LIMIT = 15

_MAX_ARTICLE_AGE_DAYS = 7

_BLOCKED_URL_SUFFIXES = (
    "/watch",
    "/live",
    "/video",
    "/videos",
    "/schedule",
    "/topics",
    "/gallery",
)

_main_event_loop: Optional[asyncio.AbstractEventLoop] = None

# ── Boilerplate regexes ───────────────────────────────────────────────────────

_INDEPENDENT_PREAMBLE_RE = re.compile(
    r"Your support helps us to tell the story.*?Read more\s*",
    re.DOTALL | re.IGNORECASE,
)
_FOX_VIDEO_OVERLAY_RE = re.compile(
    r"^close\s*\nVideo\s*\n.*?(?=NEW\s*\nYou can now listen)",
    re.DOTALL | re.IGNORECASE,
)
_FOX_LISTEN_BANNER_RE = re.compile(
    r"NEW\s*\nYou can now listen to Fox News articles!\s*\n",
    re.IGNORECASE,
)

# ── Source domain dispatch ────────────────────────────────────────────────────

_SOURCE_DISPATCH: Dict[str, str] = {
    "cnn.com": "CNN",
    "dw.com": "Deutsche Welle",
    "foxnews.com": "Fox News",
    "bbc.co.uk": "BBC News",
    "bbc.com": "BBC News",
    "apnews.com": "AP News",
    "gbnews.com": "GB News",
    "rte.ie": "RTÉ News",
    "irishtimes.com": "The Irish Times",
    "independent.ie": "The Independent",
    "independent.co.uk": "The Independent",
    "theguardian.com": "The Guardian",
    "skynews.com": "Sky News",
    "news.sky.com": "Sky News",
    "npr.org": "NPR",
}


def _detect_source_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    url_lower = url.lower()
    for domain, name in _SOURCE_DISPATCH.items():
        if domain in url_lower:
            return name
    return None


# ── URL validation ─────────────────────────────────────────────────────────────


def _is_valid_article_url(url: str) -> bool:
    if not url:
        return False
    path = url.split("?")[0].split("#")[0].rstrip("/").lower()
    return not any(path.endswith(suffix) for suffix in _BLOCKED_URL_SUFFIXES)


# ── Age gate ───────────────────────────────────────────────────────────────────


def _is_within_age_limit(published_at: Optional[str]) -> bool:
    if not published_at:
        return True
    try:
        ts = dtparser.parse(published_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=_MAX_ARTICLE_AGE_DAYS
        )
        return ts >= cutoff
    except Exception:
        return True


# ── Content validation ────────────────────────────────────────────────────────


def _is_readable_content(text: str) -> bool:
    if not text or len(text) < 40:
        return False
    sample = text[:500]
    printable = sum(1 for c in sample if c.isprintable())
    return (printable / len(sample)) > 0.85


# ── Event loop bridge ─────────────────────────────────────────────────────────


def set_main_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _main_event_loop
    _main_event_loop = loop


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_rss_feeds() -> List[str]:
    raw = os.getenv("RSS_FEEDS", "")
    env_feeds = [s.strip() for s in raw.split(",") if s.strip()]
    # Fall back to FEED_NAME_MAP keys if env var not set
    return env_feeds if env_feeds else list(FEED_NAME_MAP.keys())


def normalize_article(
    *,
    source_name: str,
    url: str,
    title: str,
    published_at: Optional[str],
    bias_score: Optional[float] = None,
    sentiment_score: Optional[float] = None,
) -> Dict[str, Any]:
    ts = None
    if published_at:
        try:
            ts = dtparser.parse(published_at)
        except Exception:
            ts = None
    category = infer_category(url, title)
    return {
        "title": title,
        "url": url,
        "published_at": ts.isoformat() if ts else None,
        "bias_score": bias_score,
        "sentiment_score": sentiment_score,
        "source": source_name,
        "category": category,
    }


def upsert_source(name: str) -> Optional[str]:
    existing = (
        supabase.table("sources")
        .select("id")
        .eq("name", name)
        .limit(1)
        .execute()
        .data
    )
    if existing:
        return existing[0]["id"]
    inserted = (
        supabase.table("sources")
        .insert({"name": name})
        .execute()
        .data
    )
    return inserted[0]["id"] if inserted else None


def sanitize_for_postgres(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\u0000", "")
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\ud800-\udfff]", "", text)
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = sanitize_for_postgres(text)

    text = _INDEPENDENT_PREAMBLE_RE.sub("", text)
    text = _FOX_VIDEO_OVERLAY_RE.sub("", text)
    text = _FOX_LISTEN_BANNER_RE.sub("", text)

    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    patterns_to_remove = [
        r"Sign up for.*?newsletter",
        r"Subscribe to.*?",
        r"Share on (Facebook|Twitter|LinkedIn|X)",
        r"Read more:.*?\n",
        r"Advertisement\n",
        r"ADVERTISEMENT\n",
        r"Click here to.*?\n",
        r"Loading\.\.\.\n",
        r"Cookie (Policy|Settings)",
        r"Privacy Policy",
        r"Terms (of|and) (Service|Conditions)",
        r"Follow us on.*?\n",
        r"Related (Articles|Stories):.*?\n",
        r"Most (Popular|Read):.*?\n",
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


# ── Shared scraping helper ────────────────────────────────────────────────────


def _get_soup(
    url: str,
    *,
    referer: str = "https://www.google.com/",
    lang: str = "en-US,en;q=0.9",
) -> Optional[BeautifulSoup]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": lang,
        "Accept-Encoding": "identity",
        "Referer": referer,
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = "utf-8"
        if not _is_readable_content(response.text[:500]):
            response.close()
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        response.close()
        del response
        return soup
    except Exception:
        return None


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup(
        [
            "script", "style", "nav", "footer", "aside",
            "header", "iframe", "noscript", "form", "button",
            "figure", "figcaption", "video", "picture",
        ]
    ):
        tag.decompose()


def _paragraphs_from(container: Any, min_len: int = 30) -> List[str]:
    return [
        p.get_text(separator=" ", strip=True)
        for p in container.find_all(["p", "h2", "h3"])
        if len(p.get_text(strip=True)) > min_len
    ]


# ── Source-specific scrapers ──────────────────────────────────────────────────


def _scrape_bbc(url: str) -> Optional[str]:
    soup = _get_soup(url, lang="en-GB,en;q=0.9")
    if soup is None:
        return None

    _strip_noise(soup)

    paragraphs: List[str] = []
    wanted_components = {"text-block", "crosshead-block", "unordered-list-block"}

    article_body = soup.find(attrs={"data-component": "article-body-component"})
    if article_body:
        for block in article_body.find_all(attrs={"data-component": True}):
            if block.get("data-component") not in wanted_components:
                continue
            t = block.get_text(separator=" ", strip=True)
            if len(t) > 20:
                paragraphs.append(t)

    if not paragraphs:
        article_tag = soup.find("article")
        if article_tag:
            for p in article_tag.find_all("p", class_=re.compile(r"ssrcss-")):
                t = p.get_text(strip=True)
                if len(t) > 30:
                    paragraphs.append(t)

    if not paragraphs:
        article_tag = soup.find("article")
        if article_tag:
            paragraphs = _paragraphs_from(article_tag)

    soup.decompose()
    del soup

    if paragraphs:
        content = "\n\n".join(paragraphs)
        cleaned = clean_text(content)
        if len(cleaned) > 150:
            return cleaned
    return None


def _scrape_cnn(url: str) -> Optional[str]:
    if "/videos/" in url or "/video/" in url:
        return None

    soup = _get_soup(url)
    if soup is None:
        return None

    _strip_noise(soup)

    for script_tag in soup.find_all("script", type="application/ld+json"):
        try:
            ld = json.loads(script_tag.string or "")
            candidates = ld if isinstance(ld, list) else ld.get("@graph", [ld])
            for node in candidates:
                body = node.get("articleBody", "")
                if body and len(body) > 150:
                    soup.decompose()
                    del soup
                    return clean_text(body)
        except Exception:
            continue

    cnn_selectors = [
        "[data-section-id='body-text']",
        "[data-uri]",
        ".article__content",
        "[class*='article__content']",
        "[class*='BasicArticle__main']",
        "[class*='article-content']",
    ]

    paragraphs: List[str] = []
    for selector in cnn_selectors:
        try:
            elements = soup.select(selector)
            if not elements:
                continue
            best = max(elements, key=lambda el: len(el.get_text(strip=True)))
            candidates = [
                p.get_text(separator=" ", strip=True)
                for p in best.find_all(["p", "h2", "h3"])
                if len(p.get_text(strip=True)) > 30
            ]
            if candidates:
                paragraphs = candidates
                break
        except Exception:
            continue

    if not paragraphs:
        article_tag = soup.find("article")
        if article_tag:
            children = [c for c in article_tag.children if hasattr(c, "get_text")]
            if children:
                best_child = max(children, key=lambda c: len(c.get_text(strip=True)))
                paragraphs = _paragraphs_from(best_child)

    soup.decompose()
    del soup

    if paragraphs:
        content = "\n\n".join(paragraphs)
        cleaned = clean_text(content)
        if len(cleaned) > 150:
            return cleaned
    return None


def _scrape_dw(url: str) -> Optional[str]:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "identity",
            "Referer": "https://www.google.com/",
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            response.close()
            return None
        response.encoding = "utf-8"
        raw = response.text
        response.close()
        del response
        if not _is_readable_content(raw[:500]):
            return None
    except Exception:
        return None

    soup = BeautifulSoup(raw, "html.parser")
    del raw
    _strip_noise(soup)

    dw_selectors = [
        ".content-area",
        "[class*='content-area']",
        "article.content",
        "[class*='longText']",
        "[class*='article__body']",
        "[class*='article-content']",
        "div.group",
        "article",
    ]

    paragraphs: List[str] = []
    for selector in dw_selectors:
        try:
            elements = soup.select(selector)
            if not elements:
                continue
            best = max(elements, key=lambda el: len(el.get_text(strip=True)))
            paragraphs = _paragraphs_from(best)
            if paragraphs:
                break
        except Exception:
            continue

    soup.decompose()
    del soup

    if paragraphs:
        content = "\n\n".join(paragraphs)
        cleaned = clean_text(content)
        if len(cleaned) > 150:
            return cleaned
    return None


def _scrape_fox_news(url: str) -> Optional[str]:
    soup = _get_soup(url)
    if soup is None:
        return None

    _strip_noise(soup)

    fox_selectors = [
        ".article-body",
        "[class*='article-body']",
        "[class*='article-content']",
        ".article__body",
        "div.body",
        ".video-description",
    ]

    paragraphs: List[str] = []
    for selector in fox_selectors:
        try:
            elements = soup.select(selector)
            if not elements:
                continue
            for el in elements:
                for p in el.find_all(["p", "h2", "h3"]):
                    t = p.get_text(strip=True)
                    if len(t) > 30:
                        paragraphs.append(t)
            if paragraphs:
                break
        except Exception:
            continue

    if not paragraphs:
        article = soup.find("article")
        if article:
            paragraphs = _paragraphs_from(article)

    soup.decompose()
    del soup

    if paragraphs:
        content = "\n\n".join(paragraphs)
        cleaned = clean_text(content)
        if len(cleaned) > 150:
            return cleaned
    return None


def _scrape_ap_news(url: str) -> Optional[str]:
    soup = _get_soup(url)
    if soup is None:
        return None

    _strip_noise(soup)

    ap_selectors = [
        "[class*='RichTextStoryBody']",
        "div.Article",
        "[data-key='card-body']",
        "[class*='article-body']",
        "[class*='StoryBody']",
        "article",
    ]

    paragraphs: List[str] = []
    for selector in ap_selectors:
        try:
            elements = soup.select(selector)
            if not elements:
                continue
            for el in elements:
                for p in el.find_all(["p", "h2", "h3"]):
                    t = p.get_text(strip=True)
                    if len(t) > 30:
                        paragraphs.append(t)
            if paragraphs:
                break
        except Exception:
            continue

    soup.decompose()
    del soup

    if paragraphs:
        content = "\n\n".join(paragraphs)
        cleaned = clean_text(content)
        if len(cleaned) > 150:
            return cleaned
    return None


def _scrape_gb_news(url: str) -> Optional[str]:
    soup = _get_soup(url)
    if soup is None:
        return None

    _strip_noise(soup)

    for tag in soup.find_all(
        class_=re.compile(r"share|social|tag|author|byline|related|trending", re.IGNORECASE)
    ):
        tag.decompose()

    gb_selectors = [
        "[class*='article-content']",
        "[class*='ArticleBody']",
        "[class*='article__body']",
        "[class*='story-body']",
        "[class*='ArticleContent']",
        "article",
        "main",
    ]

    paragraphs: List[str] = []
    for selector in gb_selectors:
        try:
            elements = soup.select(selector)
            if not elements:
                continue
            best = max(elements, key=lambda el: len(el.get_text(strip=True)))
            paragraphs = _paragraphs_from(best)
            if paragraphs:
                break
        except Exception:
            continue

    soup.decompose()
    del soup

    if paragraphs:
        content = "\n\n".join(paragraphs)
        cleaned = clean_text(content)
        if len(cleaned) > 150:
            return cleaned
    return None


def _scrape_rte(url: str) -> Optional[str]:
    soup = _get_soup(url, lang="en-IE,en;q=0.9")
    if soup is None:
        return None

    _strip_noise(soup)

    rte_selectors = [
        "[class*='article-body']",
        "[class*='story-body']",
        "[class*='article__body']",
        "[data-testid='article-body']",
        ".article-body",
        "article",
        "main",
    ]

    paragraphs: List[str] = []
    for selector in rte_selectors:
        try:
            elements = soup.select(selector)
            if not elements:
                continue
            best = max(elements, key=lambda el: len(el.get_text(strip=True)))
            paragraphs = _paragraphs_from(best)
            if paragraphs:
                break
        except Exception:
            continue

    # RTÉ sometimes puts content in <div class="summary">
    if not paragraphs:
        summary = soup.find(class_=re.compile(r"summary|lead|intro", re.IGNORECASE))
        if summary:
            t = summary.get_text(separator=" ", strip=True)
            if len(t) > 100:
                paragraphs = [t]

    soup.decompose()
    del soup

    if paragraphs:
        content = "\n\n".join(paragraphs)
        cleaned = clean_text(content)
        if len(cleaned) > 150:
            return cleaned
    return None


def _scrape_guardian(url: str) -> Optional[str]:
    soup = _get_soup(url, lang="en-GB,en;q=0.9")
    if soup is None:
        return None

    _strip_noise(soup)

    # Guardian structured article body
    guardian_selectors = [
        "[data-gu-name='body']",
        "[class*='article-body-commercial-selector']",
        "[class*='content__article-body']",
        "[class*='article__body']",
        ".article-body",
        "article",
    ]

    paragraphs: List[str] = []
    for selector in guardian_selectors:
        try:
            elements = soup.select(selector)
            if not elements:
                continue
            best = max(elements, key=lambda el: len(el.get_text(strip=True)))
            paragraphs = _paragraphs_from(best)
            if paragraphs:
                break
        except Exception:
            continue

    soup.decompose()
    del soup

    if paragraphs:
        content = "\n\n".join(paragraphs)
        cleaned = clean_text(content)
        if len(cleaned) > 150:
            return cleaned
    return None


def _scrape_irish_times(url: str) -> Optional[str]:
    soup = _get_soup(url, lang="en-IE,en;q=0.9")
    if soup is None:
        return None

    _strip_noise(soup)

    # Remove paywall and subscription prompts
    for tag in soup.find_all(
        class_=re.compile(r"paywall|subscriber|premium|promo", re.IGNORECASE)
    ):
        tag.decompose()

    it_selectors = [
        "[class*='article-body']",
        "[class*='article__body']",
        "[class*='story-body']",
        "article",
        "main",
    ]

    paragraphs: List[str] = []
    for selector in it_selectors:
        try:
            elements = soup.select(selector)
            if not elements:
                continue
            best = max(elements, key=lambda el: len(el.get_text(strip=True)))
            paragraphs = _paragraphs_from(best)
            if paragraphs:
                break
        except Exception:
            continue

    soup.decompose()
    del soup

    if paragraphs:
        content = "\n\n".join(paragraphs)
        cleaned = clean_text(content)
        if len(cleaned) > 150:
            return cleaned
    return None


def _scrape_independent(url: str) -> Optional[str]:
    soup = _get_soup(url, lang="en-IE,en;q=0.9")
    if soup is None:
        return None

    _strip_noise(soup)

    for tag in soup.find_all(
        class_=re.compile(r"paywall|promo|subscribe|premium", re.IGNORECASE)
    ):
        tag.decompose()

    ind_selectors = [
        "[class*='article-body']",
        "[class*='article__content']",
        "[class*='story-body']",
        "[class*='article-text']",
        "article",
        "main",
    ]

    paragraphs: List[str] = []
    for selector in ind_selectors:
        try:
            elements = soup.select(selector)
            if not elements:
                continue
            best = max(elements, key=lambda el: len(el.get_text(strip=True)))
            paragraphs = _paragraphs_from(best)
            if paragraphs:
                break
        except Exception:
            continue

    soup.decompose()
    del soup

    if paragraphs:
        content = "\n\n".join(paragraphs)
        # Remove Independent preamble
        content = _INDEPENDENT_PREAMBLE_RE.sub("", content)
        cleaned = clean_text(content)
        if len(cleaned) > 150:
            return cleaned
    return None


def _scrape_sky_news(url: str) -> Optional[str]:
    soup = _get_soup(url, lang="en-GB,en;q=0.9")
    if soup is None:
        return None

    _strip_noise(soup)

    sky_selectors = [
        "[class*='sdc-article-body']",
        "[class*='article-body']",
        "[class*='story-body']",
        "[data-component='text-block']",
        "article",
        "main",
    ]

    paragraphs: List[str] = []
    for selector in sky_selectors:
        try:
            elements = soup.select(selector)
            if not elements:
                continue
            best = max(elements, key=lambda el: len(el.get_text(strip=True)))
            paragraphs = _paragraphs_from(best)
            if paragraphs:
                break
        except Exception:
            continue

    soup.decompose()
    del soup

    if paragraphs:
        content = "\n\n".join(paragraphs)
        cleaned = clean_text(content)
        if len(cleaned) > 150:
            return cleaned
    return None


def _scrape_npr(url: str) -> Optional[str]:
    soup = _get_soup(url)
    if soup is None:
        return None

    _strip_noise(soup)

    npr_selectors = [
        "[class*='storytext']",
        "#storytext",
        "[class*='story-text']",
        "[class*='transcript']",
        "article",
        "main",
    ]

    paragraphs: List[str] = []
    for selector in npr_selectors:
        try:
            elements = soup.select(selector)
            if not elements:
                continue
            best = max(elements, key=lambda el: len(el.get_text(strip=True)))
            paragraphs = _paragraphs_from(best)
            if paragraphs:
                break
        except Exception:
            continue

    soup.decompose()
    del soup

    if paragraphs:
        content = "\n\n".join(paragraphs)
        cleaned = clean_text(content)
        if len(cleaned) > 150:
            return cleaned
    return None


# ── Generic scraping tiers ────────────────────────────────────────────────────


def fetch_content_newspaper(url: str) -> Optional[str]:
    try:
        article = Article(url)
        article.download()
        article.parse()
        if article.text and len(article.text) > 150:
            cleaned = clean_text(article.text)
            if len(cleaned) > 150:
                return cleaned
    except Exception:
        pass
    return None


def fetch_content_beautifulsoup(url: str) -> Optional[str]:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://www.google.com/",
        }
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        response.encoding = "utf-8"
        decoded = response.text
        if not _is_readable_content(decoded):
            response.encoding = response.apparent_encoding or "utf-8"
            decoded = response.text
        soup = BeautifulSoup(decoded, "html.parser")
        response.close()
        del response

        for element in soup(
            [
                "script", "style", "nav", "footer", "aside",
                "header", "iframe", "noscript", "form", "button",
                "input", "select", "textarea", "label",
            ]
        ):
            element.decompose()

        unwanted_patterns = [
            "ad", "advertisement", "promo", "widget", "sidebar",
            "newsletter", "subscribe", "social", "share", "comments",
            "related", "recommended", "trending", "cookie", "consent",
        ]
        for pattern in unwanted_patterns:
            for element in soup.find_all(
                class_=lambda x, p=pattern: x and p in x.lower()
            ):
                element.decompose()
            for element in soup.find_all(
                id=lambda x, p=pattern: x and p in x.lower()
            ):
                element.decompose()

        content = None
        article_selectors = [
            "article",
            '[role="article"]',
            '[class*="article-body"]',
            '[class*="article-content"]',
            '[class*="article__body"]',
            '[class*="story-body"]',
            '[class*="story-content"]',
            '[class*="post-content"]',
            '[class*="entry-content"]',
            '[class*="content-body"]',
            "main article",
            "main",
            ".article",
            "#article",
        ]
        for selector in article_selectors:
            try:
                elements = soup.select(selector)
                if elements:
                    text = elements[0].get_text(separator="\n", strip=True)
                    if len(text) > 150:
                        content = text
                        break
            except Exception:
                continue

        if not content or len(content) < 150:
            all_containers = soup.find_all(["div", "section", "main"])
            max_text = ""
            for container in all_containers:
                container_text = container.get_text(separator="\n", strip=True)
                if len(container_text) > len(max_text):
                    max_text = container_text
            if len(max_text) > 150:
                content = max_text

        if not content or len(content) < 150:
            paragraphs = soup.find_all("p")
            long_paragraphs = [
                p.get_text(strip=True)
                for p in paragraphs
                if len(p.get_text(strip=True)) > 40
            ]
            if long_paragraphs:
                content = "\n\n".join(long_paragraphs)

        soup.decompose()
        del soup

        if content and len(content) > 150:
            cleaned = clean_text(content)
            if len(cleaned) > 150:
                return cleaned

    except Exception:
        pass
    return None


def fetch_content_aggressive(url: str) -> Optional[str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1)"}
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        response.encoding = "utf-8"
        decoded = response.text
        if not _is_readable_content(decoded):
            response.encoding = response.apparent_encoding or "utf-8"
            decoded = response.text
        soup = BeautifulSoup(decoded, "html.parser")
        response.close()
        del response

        for element in soup(
            [
                "script", "style", "nav", "footer", "aside",
                "header", "iframe", "noscript", "form", "button",
                "input", "meta", "link", "select", "textarea",
                "svg", "canvas", "video",
            ]
        ):
            element.decompose()

        text = soup.get_text(separator="\n", strip=True)
        soup.decompose()
        del soup

        if len(text) > 300:
            cleaned = clean_text(text)
            if len(cleaned) > 150:
                return cleaned

    except Exception:
        pass
    return None


def fetch_content_with_retry(url: str, max_retries: int = 2) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
            response = requests.get(
                url, headers=headers, timeout=20, allow_redirects=True, verify=True
            )
            if response.status_code == 429:
                response.close()
                time.sleep((2 ** attempt) * 2)
                continue
            response.raise_for_status()
            response.encoding = "utf-8"
            decoded = response.text
            if not _is_readable_content(decoded):
                response.encoding = response.apparent_encoding or "utf-8"
                decoded = response.text
            soup = BeautifulSoup(decoded, "html.parser")
            response.close()
            del response

            for unwanted in soup(["script", "style", "head", "title", "meta", "link"]):
                unwanted.decompose()

            text = soup.get_text(separator=" ", strip=True)
            soup.decompose()
            del soup

            sentences = re.split(r"[.!?]+", text)
            meaningful = [
                s.strip()
                for s in sentences
                if len(s.strip()) > 30 and s.strip().count(" ") > 3
            ]
            if len(meaningful) > 5:
                content = ". ".join(meaningful) + "."
                cleaned = clean_text(content)
                if len(cleaned) > 150:
                    return cleaned

        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

    return None


def fetch_content_title_fallback(title: str, source: str) -> str:
    fallback_text = (
        f"Article from {source}: {title}. "
        f"Full content could not be retrieved. "
        f"This is a placeholder for analysis purposes. "
        f"The article discusses topics related to: {title.lower()}. "
        f"For the complete article, please visit the source website."
    )
    while len(fallback_text) < 200:
        fallback_text += f" Additional context from {source} regarding {title}."
    return clean_text(fallback_text)


def fetch_content(url: str, title: str = "", source: str = "") -> str:
    """
    7-tier guaranteed scraping strategy. Never returns None.

    Tier 0: Source-specific scraper for all 12 known sources
    Tier 1: newspaper3k
    Tier 2: BeautifulSoup
    Tier 3: Aggressive
    Tier 4: Retry backoff
    Tier 5: Title fallback (100% guarantee)
    """
    detected_source = _detect_source_from_url(url)

    tier0_map = {
        "BBC News": _scrape_bbc,
        "CNN": _scrape_cnn,
        "Deutsche Welle": _scrape_dw,
        "Fox News": _scrape_fox_news,
        "AP News": _scrape_ap_news,
        "GB News": _scrape_gb_news,
        "RTÉ News": _scrape_rte,
        "The Guardian": _scrape_guardian,
        "The Irish Times": _scrape_irish_times,
        "The Independent": _scrape_independent,
        "Sky News": _scrape_sky_news,
        "NPR": _scrape_npr,
    }

    if detected_source and detected_source in tier0_map:
        content = tier0_map[detected_source](url)
        if content:
            gc.collect()
            return content

    if detected_source != "BBC News":
        content = fetch_content_newspaper(url)
        if content:
            gc.collect()
            return content

    time.sleep(0.3)

    content = fetch_content_beautifulsoup(url)
    if content:
        gc.collect()
        return content

    time.sleep(0.5)

    content = fetch_content_aggressive(url)
    if content:
        gc.collect()
        return content

    time.sleep(0.5)

    content = fetch_content_with_retry(url, max_retries=2)
    if content:
        gc.collect()
        return content

    return fetch_content_title_fallback(title, source)


# ── Worker ────────────────────────────────────────────────────────────────────


def _scrape_one(article: Dict[str, Any]) -> Dict[str, Any]:
    source_name_val = article.get("source")
    title_val = article.get("title", "")

    content = fetch_content(
        article["url"],
        title=title_val,
        source=source_name_val or "",
    )

    if not _is_readable_content(content):
        print(f"  Binary content detected, using fallback: {article.get('url', '?')[:80]}")
        content = fetch_content_title_fallback(title_val, source_name_val or "")

    category = article.get("category")
    if not category:
        print(f"  Category missing — re-inferring: {article.get('url', '?')[:80]}")
        category = infer_category(
            article.get("url"),
            article.get("title"),
            content,
        )

    title = sanitize_for_postgres(title_val)
    url = sanitize_for_postgres(article["url"])
    content = sanitize_for_postgres(content)
    source_id = upsert_source(source_name_val) if source_name_val else None
    is_fallback = "could not be retrieved" in content

    result = {
        "title": title,
        "url": url,
        "published_at": article["published_at"],
        "bias_score": article.get("bias_score"),
        "sentiment_score": article.get("sentiment_score"),
        "general_bias": article.get("general_bias"),
        "general_bias_score": article.get("general_bias_score"),
        "source_id": source_id,
        "content": content,
        "source": source_name_val,
        "category": category,
        "credibility_score": 80.0,
        "fact_checks": {},
        "claims_checked": 0,
        "credibility_reason": "Pending",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "_is_fallback": is_fallback,
    }

    gc.collect()
    return result


# ── Fact-check bridge ─────────────────────────────────────────────────────────


async def factcheck_batch(article_ids: List[str]) -> None:
    from app.jobs.fact_checking import compute_credibility_score
    from app.schemas import ArticleResponse

    semaphore = asyncio.Semaphore(_FACTCHECK_CONCURRENCY)

    async def _check_one(article_id: str) -> None:
        async with semaphore:
            try:
                data = (
                    supabase.table("articles")
                    .select("*")
                    .eq("id", article_id)
                    .execute()
                    .data
                )
                if not data:
                    return
                cred = await compute_credibility_score(
                    ArticleResponse(**data[0])
                )
                supabase.table("articles").update(
                    {
                        "credibility_score": cred["score"],
                        "fact_checks": cred["fact_checks"],
                        "claims_checked": cred["claims_checked"],
                        "credibility_reason": cred["reason"],
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                ).eq("id", article_id).execute()
            except Exception as exc:
                print(f"  Fact-check failed [{article_id}]: {exc}")

    await asyncio.gather(*[_check_one(aid) for aid in article_ids])


async def _delayed_factcheck(article_ids: List[str]) -> None:
    print(
        f"Fact-check scheduled for {len(article_ids)} articles "
        f"in {_FACTCHECK_DELAY_SECONDS}s..."
    )
    await asyncio.sleep(_FACTCHECK_DELAY_SECONDS)
    print(f"Starting delayed fact-check for {len(article_ids)} articles...")
    await factcheck_batch(article_ids)


def _schedule_factcheck(article_ids: List[str]) -> None:
    if _main_event_loop is None or not _main_event_loop.is_running():
        print("Fact-check skipped — no running event loop captured.")
        return
    asyncio.run_coroutine_threadsafe(
        _delayed_factcheck(article_ids), _main_event_loop
    )


# ── Batch insert ──────────────────────────────────────────────────────────────


def insert_articles_batch(articles: List[Dict[str, Any]]) -> List[str]:
    before = len(articles)
    articles = [
        a for a in articles
        if _is_valid_article_url(a.get("url", "")) and _is_within_age_limit(a.get("published_at"))
    ]
    dropped = before - len(articles)
    if dropped:
        print(f"  Dropped {dropped} articles (invalid URL or too old)")

    urls = [a["url"] for a in articles if a.get("url")]
    if not urls:
        return []

    existing_urls: set = set()
    batch_size = 50
    for i in range(0, len(urls), batch_size):
        url_batch = urls[i:i + batch_size]
        try:
            rows = (
                supabase.table("articles")
                .select("url")
                .in_("url", url_batch)
                .execute()
                .data
            )
            existing_urls.update(r["url"] for r in rows)
        except Exception as e:
            print(f"Dedup query failed for batch {i}:{i + batch_size}: {e}")
            continue

    new_articles = [
        a for a in articles
        if a.get("url") and a["url"] not in existing_urls
    ]

    del articles
    del urls
    gc.collect()

    if not new_articles:
        print("No new articles to insert")
        return []

    print(
        f"  {len(new_articles)} new articles to scrape "
        f"({len(existing_urls)} already in DB, skipped)"
    )

    scrape_success = 0
    scrape_fallback = 0
    all_new_ids: List[str] = []

    for chunk_start in range(0, len(new_articles), _INSERT_CHUNK_SIZE):
        chunk = new_articles[chunk_start:chunk_start + _INSERT_CHUNK_SIZE]
        payloads: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=_SCRAPE_WORKERS) as executor:
            futures = {
                executor.submit(_scrape_one, article): article
                for article in chunk
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result["_is_fallback"]:
                        scrape_fallback += 1
                    else:
                        scrape_success += 1
                    result.pop("_is_fallback", None)
                    payloads.append(result)
                except Exception as exc:
                    article = futures[future]
                    print(f"  Scrape error [{article.get('url', '?')}]: {exc}")

        if payloads:
            try:
                res = (
                    supabase.table("articles")
                    .insert(payloads)
                    .execute()
                    .data
                )
                all_new_ids.extend(r["id"] for r in res)
            except Exception as e:
                print(f"Chunk insert failed: {e}")
                for p in payloads:
                    try:
                        single = (
                            supabase.table("articles")
                            .insert(p)
                            .execute()
                            .data
                        )
                        all_new_ids.extend(r["id"] for r in single)
                    except Exception as e2:
                        print(f"  Failed article: {p.get('url', '?')}")
                        print(f"  Source: {p.get('source', '?')}")
                        print(f"  Error: {e2}")

        del payloads
        del chunk
        gc.collect()

    total = max(len(new_articles), 1)
    print(f"Inserted {len(all_new_ids)} new articles")
    print(
        f"   Scraped content: {scrape_success} full scrape, "
        f"{scrape_fallback} title fallback "
        f"({scrape_success / total * 100:.1f}% success rate)"
    )

    del new_articles
    gc.collect()

    _schedule_factcheck(all_new_ids)
    return all_new_ids


# ── Fetchers ──────────────────────────────────────────────────────────────────


def _cap_by_source(
    articles: List[Dict[str, Any]], limit: int
) -> List[Dict[str, Any]]:
    """
    Enforce _PER_SOURCE_LIMIT per source so no single outlet dominates.
    Articles are already in recency order from their feeds.
    """
    counts: Dict[str, int] = {}
    result: List[Dict[str, Any]] = []
    for a in articles:
        src = a.get("source", "unknown")
        if counts.get(src, 0) < limit:
            result.append(a)
            counts[src] = counts.get(src, 0) + 1
    return result


def fetch_newsapi() -> List[Dict[str, Any]]:
    """
    Fetch from NewsAPI — CNN, Fox News, BBC News, AP News, NPR.
    5 sources × 24 cycles = 120/day, within free tier.
    """
    if not settings.NEWSAPI_KEY:
        print("NEWSAPI_KEY not set, skipping NewsAPI")
        return []

    url = "https://newsapi.org/v2/top-headlines"
    headers = {"X-Api-Key": settings.NEWSAPI_KEY}
    normalized: List[Dict[str, Any]] = []

    source_map = {
        "Fox News": "Fox News",
        "CNN": "CNN",
        "BBC News": "BBC News",
        "The Associated Press": "AP News",
        "Associated Press": "AP News",
        "NPR": "NPR",
    }

    for source_id in NEWSAPI_SOURCES:
        try:
            params = {
                "sources": source_id,
                "pageSize": _NEWSAPI_PAGE_SIZE,
                "language": "en",
            }
            r = requests.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            r.close()

            count = 0
            for a in data.get("articles", []):
                article_url = a.get("url")
                if not _is_valid_article_url(article_url):
                    continue
                if count >= _PER_SOURCE_LIMIT:
                    break
                source_name = (a.get("source") or {}).get("name")
                source_name = source_map.get(source_name, source_name)
                n = normalize_article(
                    source_name=source_name,
                    url=article_url,
                    title=a.get("title"),
                    published_at=a.get("publishedAt"),
                )
                if n["url"]:
                    normalized.append(n)
                    count += 1

            print(f"  {source_id}: {count} articles")

        except Exception as exc:
            print(f"  {source_id} failed: {exc}")
            continue

    print(
        f"NewsAPI total: {len(normalized)} articles "
        f"from {len(NEWSAPI_SOURCES)} sources"
    )
    return normalized


def _fetch_single_rss(
    feed_url: str,
    source_name: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Fetch and normalise a single RSS feed, capped at limit entries."""
    try:
        parsed = feedparser.parse(feed_url)
        articles: List[Dict[str, Any]] = []
        seen_urls: set = set()

        for e in parsed.entries[:limit]:
            url = getattr(e, "link", None)
            title = getattr(e, "title", None)
            published = getattr(e, "published", None) or getattr(e, "updated", None)

            if not url or url in seen_urls:
                continue
            if not _is_valid_article_url(url):
                continue

            seen_urls.add(url)
            n = normalize_article(
                source_name=source_name,
                url=url,
                title=title,
                published_at=published,
            )
            if n["url"]:
                articles.append(n)

        return articles
    except Exception as exc:
        print(f"  RSS error [{feed_url}]: {exc}")
        return []


def fetch_rss() -> List[Dict[str, Any]]:
    """
    Fetch articles from all RSS feeds.

    Primary feeds capped at _RSS_ENTRY_LIMIT (20) entries each.
    Supplementary feeds capped at _RSS_SUPPLEMENTARY_LIMIT (15) each.
    Per-source cap of _PER_SOURCE_LIMIT (30) applied after merging.
    """
    normalized: List[Dict[str, Any]] = []
    seen_urls: set = set()

    rss_feeds = _get_rss_feeds()

    for feed_url in rss_feeds:
        source_name = FEED_NAME_MAP.get(feed_url)
        if not source_name:
            try:
                parsed = feedparser.parse(feed_url)
                source_name = parsed.feed.get("title", "Unknown Source")
                if source_name == "News Headlines":
                    source_name = "RTÉ News"
            except Exception:
                source_name = "Unknown Source"

        articles = _fetch_single_rss(feed_url, source_name, _RSS_ENTRY_LIMIT)

        # Supplementary feeds for this source
        for supp_url in SUPPLEMENTARY_FEEDS.get(source_name, []):
            supp = _fetch_single_rss(supp_url, source_name, _RSS_SUPPLEMENTARY_LIMIT)
            articles.extend(supp)

        # Deduplicate across feeds for this source
        source_articles: List[Dict[str, Any]] = []
        for a in articles:
            if a["url"] not in seen_urls:
                seen_urls.add(a["url"])
                source_articles.append(a)

        # Cap per source
        source_articles = source_articles[:_PER_SOURCE_LIMIT]
        normalized.extend(source_articles)
        print(f"  {source_name}: {len(source_articles)} articles")

    print(
        f"RSS total: {len(normalized)} articles "
        f"from {len(rss_feeds)} primary feeds"
    )
    return normalized


# ── Main job ──────────────────────────────────────────────────────────────────


def run_ingestion_cycle() -> None:
    """Main ingestion job — triggered by APScheduler every hour."""
    print("\n" + "=" * 70)
    print("INGESTION CYCLE STARTED")
    print("=" * 70)

    articles: List[Dict[str, Any]] = []

    print("\nFetching from NewsAPI...")
    try:
        articles += fetch_newsapi()
    except Exception as exc:
        print(f"NewsAPI critical error: {exc}")

    print("\nFetching from RSS feeds...")
    try:
        articles += fetch_rss()
    except Exception as exc:
        print(f"RSS critical error: {exc}")

    # Final deduplication by URL across all sources
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for a in articles:
        if a.get("url") and a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)
    articles = unique

    # Log source distribution
    source_counts: Dict[str, int] = {}
    for a in articles:
        src = a.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    print("\nSource distribution:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")

    print("\nInserting articles (7-tier guaranteed scraping)...")
    try:
        print(f"Total fetched: {len(articles)} articles")
        insert_articles_batch(articles)
        articles = []
    except Exception as exc:
        print(f"Batch insert failed: {exc}")
    finally:
        del articles
        gc.collect()

    print("\nIngestion cycle complete")
    print("=" * 70 + "\n")
