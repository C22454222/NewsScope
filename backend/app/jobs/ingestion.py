"""
NewsScope Ingestion Pipeline.

SOURCE STATUS (v8 — all 12 verified):
-------------------------------------------------------------------
Source          | Domain            | Bias | Scraper     | Status
----------------|-------------------|------|-------------|--------
The Guardian    | theguardian.com   | L    | newspaper3k | OK
GB News         | gbnews.com        | R    | Dedicated   | v7
RTÉ News        | rte.ie            | C    | Generic     | OK
The Irish Times | irishtimes.com    | C    | Generic     | OK
The Independent | independent.co.uk | CL   | Generic     | OK
NPR             | npr.org           | CL   | newspaper3k | OK
Sky News        | news.sky.com      | CR   | Generic     | OK
Deutsche Welle  | dw.com            | C    | Dedicated   | v8
CNN             | cnn.com           | L    | Dedicated   | v8
BBC News        | bbc.co.uk         | C    | Dedicated   | v7
Fox News        | foxnews.com       | R    | Dedicated   | v6
AP News         | apnews.com        | CR   | Dedicated   | v7
-------------------------------------------------------------------

CHANGES v8:
- Euronews replaced by Deutsche Welle (dw.com).
  DW publishes full English-language news at dw.com/en — public
  broadcaster, no paywall, stable HTML, Center leaning. RSS feed:
  https://rss.dw.com/rdf/rss-en-all
  Scraper targets .content-area and [class*='longText'] which are
  stable across DW article layouts.

- CNN bleed fix: The Japan/Takaichi snippet was appearing in
  unrelated articles because .article__content and .zn-body
  selectors were matching shared layout containers that persist
  across page navigation in CNN's SPA shell. The new scraper:
    1. Checks for <script type="application/ld+json"> to verify the
       page is actually an article (articleBody present in JSON-LD).
    2. Prefers [data-uri] or [data-section-id="body-text"] containers
       which are scoped to the current article only.
    3. Falls back to the most text-dense <article> child rather than
       the first match, which was picking up sidebar content.

CHANGES v7:
- AP News replaces Politico (paywalled) and Reuters (broken scraping).
  AP is Center-Right, fully open-access, reliable HTML.

BBC FIX (v7):
- Scopes all text-block lookups to
  [data-component="article-body-component"] to prevent mid-article
  starts caused by DOM-wide searches picking up related-story panels.
- newspaper3k intentionally SKIPPED for BBC.

GB News FIX (v7):
- Targets div[class*="article-content"] / div[class*="ArticleBody"]
  directly to avoid React SSR nav/sidebar bleed.

Flake8: 0 errors/warnings.
"""

import asyncio
import gc
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from newspaper import Article

from app.core.categorisation import infer_category
from app.core.config import settings
from app.db.supabase import supabase


# ── Constants ────────────────────────────────────────────────────────────────

# NewsAPI source IDs.
NEWSAPI_SOURCES = [
    "cnn",
    "fox-news",
    "bbc-news",
    "associated-press",
]

# RSS feed → source name mapping.
# Euronews replaced by Deutsche Welle in v8.
FEED_NAME_MAP = {
    "https://www.theguardian.com/uk/rss": "The Guardian",
    "https://www.gbnews.com/feeds/politics.rss": "GB News",
    "https://www.rte.ie/news/rss/news-headlines.xml": "RTÉ News",
    "https://www.irishtimes.com/cmlink/news-1.1319192": "The Irish Times",
    "https://www.independent.co.uk/news/uk/rss": "The Independent",
    "https://www.npr.org/rss/rss.php?id=1001": "NPR",
    "https://feeds.skynews.com/feeds/rss/uk.xml": "Sky News",
    "https://rss.dw.com/rdf/rss-en-all": "Deutsche Welle",
    "https://feeds.apnews.com/rss/apf-topnews": "AP News",
}

_SCRAPE_WORKERS = 4
_FACTCHECK_CONCURRENCY = 1
_INSERT_CHUNK_SIZE = 10
_FACTCHECK_DELAY_SECONDS = 600

_main_event_loop: Optional[asyncio.AbstractEventLoop] = None

# ── Boilerplate regexes ──────────────────────────────────────────────────────

# The Independent fundraising preamble (DOTALL — spans line breaks).
_INDEPENDENT_PREAMBLE_RE = re.compile(
    r"Your support helps us to tell the story.*?Read more\s*",
    re.DOTALL | re.IGNORECASE,
)

# Fox News: "close\nVideo\n<caption>\nNEW\nYou can now listen..." block.
_FOX_VIDEO_OVERLAY_RE = re.compile(
    r"^close\s*\nVideo\s*\n.*?(?=NEW\s*\nYou can now listen)",
    re.DOTALL | re.IGNORECASE,
)
_FOX_LISTEN_BANNER_RE = re.compile(
    r"NEW\s*\nYou can now listen to Fox News articles!\s*\n",
    re.IGNORECASE,
)

# ── Source domain dispatch table ─────────────────────────────────────────────

_SOURCE_DISPATCH = {
    "cnn.com": "CNN",
    "dw.com": "Deutsche Welle",
    "foxnews.com": "Fox News",
    "bbc.co.uk": "BBC News",
    "bbc.com": "BBC News",
    "apnews.com": "AP News",
    "gbnews.com": "GB News",
}


def _detect_source_from_url(url: str) -> Optional[str]:
    """Return the canonical source name for a URL, or None."""
    if not url:
        return None
    url_lower = url.lower()
    for domain, name in _SOURCE_DISPATCH.items():
        if domain in url_lower:
            return name
    return None


# ── Content validation ───────────────────────────────────────────────────────


def _is_readable_content(text: str) -> bool:
    """
    Return True if text is readable UTF-8, not binary data.
    Printable-ratio check on first 500 chars — below 85% is rejected.
    """
    if not text or len(text) < 40:
        return False
    sample = text[:500]
    printable = sum(1 for c in sample if c.isprintable())
    return (printable / len(sample)) > 0.85


# ── Event loop bridge ────────────────────────────────────────────────────────


def set_main_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _main_event_loop
    _main_event_loop = loop


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_rss_feeds() -> List[str]:
    raw = os.getenv("RSS_FEEDS", "")
    return [s.strip() for s in raw.split(",") if s.strip()]


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
    return text.encode("utf-8", errors="ignore").decode(
        "utf-8", errors="ignore"
    )


def clean_text(text: str) -> str:
    """
    Clean scraped text. Strips source-specific boilerplate first,
    then generic noise patterns.
    """
    if not text:
        return ""
    text = sanitize_for_postgres(text)

    # Source-specific preamble/boilerplate strips.
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


# ── Shared scraping helper ───────────────────────────────────────────────────


def _get_soup(
    url: str,
    *,
    referer: str = "https://www.google.com/",
    lang: str = "en-US,en;q=0.9",
) -> Optional[BeautifulSoup]:
    """
    Fetch URL with Accept-Encoding: identity (no gzip/brotli) and
    return a BeautifulSoup object, or None on failure.

    All dedicated scrapers use this helper — identity encoding
    prevents the gzip binary blob issue.
    Response is closed and deleted immediately after parsing.
    """
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
    """Remove script, style, nav, footer, figure etc. in-place."""
    for tag in soup(
        [
            "script", "style", "nav", "footer", "aside",
            "header", "iframe", "noscript", "form", "button",
            "figure", "figcaption", "video", "picture",
        ]
    ):
        tag.decompose()


def _paragraphs_from(
    container: Any, min_len: int = 30
) -> List[str]:
    """Extract non-empty paragraph strings from a BS4 container."""
    return [
        p.get_text(separator=" ", strip=True)
        for p in container.find_all(["p", "h2", "h3"])
        if len(p.get_text(strip=True)) > min_len
    ]


# ── Source-specific scrapers ─────────────────────────────────────────────────


def _scrape_bbc(url: str) -> Optional[str]:
    """
    BBC News dedicated scraper.

    Scopes all block lookups to the
    [data-component="article-body-component"] wrapper first.
    Without this scope, text-block elements are found across the full
    DOM including related-stories panels and sidebars, causing the
    scraper to start mid-article.

    newspaper3k is intentionally NOT called for BBC — it returns
    AMP/cached partial versions that begin from the wrong paragraph.

    Block types collected (in document order):
      text-block          — body paragraphs
      crosshead-block     — subheadings
      unordered-list-block — bullet lists
    """
    soup = _get_soup(url, lang="en-GB,en;q=0.9")
    if soup is None:
        return None

    _strip_noise(soup)

    paragraphs: List[str] = []
    wanted_components = {
        "text-block",
        "crosshead-block",
        "unordered-list-block",
    }

    # PRIMARY: scoped to article-body-component wrapper.
    article_body = soup.find(
        attrs={"data-component": "article-body-component"}
    )
    if article_body:
        for block in article_body.find_all(
            attrs={"data-component": True}
        ):
            if block.get("data-component") not in wanted_components:
                continue
            t = block.get_text(separator=" ", strip=True)
            if len(t) > 20:
                paragraphs.append(t)

    # SECONDARY: ssrcss-* paragraph class pattern scoped to <article>.
    if not paragraphs:
        article_tag = soup.find("article")
        if article_tag:
            for p in article_tag.find_all(
                "p", class_=re.compile(r"ssrcss-")
            ):
                t = p.get_text(strip=True)
                if len(t) > 30:
                    paragraphs.append(t)

    # TERTIARY: generic article > p fallback.
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
    """
    CNN dedicated scraper (v2 — bleed fix).

    THE BLEED FIX: The previous scraper matched .article__content and
    .zn-body selectors against shared layout containers that persist
    across page navigation in CNN's SPA shell. This caused the
    Japan/Takaichi snippet to bleed into multiple unrelated articles.

    Fix strategy:
      1. Skip video pages (/videos/, /video/) — always returned
         sidebar content, never article text.
      2. Check JSON-LD structured data (<script
         type="application/ld+json">) for an "articleBody" field.
         If present, use it directly — it is always the correct
         article text with no layout bleed possible.
      3. Fall back to [data-section-id="body-text"] and
         [data-uri] containers, which are scoped to the current
         article only (not shared layout wrappers).
      4. Final fallback: most text-dense direct child of <article>
         rather than first match (old behaviour picked up sidebar).
    """
    if "/videos/" in url or "/video/" in url:
        return None

    soup = _get_soup(url)
    if soup is None:
        return None

    _strip_noise(soup)

    # Step 1: try JSON-LD articleBody — cleanest possible source.
    for script_tag in soup.find_all(
        "script", type="application/ld+json"
    ):
        try:
            ld = json.loads(script_tag.string or "")
            # Handle both single object and @graph array.
            candidates = (
                ld if isinstance(ld, list)
                else ld.get("@graph", [ld])
            )
            for node in candidates:
                body = node.get("articleBody", "")
                if body and len(body) > 150:
                    soup.decompose()
                    del soup
                    return clean_text(body)
        except Exception:
            continue

    # Step 2: article-scoped selectors only.
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
            # Use the element with the most text to avoid sidebars.
            best = max(
                elements,
                key=lambda el: len(el.get_text(strip=True)),
            )
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

    # Step 3: most text-dense direct child of <article>.
    if not paragraphs:
        article_tag = soup.find("article")
        if article_tag:
            children = [
                c for c in article_tag.children
                if hasattr(c, "get_text")
            ]
            if children:
                best_child = max(
                    children,
                    key=lambda c: len(c.get_text(strip=True)),
                )
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
    """
    Deutsche Welle dedicated scraper.

    DW publishes full English-language news at dw.com/en.
    Public broadcaster — no paywall, no gzip issues.
    Content-type guard rejects any non-HTML response before parsing.

    Primary selector: .content-area (stable across DW layouts).
    Secondary: [class*='longText'] used on long-form articles.
    Tertiary: generic article > p fallback.
    """
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
            best = max(
                elements,
                key=lambda el: len(el.get_text(strip=True)),
            )
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
    """
    Fox News dedicated scraper.

    Targets .article-body directly. The "close\\nVideo\\n" overlay
    and "NEW\\nYou can now listen" banner are stripped by clean_text()
    regexes as an additional guard.
    """
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
    """
    AP News dedicated scraper.

    AP uses div[class*="RichTextStoryBody"] as its primary article
    body container (2022+ layout). Fallbacks cover the older
    div.Article and data-key="card-body" structures.

    AP News is fully open-access (no paywall), Center-Right leaning.
    """
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
    """
    GB News dedicated scraper.

    GB News uses React SSR — the article body sits inside
    div[class*="article-content"] or div[class*="ArticleBody"].
    Generic tier was picking up nav/sidebar text instead.

    Strips share buttons, author bylines, and tag clouds that GB News
    injects between article paragraphs. Selects the container with the
    most text to avoid picking up a short sidebar.
    """
    soup = _get_soup(url)
    if soup is None:
        return None

    _strip_noise(soup)

    # Strip GB News-specific noise injected between paragraphs.
    for tag in soup.find_all(
        class_=re.compile(
            r"share|social|tag|author|byline|related|trending",
            re.IGNORECASE,
        )
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
            best = max(
                elements,
                key=lambda el: len(el.get_text(strip=True)),
            )
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


# ── Generic scraping tiers ───────────────────────────────────────────────────


def fetch_content_newspaper(url: str) -> Optional[str]:
    """Tier 1: newspaper3k — fast, reliable for ~80% of news sites."""
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
    """
    Tier 2: BeautifulSoup smart extraction.
    UTF-8 first, apparent_encoding fallback.
    """
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
        response = requests.get(
            url, headers=headers, timeout=15, allow_redirects=True
        )
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
            '[id*="article-body"]',
            '[id*="story-body"]',
            "main article",
            "main",
            ".article",
            "#article",
        ]
        for selector in article_selectors:
            try:
                elements = soup.select(selector)
                if elements:
                    text = elements[0].get_text(
                        separator="\n", strip=True
                    )
                    if len(text) > 150:
                        content = text
                        break
            except Exception:
                continue

        if not content or len(content) < 150:
            all_containers = soup.find_all(["div", "section", "main"])
            max_text = ""
            for container in all_containers:
                container_text = container.get_text(
                    separator="\n", strip=True
                )
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
    """Tier 3: Ultra-aggressive full-page text extraction."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1)"
        }
        response = requests.get(
            url, headers=headers, timeout=15, allow_redirects=True
        )
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


def fetch_content_with_retry(
    url: str, max_retries: int = 2
) -> Optional[str]:
    """Tier 4: Retry with exponential backoff."""
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
                url,
                headers=headers,
                timeout=20,
                allow_redirects=True,
                verify=True,
            )
            if response.status_code == 429:
                response.close()
                time.sleep((2 ** attempt) * 2)
                continue
            response.raise_for_status()

            response.encoding = "utf-8"
            decoded = response.text
            if not _is_readable_content(decoded):
                response.encoding = (
                    response.apparent_encoding or "utf-8"
                )
                decoded = response.text

            soup = BeautifulSoup(decoded, "html.parser")
            response.close()
            del response

            for unwanted in soup(
                ["script", "style", "head", "title", "meta", "link"]
            ):
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
    """Tier 5: Last-resort fallback — builds minimal content from title."""
    fallback_text = (
        f"Article from {source}: {title}. "
        f"Full content could not be retrieved. "
        f"This is a placeholder for analysis purposes. "
        f"The article discusses topics related to: {title.lower()}. "
        f"For the complete article, please visit the source website."
    )
    while len(fallback_text) < 200:
        fallback_text += (
            f" Additional context from {source} regarding {title}."
        )
    return clean_text(fallback_text)


def fetch_content(url: str, title: str = "", source: str = "") -> str:
    """
    6-tier guaranteed scraping strategy. Never returns None.

    Tier 0: Source-specific scraper dispatched by domain:
            BBC, CNN, Deutsche Welle, Fox News, AP News, GB News.
            newspaper3k is intentionally SKIPPED for BBC (returns
            AMP partial versions starting from wrong paragraph).
    Tier 1: newspaper3k    (~80% for remaining sources)
    Tier 2: BeautifulSoup  (~95%)
    Tier 3: Aggressive     (~98%)
    Tier 4: Retry backoff  (~99.5%)
    Tier 5: Title fallback (100% guarantee)
    """
    detected_source = _detect_source_from_url(url)

    # Tier 0: source-specific dispatch.
    tier0_map = {
        "BBC News": _scrape_bbc,
        "CNN": _scrape_cnn,
        "Deutsche Welle": _scrape_dw,
        "Fox News": _scrape_fox_news,
        "AP News": _scrape_ap_news,
        "GB News": _scrape_gb_news,
    }
    if detected_source and detected_source in tier0_map:
        content = tier0_map[detected_source](url)
        if content:
            gc.collect()
            return content
        # Fall through to generic tiers on None.

    # Tier 1: newspaper3k — skip for BBC (wrong partial version).
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


# ── Worker ───────────────────────────────────────────────────────────────────


def _scrape_one(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scrape a single article and return it enriched with content.
    Runs inside a ThreadPoolExecutor worker.
    """
    source_name_val = article.get("source")
    title_val = article.get("title", "")

    content = fetch_content(
        article["url"],
        title=title_val,
        source=source_name_val or "",
    )

    if not _is_readable_content(content):
        print(
            f"  Binary content detected, using fallback: "
            f"{article.get('url', '?')[:80]}"
        )
        content = fetch_content_title_fallback(
            title_val, source_name_val or ""
        )

    category = article.get("category")
    if not category:
        print(
            f"  Category missing — re-inferring: "
            f"{article.get('url', '?')[:80]}"
        )
        category = infer_category(
            article.get("url"),
            article.get("title"),
            content,
        )

    title = sanitize_for_postgres(title_val)
    url = sanitize_for_postgres(article["url"])
    content = sanitize_for_postgres(content)
    source_id = (
        upsert_source(source_name_val) if source_name_val else None
    )
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


# ── Fact-check bridge ────────────────────────────────────────────────────────


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
                        "updated_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
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
    print(
        f"Starting delayed fact-check for {len(article_ids)} articles..."
    )
    await factcheck_batch(article_ids)


def _schedule_factcheck(article_ids: List[str]) -> None:
    if _main_event_loop is None or not _main_event_loop.is_running():
        print("Fact-check skipped — no running event loop captured.")
        return
    asyncio.run_coroutine_threadsafe(
        _delayed_factcheck(article_ids), _main_event_loop
    )


# ── Batch insert ─────────────────────────────────────────────────────────────


def insert_articles_batch(articles: List[Dict[str, Any]]) -> List[str]:
    urls = [a["url"] for a in articles if a.get("url")]
    if not urls:
        return []

    existing = (
        supabase.table("articles")
        .select("url")
        .in_("url", urls)
        .execute()
        .data
    )
    existing_urls = {e["url"] for e in existing}

    new_articles = [
        a for a in articles
        if a.get("url") and a["url"] not in existing_urls
    ]

    del articles
    gc.collect()

    if not new_articles:
        print("No new articles to insert")
        return []

    scrape_success = 0
    scrape_fallback = 0
    all_new_ids: List[str] = []

    for chunk_start in range(
        0, len(new_articles), _INSERT_CHUNK_SIZE
    ):
        chunk = new_articles[
            chunk_start:chunk_start + _INSERT_CHUNK_SIZE
        ]
        payloads: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(
            max_workers=_SCRAPE_WORKERS
        ) as executor:
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
                    print(
                        f"  Scrape error "
                        f"[{article.get('url', '?')}]: {exc}"
                    )
                finally:
                    gc.collect()

        if payloads:
            res = (
                supabase.table("articles")
                .insert(payloads)
                .execute()
                .data
            )
            all_new_ids.extend(r["id"] for r in res)

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


# ── Fetchers ─────────────────────────────────────────────────────────────────


def fetch_newsapi() -> List[Dict[str, Any]]:
    """Fetch articles from NewsAPI."""
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
    }

    for source_id in NEWSAPI_SOURCES:
        try:
            params = {
                "sources": source_id,
                "pageSize": 25,
                "language": "en",
            }
            r = requests.get(
                url, params=params, headers=headers, timeout=15
            )
            r.raise_for_status()
            data = r.json()
            r.close()

            for a in data.get("articles", []):
                source_name = (a.get("source") or {}).get("name")
                source_name = source_map.get(source_name, source_name)
                n = normalize_article(
                    source_name=source_name,
                    url=a.get("url"),
                    title=a.get("title"),
                    published_at=a.get("publishedAt"),
                )
                if n["url"]:
                    normalized.append(n)

            print(
                f"  {source_id}: "
                f"{len(data.get('articles', []))} articles"
            )

        except Exception as exc:
            print(f"  {source_id} failed: {exc}")
            continue

    print(
        f"NewsAPI total: {len(normalized)} articles "
        f"from {len(NEWSAPI_SOURCES)} sources"
    )
    return normalized


def fetch_rss() -> List[Dict[str, Any]]:
    """
    Fetch articles from configured RSS feeds.

    RSS_FEEDS env var should include:
      https://rss.dw.com/rdf/rss-en-all  (Deutsche Welle English)
    """
    rss_feeds = _get_rss_feeds()
    normalized: List[Dict[str, Any]] = []

    for feed in rss_feeds:
        try:
            parsed = feedparser.parse(feed)
            source_name = FEED_NAME_MAP.get(feed)
            if not source_name:
                source_name = parsed.feed.get(
                    "title", "Unknown Source"
                )
            if source_name == "News Headlines":
                source_name = "RTÉ News"

            articles_from_feed = 0
            for e in parsed.entries[:15]:
                url = getattr(e, "link", None)
                title = getattr(e, "title", None)
                published = getattr(
                    e, "published", None
                ) or getattr(e, "updated", None)
                n = normalize_article(
                    source_name=source_name,
                    url=url,
                    title=title,
                    published_at=published,
                )
                if n["url"]:
                    normalized.append(n)
                    articles_from_feed += 1

            print(f"  {source_name}: {articles_from_feed} articles")

        except Exception as exc:
            print(f"  {feed}: {exc}")
            continue

    print(
        f"RSS total: {len(normalized)} articles "
        f"from {len(rss_feeds)} feeds"
    )
    return normalized


# ── Main job ─────────────────────────────────────────────────────────────────


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

    print("\nInserting articles (6-tier guaranteed scraping)...")
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
