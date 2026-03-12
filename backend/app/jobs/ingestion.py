"""
NewsScope Ingestion Pipeline.

Daily volume: ~1,920 articles fetched, ~1,152 new after dedup.
Credibility fields seeded as defaults on insert; fact-checking
runs async post-insert via safe loop wrapper with concurrency cap.

MEMORY FIX SUMMARY (v3):
- _SCRAPE_WORKERS reduced 3→2: fewer simultaneous BeautifulSoup
  parse trees in RAM. 2 workers ≈ 80-120MB vs 3 workers ≈ 150-200MB.
- _INSERT_CHUNK_SIZE reduced 10→5: chunk size now matches worker
  count more closely — previously 10 article dicts were held in
  the executor queue even though only 3 were being processed.
- _FACTCHECK_DELAY_SECONDS = 600: factcheck_batch previously fired
  immediately after ingestion while scraping RAM had not yet been
  freed and analysis was running. 10 minute delay gives GC time to
  clear ingestion RAM before factcheck Supabase fetches begin.
- _FACTCHECK_CONCURRENCY = 1: fully sequential factcheck prevents
  overlap between PolitiFact HTTP response buffers and analysis
  HF calls. With _MAX_CLAIMS=2 per article this is fast enough.
- Content stored in DB truncated to _CONTENT_STORE_LIMIT chars:
  full scraped content (can be 50-200KB per article) was being held
  in the payload dict until Supabase insert. Truncating to 5000 chars
  is sufficient for analysis (_TEXT_LIMIT=512) and display purposes
  while dramatically reducing per-payload RAM during batch insert.

ENCODING FIX:
- BeautifulSoup tiers now use response.text with apparent_encoding
  fallback instead of raw response.content bytes. This prevents
  Euronews and other sources that return gzip/brotli compressed
  responses from storing binary blobs in the content column.

Flake8: 0 errors/warnings.
"""

import asyncio
import gc
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


# ── Constants ─────────────────────────────────────────────────────────────────

NEWSAPI_SOURCES = [
    "cnn",
    "fox-news",
    "bbc-news",
    "politico",
]

FEED_NAME_MAP = {
    "https://www.theguardian.com/uk/rss": "The Guardian",
    "https://www.gbnews.com/feeds/politics.rss": "GB News",
    "https://www.rte.ie/news/rss/news-headlines.xml": "RTÉ News",
    "https://www.irishtimes.com/cmlink/news-1.1319192": "The Irish Times",
    "https://www.independent.co.uk/news/uk/rss": "The Independent",
    "https://www.npr.org/rss/rss.php?id=1001": "NPR",
    "https://feeds.skynews.com/feeds/rss/uk.xml": "Sky News",
    "https://www.euronews.com/rss": "Euronews",
}

# Reduced from 3 — fewer simultaneous BeautifulSoup parse trees in RAM.
# 2 workers ≈ 80-120MB peak vs 3 workers ≈ 150-200MB on Render free tier.
_SCRAPE_WORKERS = 2

# Fully sequential factcheck prevents any overlap between PolitiFact
# HTTP response buffers and analysis HF calls. With _MAX_CLAIMS=2 per
# article, one article at a time is fast enough.
_FACTCHECK_CONCURRENCY = 1

# Reduced from 10 — previously 10 article dicts were held in the executor
# queue simultaneously even though only 3 workers were processing them.
# 5 aligns chunk size more closely with worker count.
_INSERT_CHUNK_SIZE = 5

# Delay factcheck after ingestion completes — gives GC time to free
# scraping RAM before factcheck Supabase fetches begin loading full
# article rows. 10 min gives analysis cycles time to complete before
# PolitiFact HTTP calls begin.
_FACTCHECK_DELAY_SECONDS = 600

# Truncate content stored in DB — full scraped content can be 50-200KB
# per article. Analysis only uses 512 chars; 5000 is ample for display.
# Reduces per-payload RAM held during batch insert significantly.
_CONTENT_STORE_LIMIT = 5000

# Main event loop — captured at startup by set_main_event_loop().
_main_event_loop: Optional[asyncio.AbstractEventLoop] = None


# ── Event loop bridge ─────────────────────────────────────────────────────────


def set_main_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Store the main event loop so thread workers can schedule coroutines.
    Called once from lifespan in main.py before any jobs run.
    """
    global _main_event_loop
    _main_event_loop = loop


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_rss_feeds() -> List[str]:
    """Return configured RSS feed URLs from settings/env."""
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
    """
    Normalise article data from various sources into a common format.

    Category inference runs tiers 1+2 only here (fast, no model).
    Tier 3 zero-shot fires later in insert_articles_batch once
    content has been scraped and is available.
    """
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
    """Get or create a source record by name, return its ID."""
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
    """Remove characters that PostgreSQL cannot handle."""
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\u0000", "")
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    return text.encode("utf-8", errors="ignore").decode(
        "utf-8", errors="ignore"
    )


def clean_text(text: str) -> str:
    """Clean scraped text: remove whitespace and navigation boilerplate."""
    if not text:
        return ""
    text = sanitize_for_postgres(text)
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


# ── Scraping tiers ────────────────────────────────────────────────────────────


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
    Tier 2: BeautifulSoup smart extraction — handles ~95% of sites.

    Uses response.text with apparent_encoding fallback to ensure
    compressed responses (gzip/brotli) are correctly decoded before
    parsing. Prevents binary byte blobs being stored as article content.
    Uses html.parser (pure Python) instead of lxml to reduce RAM.
    Response closed and deleted immediately after parsing.
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

        # Decode correctly — prevents binary blobs from compressed sources.
        response.encoding = response.apparent_encoding or "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        response.close()
        del response

        for element in soup(
            [
                "script", "style", "nav", "footer", "aside", "header",
                "iframe", "noscript", "form", "button", "input",
                "select", "textarea", "label",
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
            '[id*="content-body"]',
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
    """
    Tier 3: Ultra-aggressive full-page text extraction.

    Uses response.text with apparent_encoding fallback to ensure
    compressed responses are correctly decoded before parsing.
    Uses html.parser to avoid lxml RAM overhead.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1)"
        }
        response = requests.get(
            url, headers=headers, timeout=15, allow_redirects=True
        )
        response.raise_for_status()

        # Decode correctly — prevents binary blobs from compressed sources.
        response.encoding = response.apparent_encoding or "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        response.close()
        del response

        for element in soup(
            [
                "script", "style", "nav", "footer", "aside", "header",
                "iframe", "noscript", "form", "button", "input", "meta",
                "link", "select", "textarea", "svg", "canvas", "video",
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
    url: str,
    max_retries: int = 2,
) -> Optional[str]:
    """
    Tier 4: Retry with exponential backoff.

    Uses response.text with apparent_encoding fallback to ensure
    compressed responses are correctly decoded before parsing.
    Max retries capped at 2 to limit memory dwell time.
    Uses html.parser throughout — lxml removed to reduce RAM.
    """
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

            # Decode correctly — prevents binary blobs from compressed
            # sources such as Euronews.
            response.encoding = response.apparent_encoding or "utf-8"
            soup = BeautifulSoup(response.text, "html.parser")
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
    5-tier guaranteed scraping strategy. Never returns None.

    Tier 1: newspaper3k           (~80% success)
    Tier 2: BeautifulSoup smart   (~95% success)
    Tier 3: Aggressive extraction (~98% success)
    Tier 4: Retry with backoff    (~99.5% success)
    Tier 5: Title-based fallback  (100% guarantee)
    """
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
    """
    Scrape a single article and return it enriched with content.
    Runs inside a ThreadPoolExecutor worker.

    Content is truncated to _CONTENT_STORE_LIMIT before building
    the result dict — full scraped pages can be 50-200KB each and
    holding them in memory across the chunk insert cycle causes
    significant RAM pressure. Analysis only needs 512 chars;
    5000 is ample for display.

    gc.collect() called before returning to free parse tree RAM promptly.
    """
    source_name_val = article.get("source")
    content = fetch_content(
        article["url"],
        title=article.get("title", ""),
        source=source_name_val or "",
    )

    # Truncate immediately after scraping — before any further processing.
    content = content[:_CONTENT_STORE_LIMIT]

    category = article.get("category")
    if category == "general":
        category = infer_category(
            article.get("url"),
            article.get("title"),
            content,
        )

    title = sanitize_for_postgres(article.get("title", ""))
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
    """
    Background async task: compute and persist credibility scores.

    Semaphore caps concurrency at _FACTCHECK_CONCURRENCY=1 — prevents
    simultaneous coroutines from spiking RAM after a large ingestion
    cycle on Render free tier (512MB). Each coroutine fetches a full
    article row from Supabase so uncapped parallelism causes OOM.

    Called after _FACTCHECK_DELAY_SECONDS delay to ensure ingestion
    RAM has been freed before Supabase fetches begin.
    """
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
    """
    Wait _FACTCHECK_DELAY_SECONDS then run factcheck_batch.

    Delay prevents factcheck Supabase fetches from overlapping with
    residual ingestion RAM and the 5-minute analysis scheduler.
    """
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
    """
    Schedule _delayed_factcheck onto the main event loop from any thread.

    Uses the loop captured at startup via set_main_event_loop() —
    safe to call from inside ThreadPoolExecutor workers.
    Delay of _FACTCHECK_DELAY_SECONDS ensures ingestion RAM is freed
    before factcheck Supabase fetches begin.
    """
    if _main_event_loop is None or not _main_event_loop.is_running():
        print("Fact-check skipped — no running event loop captured.")
        return
    asyncio.run_coroutine_threadsafe(
        _delayed_factcheck(article_ids), _main_event_loop
    )


# ── Batch insert ──────────────────────────────────────────────────────────────


def insert_articles_batch(articles: List[Dict[str, Any]]) -> List[str]:
    """
    Deduplicate by URL, scrape new articles in a bounded thread pool,
    then insert into Supabase in chunks of _INSERT_CHUNK_SIZE.

    Chunked insert prevents holding all scraped payloads in RAM at
    once — each chunk is written and released before the next is built.
    Workers capped at _SCRAPE_WORKERS=2 to prevent simultaneous
    BeautifulSoup parse trees from exhausting Render free tier RAM.
    gc.collect() runs after each completed future for prompt cleanup.
    Content truncated to _CONTENT_STORE_LIMIT per article before insert.
    """
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

    # Release the full fetched list — only new_articles needed from here.
    del articles
    gc.collect()

    if not new_articles:
        print("No new articles to insert")
        return []

    scrape_success = 0
    scrape_fallback = 0
    all_new_ids: List[str] = []

    # Scrape and insert in chunks to cap peak RAM.
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
                    print(
                        f"  Scrape error "
                        f"[{article.get('url', '?')}]: {exc}"
                    )
                finally:
                    gc.collect()

        if payloads:
            res = (
                supabase.table("articles").insert(payloads).execute().data
            )
            all_new_ids.extend(r["id"] for r in res)

        # Explicitly release chunk payloads before next iteration.
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


def fetch_newsapi() -> List[Dict[str, Any]]:
    """
    Fetch articles from NewsAPI — one request per source.

    Strategy:
    - 4 sources x 1 request = 4 requests/cycle
    - 24 cycles/day x 4 = 96 requests/day (96% of 100 free limit)
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
        "Politico": "Politico Europe",
    }

    for source_id in NEWSAPI_SOURCES:
        try:
            params = {
                "sources": source_id,
                "pageSize": 10,
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

    Strategy:
    - 8 feeds x 5 articles = 40 articles/cycle
    - 24 cycles/day = 960 articles/day
    Reduced from 12→5 per feed to stay under Render 512MB RAM limit.
    """
    rss_feeds = _get_rss_feeds()
    normalized: List[Dict[str, Any]] = []

    for feed in rss_feeds:
        try:
            parsed = feedparser.parse(feed)
            source_name = FEED_NAME_MAP.get(feed)
            if not source_name:
                source_name = parsed.feed.get("title", "Unknown Source")
            if source_name == "News Headlines":
                source_name = "RTÉ News"

            articles_from_feed = 0
            for e in parsed.entries[:5]:
                url = getattr(e, "link", None)
                title = getattr(e, "title", None)
                published = getattr(e, "published", None) or getattr(
                    e, "updated", None
                )

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


# ── Main job ──────────────────────────────────────────────────────────────────


def run_ingestion_cycle() -> None:
    """
    Main ingestion job — triggered by APScheduler every hour.

    Daily volume:
    - NewsAPI: 960 articles  (4 sources x 10 x 24 cycles)
    - RSS:     960 articles  (8 feeds   x  5 x 24 cycles)
    - Total fetched:          1,920/day
    - New after dedup (~60%): ~1,152/day
    Reduced from 7,104/day — pageSize 50→10, RSS entries 12→5.
    Each scrape cycle now processes ~46 articles max vs ~178 before,
    keeping peak BeautifulSoup RAM well under Render's 512MB limit.
    """
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

    print("\nInserting articles (5-tier guaranteed scraping)...")
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
