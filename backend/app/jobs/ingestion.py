# app/jobs/ingestion.py
import os
import requests
import feedparser
from dateutil import parser as dtparser
from app.db.supabase import supabase
from newspaper import Article

# NewsAPI key for fetching CNN headlines
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Comma-separated list of RSS feed URLs configured via environment
RSS_FEEDS = [s.strip() for s in os.getenv("RSS_FEEDS", "").split(",") if s.strip()]

# Optional mapping from raw feed URL to a clean, user-facing source name
FEED_NAME_MAP = {
    "http://feeds.bbci.co.uk/news/rss.xml": "BBC News",
    "https://www.rte.ie/news/rss/news-headlines.xml": "RTÉ News",
    "https://www.gbnews.com/feeds/news.rss": "GB News",
}


def normalize_article(
    *,
    source_name: str,
    url: str,
    title: str,
    published_at,
    bias_score=None,
    sentiment_score=None,
):
    """
    Convert provider-specific article fields into a consistent schema.

    Args:
        source_name: Human-readable name of the news outlet.
        url: Canonical article URL.
        title: Article title.
        published_at: Raw published date string from provider.
        bias_score: Optional bias score from analysis pipeline.
        sentiment_score: Optional sentiment score from analysis pipeline.

    Returns:
        dict: Normalized article ready for database insertion.
    """
    ts = None
    if published_at:
        try:
            ts = dtparser.parse(published_at)
        except Exception:
            ts = None

    return {
        "title": title,
        "url": url,
        "published_at": ts.isoformat() if ts else None,
        "bias_score": bias_score,
        "sentiment_score": sentiment_score,
        "source": source_name,
    }


def upsert_source(name: str):
    """
    Ensure a source record exists for the given name and return its ID.

    Performs a read-before-write to avoid duplicate source records.
    """
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

    inserted = supabase.table("sources").insert({"name": name}).execute().data
    return inserted[0]["id"]


def fetch_content(url: str) -> str | None:
    """
    Download and extract the main text content of an article.

    Returns:
        str | None: Cleaned article body text or None if parsing fails.
    """
    try:
        art = Article(url)
        art.download()
        art.parse()
        return art.text
    except Exception:
        return None


def insert_articles_batch(articles: list[dict]):
    """
    Insert a batch of normalized articles, skipping duplicates by URL.

    Returns:
        list[str]: IDs of newly inserted article records.
    """
    urls = [a["url"] for a in articles if a.get("url")]
    if not urls:
        return []

    # Fetch existing article URLs to avoid inserting duplicates
    existing = (
        supabase.table("articles")
        .select("url")
        .in_("url", urls)
        .execute()
        .data
    )
    existing_urls = {e["url"] for e in existing}

    payloads = []
    for article in articles:
        # Skip missing URLs and URLs we already have
        if not article.get("url") or article["url"] in existing_urls:
            continue

        source_name_val = article.get("source")
        source_id = upsert_source(source_name_val) if source_name_val else None
        content = fetch_content(article["url"])

        payloads.append(
            {
                "title": article.get("title"),
                "url": article["url"],
                "published_at": article["published_at"],
                "bias_score": article.get("bias_score"),
                "sentiment_score": article.get("sentiment_score"),
                "source_id": source_id,
                "content": content,
                "source": source_name_val,
            }
        )

    if payloads:
        res = supabase.table("articles").insert(payloads).execute().data
        print(f"Inserted {len(res)} new articles")
        return [r["id"] for r in res]
    else:
        print("No new articles to insert")
    return []


def fetch_newsapi():
    """
    Fetch top headlines from CNN via NewsAPI and normalize them.

    Returns:
        list[dict]: List of normalized article dictionaries.
    """
    if not NEWSAPI_KEY:
        return []

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "language": "en",
        "pageSize": 5,
        "sources": "cnn",
    }
    headers = {"X-Api-Key": NEWSAPI_KEY}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()

    print("NewsAPI raw response:", data)

    normalized = []
    for a in data.get("articles", []):
        n = normalize_article(
            source_name=(a.get("source") or {}).get("name"),
            url=a.get("url"),
            title=a.get("title"),
            published_at=a.get("publishedAt"),
        )
        if n["url"]:
            normalized.append(n)

    print(f"Fetched {len(normalized)} articles from NewsAPI (CNN)")
    return normalized


def fetch_rss():
    """
    Fetch and normalize articles from configured RSS feeds.

    Returns:
        list[dict]: List of normalized article dictionaries.
    """
    normalized = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)

            # Prefer a friendly, hard-coded name where available
            source_name = FEED_NAME_MAP.get(feed)
            if not source_name:
                source_name = parsed.feed.get("title", "Unknown Source")
            # Fix generic RTÉ feed title to something user-facing
            if source_name == "News Headlines":
                source_name = "RTÉ News"

            # Limit to a small number of entries per feed to control load
            for e in parsed.entries[:5]:
                url = getattr(e, "link", None)
                title = getattr(e, "title", None)
                published = getattr(e, "published", None) or getattr(
                    e,
                    "updated",
                    None,
                )
                n = normalize_article(
                    source_name=source_name,
                    url=url,
                    title=title,
                    published_at=published,
                )
                if n["url"]:
                    normalized.append(n)
        except Exception as exc:
            # Log and continue with the remaining feeds
            print(f"RSS fetch error for {feed}: {exc}")
            continue

    print(f"Fetched {len(normalized)} articles from RSS")
    return normalized


def run_ingestion_cycle():
    """
    End-to-end ingestion cycle for all providers.

    1. Fetch articles from NewsAPI.
    2. Fetch articles from RSS feeds.
    3. Insert only new articles into the articles table.

    This function is designed to be scheduled periodically by APScheduler.
    """
    articles = []

    try:
        articles += fetch_newsapi()
    except Exception as exc:
        print(f"NewsAPI fetch failed: {exc}")

    try:
        articles += fetch_rss()
    except Exception as exc:
        print(f"RSS fetch failed: {exc}")

    try:
        insert_articles_batch(articles)
    except Exception as exc:
        print(f"Batch insert failed: {exc}")
