# app/jobs/ingestion.py
import os
import requests
import feedparser
from dateutil import parser as dtparser
from app.db.supabase import supabase
from newspaper import Article


NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
RSS_FEEDS = [
    s.strip()
    for s in os.getenv("RSS_FEEDS", "").split(",")
    if s.strip()
]

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
    """Normalize article data from various sources into common format."""
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
    """Get or create source by name, return source ID."""
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
    return inserted[0]["id"]


def fetch_content(url: str) -> str | None:
    """Scrape full article content using newspaper3k."""
    try:
        art = Article(url)
        art.download()
        art.parse()
        return art.text
    except Exception:
        return None


def insert_articles_batch(articles: list[dict]):
    """Insert new articles, skip duplicates based on URL."""
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

    payloads = []
    for article in articles:
        if not article.get("url") or article["url"] in existing_urls:
            continue

        source_name_val = article.get("source")
        source_id = (
            upsert_source(source_name_val)
            if source_name_val
            else None
        )
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
    """Fetch top headlines from NewsAPI (CNN source)."""
    if not NEWSAPI_KEY:
        return []

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "language": "en",
        "pageSize": 20,
        "sources": "cnn",
    }
    headers = {"X-Api-Key": NEWSAPI_KEY}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()

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
    """Fetch articles from configured RSS feeds."""
    normalized = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)

            source_name = FEED_NAME_MAP.get(feed)
            if not source_name:
                source_name = parsed.feed.get("title", "Unknown Source")
            if source_name == "News Headlines":
                source_name = "RTÉ News"

            for e in parsed.entries[:10]:
                url = getattr(e, "link", None)
                title = getattr(e, "title", None)
                published = (
                    getattr(e, "published", None) or getattr(e, "updated", None)
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
            print(f"RSS fetch error for {feed}: {exc}")
            continue

    print(f"📡 Fetched {len(normalized)} articles from RSS")
    return normalized


def run_ingestion_cycle():
    """
    Fetch from NewsAPI (20) + RSS feeds (4 sources × 10 = 40).
    Total: ~60 articles per run, filtered down to ~20 new ones.
    Runs every hour at :00.
    """
    print("🔄 Starting ingestion cycle...")
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
