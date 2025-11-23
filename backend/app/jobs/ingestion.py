import os
import requests
import feedparser
from dateutil import parser as dtparser
from app.db.supabase import supabase

# For parsing article text
from newspaper import Article

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# RSS feeds configured via .env (BBC, GB News, RTÉ)
RSS_FEEDS = [s.strip() for s in os.getenv("RSS_FEEDS", "").split(",") if s.strip()]


def normalize_article(
    *, source_name: str, url: str, published_at,
    bias_score=None, sentiment_score=None
):
    ts = None
    if published_at:
        try:
            ts = dtparser.parse(published_at)
        except Exception:
            ts = None
    return {
        "url": url,
        "published_at": ts.isoformat() if ts else None,
        "bias_score": bias_score,
        "sentiment_score": sentiment_score,
        "source_name": source_name,
    }


def upsert_source(name: str):
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
    """Download and parse article text from URL."""
    try:
        art = Article(url)
        art.download()
        art.parse()
        return art.text
    except Exception:
        return None


def insert_articles_batch(articles: list[dict]):
    """Batch insert new articles, skipping duplicates by URL."""
    urls = [a["url"] for a in articles if a.get("url")]
    if not urls:
        return []

    # Check existing URLs in one query
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
        source_id = (
            upsert_source(article["source_name"])
            if article.get("source_name")
            else None
        )
        content = fetch_content(article["url"])
        payloads.append(
            {
                "url": article["url"],
                "published_at": article["published_at"],
                "bias_score": article.get("bias_score"),
                "sentiment_score": article.get("sentiment_score"),
                "source_id": source_id,
                "content": content,
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
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/top-headlines"
    # Only CNN here – RTÉ is not supported by NewsAPI
    params = {
        "language": "en",
        "pageSize": 5,  # limit to 5 articles for prototype
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
            published_at=a.get("publishedAt"),
        )
        if n["url"]:
            normalized.append(n)
    print(f"Fetched {len(normalized)} articles from NewsAPI (CNN)")
    return normalized


def fetch_rss():
    normalized = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            # limit to first 5 entries per RSS feed
            for e in parsed.entries[:5]:
                url = getattr(e, "link", None)
                published = getattr(e, "published", None) or getattr(
                    e, "updated", None
                )
                source_name = parsed.feed.get("title")
                n = normalize_article(
                    source_name=source_name,
                    url=url,
                    published_at=published,
                )
                if n["url"]:
                    normalized.append(n)
        except Exception as exc:
            print(f"RSS fetch error for {feed}: {exc}")
            continue
    print(f"Fetched {len(normalized)} articles from RSS (BBC/GB/RTÉ)")
    return normalized


def run_ingestion_cycle():
    articles: list[dict] = []
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
