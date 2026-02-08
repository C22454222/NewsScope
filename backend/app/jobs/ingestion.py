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

# NewsAPI sources (4 requests per cycle = 96/day)
NEWSAPI_SOURCES = [
    "cnn",              # Left, US
    "fox-news",         # Right, US
    "reuters",          # Centre, International/Europe
    "politico",         # Centre-Right, Europe
]

# Map RSS feed URLs to clean source names
FEED_NAME_MAP = {
    "http://feeds.bbci.co.uk/news/rss.xml": "BBC News",
    "https://www.theguardian.com/uk/rss": "The Guardian",
    "https://www.gbnews.com/feeds/politics.rss": "GB News",
    "https://www.rte.ie/news/rss/news-headlines.xml": "RTÉ News",
    "https://www.irishtimes.com/cmlink/news-1.1319192": "The Irish Times",
    "https://www.independent.co.uk/news/uk/rss": "The Independent",
    "https://www.npr.org/rss/rss.php?id=1001": "NPR",
    "https://feeds.skynews.com/feeds/rss/uk.xml": "Sky News",
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
        print(f"✅ Inserted {len(res)} new articles")
        return [r["id"] for r in res]

    print("ℹ️  No new articles to insert")
    return []


def fetch_newsapi():
    """
    Fetch articles from NewsAPI - one request per source.

    Strategy:
    - 4 sources × 1 request each = 4 requests/cycle
    - 24 cycles/day × 4 requests = 96 requests/day (96% of 100 limit)
    - 50 articles per source × 4 sources × 24 cycles = 4,800 articles/day
    """
    if not NEWSAPI_KEY:
        print("⚠️  NEWSAPI_KEY not set, skipping NewsAPI")
        return []

    url = "https://newsapi.org/v2/top-headlines"
    headers = {"X-Api-Key": NEWSAPI_KEY}

    normalized = []

    for source_id in NEWSAPI_SOURCES:
        try:
            params = {
                "sources": source_id,
                "pageSize": 50,
                "language": "en",
            }

            r = requests.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()

            for a in data.get("articles", []):
                source_name = (a.get("source") or {}).get("name")

                source_map = {
                    "Fox News": "Fox News",
                    "CNN": "CNN",
                    "Reuters": "Reuters",
                    "Politico": "Politico Europe",
                }
                source_name = source_map.get(source_name, source_name)

                n = normalize_article(
                    source_name=source_name,
                    url=a.get("url"),
                    title=a.get("title"),
                    published_at=a.get("publishedAt"),
                )
                if n["url"]:
                    normalized.append(n)

            print(f"  ✓ {source_id}: {len(data.get('articles', []))} articles")

        except Exception as exc:
            print(f"  ✗ {source_id} failed: {exc}")
            continue

    print(
        f"📰 NewsAPI total: {len(normalized)} articles "
        f"from {len(NEWSAPI_SOURCES)} sources"
    )
    return normalized


def fetch_rss():
    """
    Fetch articles from configured RSS feeds.

    Strategy:
    - 8 feeds × 12 articles each = 96 articles/cycle
    - 24 cycles/day = 2,304 articles/day
    """
    normalized = []

    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)

            source_name = FEED_NAME_MAP.get(feed)
            if not source_name:
                source_name = parsed.feed.get("title", "Unknown Source")

            if source_name == "News Headlines":
                source_name = "RTÉ News"

            articles_from_feed = 0
            for e in parsed.entries[:12]:
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
                    articles_from_feed += 1

            print(f"  ✓ {source_name}: {articles_from_feed} articles")

        except Exception as exc:
            print(f"  ✗ {feed}: {exc}")
            continue

    print(
        f"📡 RSS total: {len(normalized)} articles "
        f"from {len(RSS_FEEDS)} feeds"
    )
    return normalized


def run_ingestion_cycle():
    """
    Main ingestion job - runs every hour at :00.

    Daily volume:
    - NewsAPI: 4,800 articles (4 sources × 50 articles × 24 cycles)
    - RSS: 2,304 articles (8 feeds × 12 articles × 24 cycles)
    - Total fetched: 7,104/day
    - New after dedup (~60%): 4,262/day
    - 30-day DB: ~128,000 articles (~375 MB, 75% of 500MB limit)

    Source distribution (12 total):
    - US (3): CNN (Left), Fox News (Right), NPR (Centre-Left)
    - UK (3): BBC News (Centre), The Guardian (Left), GB News (Right)
    - Ireland (3): RTÉ News (Centre), Irish Times (Centre), Independent (Centre-Left)
    - Europe (3): Reuters (Centre), Politico Europe (Centre-Right), Sky News (Centre-Right)

    Bias distribution:
    - Left (2): CNN, The Guardian
    - Centre-Left (2): NPR, The Independent
    - Centre (4): BBC, RTÉ, Reuters, Irish Times
    - Centre-Right (2): Politico Europe, Sky News
    - Right (2): Fox News, GB News

    API usage:
    - NewsAPI: 96/100 requests per day (96%)
    - RSS: unlimited
    """
    print("\n" + "=" * 70)
    print("🔄 INGESTION CYCLE STARTED")
    print("=" * 70)

    articles = []

    print("\n📰 Fetching from NewsAPI...")
    try:
        articles += fetch_newsapi()
    except Exception as exc:
        print(f"❌ NewsAPI critical error: {exc}")

    print("\n📡 Fetching from RSS feeds...")
    try:
        articles += fetch_rss()
    except Exception as exc:
        print(f"❌ RSS critical error: {exc}")

    print("\n💾 Inserting articles...")
    try:
        print(f"📊 Total fetched: {len(articles)} articles")
        insert_articles_batch(articles)
    except Exception as exc:
        print(f"❌ Batch insert failed: {exc}")

    print("\n✅ Ingestion cycle complete")
    print("=" * 70 + "\n")
