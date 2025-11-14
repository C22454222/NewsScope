# app/jobs/ingestion.py
import os
import requests
import feedparser
from dateutil import parser as dtparser
from app.db.supabase import supabase


NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
RSS_FEEDS = [s.strip() for s in os.getenv("RSS_FEEDS", "").split(",") if s.strip()]


def normalize_article(*, source_name: str, url: str, published_at, bias_score=None, sentiment_score=None):
    # published_at can be None or string → convert to ISO
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
        # we store normalized source in sources table; attach later as source_id
        # keep source_name for lookup/insert
        "source_name": source_name,
    }


def upsert_source(name: str):
    # get or create source row, return id
    existing = supabase.table("sources").select("id").eq("name", name).limit(1).execute().data
    if existing:
        return existing[0]["id"]
    inserted = supabase.table("sources").insert({"name": name}).execute().data
    return inserted[0]["id"]


def insert_article_if_new(article: dict):
    # dedupe by URL
    exists = supabase.table("articles").select("id").eq("url", article["url"]).limit(1).execute().data
    if exists:
        return exists[0]["id"]
    source_id = upsert_source(article["source_name"]) if article.get("source_name") else None
    payload = {
        "url": article["url"],
        "published_at": article["published_at"],
        "bias_score": article.get("bias_score"),
        "sentiment_score": article.get("sentiment_score"),
        "source_id": source_id
    }
    res = supabase.table("articles").insert(payload).execute().data
    return res[0]["id"]


def fetch_newsapi():
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/top-headlines"
    params = {"language": "en", "pageSize": 50}
    headers = {"Authorization": NEWSAPI_KEY}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    normalized = []
    for a in data.get("articles", []):
        n = normalize_article(
            source_name=(a.get("source") or {}).get("name"),
            url=a.get("url"),
            published_at=a.get("publishedAt"),
        )
        if n["url"]:
            normalized.append(n)
    return normalized


def fetch_rss():
    normalized = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for e in parsed.entries:
                url = getattr(e, "link", None)
                published = getattr(e, "published", None) or getattr(e, "updated", None)
                source_name = parsed.feed.get("title")
                n = normalize_article(
                    source_name=source_name,
                    url=url,
                    published_at=published,
                )
                if n["url"]:
                    normalized.append(n)
        except Exception:
            continue
    return normalized


def run_ingestion_cycle():
    articles = []
    try:
        articles += fetch_newsapi()
    except Exception:
        pass
    try:
        articles += fetch_rss()
    except Exception:
        pass
    for a in articles:
        try:
            insert_article_if_new(a)
        except Exception:
            continue
