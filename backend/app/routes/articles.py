from fastapi import APIRouter, Query
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.db.supabase import supabase
from app.schemas import ArticleCreate

router = APIRouter()


@router.get("")
def get_articles(category: Optional[str] = Query(default=None)):
    """
    Retrieve articles from the last 30 days, optionally filtered by category,
    sorted newest first.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=30)
    ).isoformat()

    query = (
        supabase.table("articles")
        .select("*")
        .gte("published_at", cutoff)
    )

    if category:
        query = query.eq("category", category)

    response = (
        query
        .order("published_at", desc=True)
        .limit(1000)
        .execute()
    )
    return response.data


@router.get("/compare")
def get_comparison_articles(
    topic: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
):
    """
    Get articles for comparison view, optionally filtered by topic AND category.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    query = (
        supabase.table("articles")
        .select("*")
        .gte("published_at", cutoff)
    )

    if topic:
        query = query.ilike("content", f"%{topic}%")
    if category:
        query = query.eq("category", category)

    response = (
        query
        .order("published_at", desc=True)
        .limit(30)
        .execute()
    )
    return response.data


@router.post("")
def add_article(article: ArticleCreate):
    insert_response = (
        supabase.table("articles")
        .insert({
            "source": article.source,
            "url": article.url,
            "bias_score": article.bias_score,
            "sentiment_score": article.sentiment_score,
            "published_at": article.published_at,
            "content": article.content,
            "category": getattr(article, "category", None),
            "general_bias": getattr(article, "general_bias", None),         # ← NEW
            "general_bias_score": getattr(article, "general_bias_score", None),  # ← NEW
        })
        .execute()
    )
    return insert_response.data
