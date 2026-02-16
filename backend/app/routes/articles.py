# app/routes/articles.py
from fastapi import APIRouter, Query
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.db.supabase import supabase
from backend.app.schemas import ArticleCreate


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


@router.post("")
def add_article(article: ArticleCreate):
    insert_response = (
        supabase.table("articles")
        .insert(
            {
                "source": article.source,
                "url": article.url,
                "bias_score": article.bias_score,
                "sentiment_score": article.sentiment_score,
                "published_at": article.published_at,
                "content": article.content,
                # optional: allow category to be set via API if you want
                # "category": article.category,
            }
        )
        .execute()
    )
    return insert_response.data
