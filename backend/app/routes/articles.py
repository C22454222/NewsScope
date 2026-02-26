# app/routes/articles.py
"""
NewsScope Articles API Router.
Week 4: Credibility scoring + fact-checking endpoints.
Flake8: 0 errors.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.db.supabase import supabase
from app.jobs.fact_checking import batch_factcheck_recent, compute_credibility_score
from app.schemas import ArticleCreate, ArticleResponse

router = APIRouter(prefix="/articles", tags=["articles"])


@router.get("")
def get_articles(
    category: Optional[str] = Query(default=None),
) -> List[dict]:
    """Recent articles (30d), optionally filtered by category."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    query = supabase.table("articles").select("*").gte("published_at", cutoff)

    if category:
        query = query.eq("category", category)

    return query.order("published_at", desc=True).limit(1000).execute().data


@router.get("/compare")
def get_comparison_articles(
    topic: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
) -> List[dict]:
    """Articles for comparison view, filtered by topic and/or category."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    query = supabase.table("articles").select("*").gte("published_at", cutoff)

    if topic:
        query = query.ilike("content", f"%{topic}%")
    if category:
        query = query.eq("category", category)

    return query.order("published_at", desc=True).limit(30).execute().data


@router.post("")
async def add_article(article: ArticleCreate) -> dict:
    """Add article with automatic credibility enrichment."""
    insert_data = {
        "source": article.source,
        "url": article.url,
        "title": getattr(article, "title", None),
        "description": getattr(article, "description", None),
        "bias_score": getattr(article, "bias_score", None),
        "sentiment_score": getattr(article, "sentiment_score", None),
        "general_bias": getattr(article, "general_bias", None),
        "general_bias_score": getattr(article, "general_bias_score", None),
        "published_at": article.published_at,
        "content": article.content,
        "category": getattr(article, "category", None),
        "credibility_score": 80.0,
        "fact_checks": {},
        "claims_checked": 0,
        "credibility_reason": "Pending",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    cred = await compute_credibility_score(ArticleResponse(**insert_data))
    insert_data.update(
        {
            "credibility_score": cred["score"],
            "fact_checks": cred["fact_checks"],
            "claims_checked": cred["claims_checked"],
            "credibility_reason": cred["reason"],
        }
    )

    resp = supabase.table("articles").insert(insert_data).execute()
    if not resp.data:
        raise HTTPException(status_code=500, detail="Insert failed")
    return resp.data[0]


@router.post("/{article_id}/factcheck")
async def factcheck_article(article_id: str) -> dict:
    """Manually re-run fact-check for a single article."""
    resp = (
        supabase.table("articles").select("*").eq("id", article_id).execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Article not found")

    cred = await compute_credibility_score(ArticleResponse(**resp.data[0]))

    supabase.table("articles").update(
        {
            "credibility_score": cred["score"],
            "fact_checks": cred["fact_checks"],
            "claims_checked": cred["claims_checked"],
            "credibility_reason": cred["reason"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ).eq("id", article_id).execute()

    return cred


@router.get("/recent-factchecks")
async def recent_factchecks(hours: int = 24) -> List[dict]:
    """Return recently fact-checked articles for Flutter dashboard."""
    return await batch_factcheck_recent(hours)
