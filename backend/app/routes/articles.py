"""
NewsScope Articles API Router.

Credibility scoring + fact-checking endpoints.
Category filtering uses CATEGORY_GROUP_MAP to resolve sub-categories
(football, climate, film, etc.) to their parent group so Flutter's
8-chip filter correctly matches all stored variants.

Flake8: 0 errors/warnings.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.db.supabase import supabase
from app.jobs.fact_checking import (
    batch_factcheck_recent,
    compute_credibility_score,
)
from app.schemas import ArticleCreate, ArticleResponse


router = APIRouter(tags=["articles"])


# Maps granular backend sub-categories → Flutter parent chip categories.
# Ensures ?category=sport returns articles tagged football/rugby/gaa etc.
# Mirrors the 8 chips in HomeFeedTab._categories (lowercased).
CATEGORY_GROUP_MAP: dict = {
    # sport
    "football": "sport",
    "rugby": "sport",
    "gaa": "sport",
    "cricket": "sport",
    # science
    "environment": "science",
    "climate": "science",
    # entertainment
    "culture": "entertainment",
    "film": "entertainment",
    "tv": "entertainment",
    "music": "entertainment",
    # world
    "us": "world",
    "uk": "world",
    "ireland": "world",
    "europe": "world",
    # business
    "economy": "business",
    "markets": "business",
    "finance": "business",
}


@router.get("")
def get_articles(
    category: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
) -> List[dict]:
    """
    Recent articles (30d), optionally filtered by category and/or source.

    Category resolves sub-categories via CATEGORY_GROUP_MAP so
    ?category=sport returns articles tagged sport, football, rugby etc.
    Source matches the exact source name stored in the articles table,
    e.g. ?source=BBC+News or ?source=RTÉ+News.
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
        related = [
            k for k, v in CATEGORY_GROUP_MAP.items() if v == category
        ] + [category]
        query = query.in_("category", related)

    if source:
        query = query.eq("source", source)

    return (
        query.order("published_at", desc=True).limit(1000).execute().data
    )


@router.get("/compare")
def get_comparison_articles(
    topic: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
) -> List[dict]:
    """
    Articles for comparison view, filtered by topic, category, and/or source.

    Topic searches both title and content so articles with empty content
    fields are still matched by title.
    Category resolves sub-categories via CATEGORY_GROUP_MAP.
    Source matches the exact source name stored in the articles table.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=30)
    ).isoformat()

    query = (
        supabase.table("articles")
        .select("*")
        .gte("published_at", cutoff)
    )

    if topic:
        # Search title and content — catches articles with empty content
        query = query.or_(
            f"title.ilike.%{topic}%,content.ilike.%{topic}%"
        )

    if category:
        related = [
            k for k, v in CATEGORY_GROUP_MAP.items() if v == category
        ] + [category]
        query = query.in_("category", related)

    if source:
        query = query.eq("source", source)

    return (
        query.order("published_at", desc=True).limit(30).execute().data
    )


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
        "general_bias_score": getattr(
            article, "general_bias_score", None
        ),
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


@router.get("/recent-factchecks")
async def recent_factchecks(hours: int = 24) -> List[dict]:
    """Return recently fact-checked articles for Flutter dashboard."""
    return await batch_factcheck_recent(hours)


@router.get("/{article_id}")
def get_article(article_id: str) -> dict:
    """Return a single article by ID."""
    resp = (
        supabase.table("articles")
        .select("*")
        .eq("id", article_id)
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Article not found")
    return resp.data[0]


@router.post("/{article_id}/factcheck")
async def factcheck_article(article_id: str) -> dict:
    """Manually re-run fact-check for a single article."""
    resp = (
        supabase.table("articles")
        .select("*")
        .eq("id", article_id)
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Article not found")

    cred = await compute_credibility_score(
        ArticleResponse(**resp.data[0])
    )

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
