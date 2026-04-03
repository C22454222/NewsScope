"""
NewsScope Articles API Router.

Credibility scoring + fact-checking endpoints.
Category filtering uses CATEGORY_GROUP_MAP to resolve sub-categories
(football, climate, film, etc.) to their parent group so Flutter's
chip filter correctly matches all stored variants.

Archive window: 7 days — weekly rolling article pool.

Flake8: 0 errors/warnings.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.db.supabase import supabase
from app.jobs.fact_checking import (
    batch_factcheck_recent,
    compute_credibility_score,
    retroactive_factcheck_all,
)
from app.schemas import ArticleCreate, ArticleResponse


router = APIRouter(tags=["articles"])

# Rolling archive window — 7 days for a weekly article pool.
_ARCHIVE_DAYS = 7

# Maps granular backend sub-categories → Flutter parent chip categories.
# Ensures ?category=sport returns articles tagged football/rugby/gaa etc.
# Mirrors the chip categories in HomeFeedTab._categories (lowercased).
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


def _archive_cutoff() -> str:
    """ISO timestamp for the start of the rolling archive window."""
    return (
        datetime.now(timezone.utc) - timedelta(days=_ARCHIVE_DAYS)
    ).isoformat()


def _write_fact_check_rows(article_id: str, structured_checks: list) -> None:
    """
    Upsert normalised fact-check rows into the fact_checks table.
    Deletes existing rows for the article first to avoid duplicates.
    No-op if structured_checks is empty.
    """
    if not structured_checks:
        return
    supabase.table("fact_checks") \
        .delete() \
        .eq("article_id", article_id) \
        .execute()
    rows = [
        {**row, "article_id": article_id}
        for row in structured_checks
    ]
    supabase.table("fact_checks").insert(rows).execute()


@router.get("")
def get_articles(
    category: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
) -> List[dict]:
    """
    Recent articles (7d rolling window), optionally filtered by
    category and/or source.

    Category resolves sub-categories via CATEGORY_GROUP_MAP so
    ?category=sport returns articles tagged sport, football, rugby etc.
    Source matches the exact source name stored in the articles table,
    e.g. ?source=BBC+News or ?source=RTÉ+News.
    """
    query = (
        supabase.table("articles")
        .select("*")
        .gte("published_at", _archive_cutoff())
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
    query = (
        supabase.table("articles")
        .select("*")
        .gte("published_at", _archive_cutoff())
    )

    if topic:
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
    now = datetime.now(timezone.utc).isoformat()

    insert_data = {
        "source": article.source,
        "url": article.url,
        "title": article.title,
        "content": article.content,
        "bias_score": article.bias_score,
        "bias_intensity": article.bias_intensity,
        "sentiment_score": article.sentiment_score,
        "general_bias": article.general_bias,
        "general_bias_score": article.general_bias_score,
        "political_bias": article.political_bias,
        "political_bias_score": article.political_bias_score,
        "published_at": (
            article.published_at.isoformat()
            if article.published_at else None
        ),
        "category": article.category,
        "credibility_score": 80.0,
        "fact_checks": {},
        "claims_checked": 0,
        "credibility_reason": "Pending",
        "credibility_updated_at": None,
        "updated_at": now,
    }

    cred = await compute_credibility_score(ArticleResponse(**insert_data))
    insert_data.update({
        "credibility_score": cred["score"],
        "fact_checks": cred["fact_checks"],
        "claims_checked": cred["claims_checked"],
        "credibility_reason": cred["reason"],
        "credibility_updated_at": now,
    })

    resp = supabase.table("articles").insert(insert_data).execute()
    if not resp.data:
        raise HTTPException(status_code=500, detail="Insert failed")

    article_id = resp.data[0]["id"]
    _write_fact_check_rows(article_id, cred.get("structured_checks", []))

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
    """
    Manually re-run fact-check for a single article.
    Updates both articles and fact_checks tables.
    """
    resp = (
        supabase.table("articles")
        .select("*")
        .eq("id", article_id)
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Article not found")

    now = datetime.now(timezone.utc).isoformat()
    cred = await compute_credibility_score(ArticleResponse(**resp.data[0]))

    supabase.table("articles").update({
        "credibility_score": cred["score"],
        "fact_checks": cred["fact_checks"],
        "claims_checked": cred["claims_checked"],
        "credibility_reason": cred["reason"],
        "credibility_updated_at": now,
        "updated_at": now,
    }).eq("id", article_id).execute()

    _write_fact_check_rows(article_id, cred.get("structured_checks", []))

    return cred


@router.post("/admin/factcheck/recent")
async def run_recent_factcheck(hours: int = 48) -> dict:
    """
    Re-run fact-checks for articles in the last `hours` hours
    with credibility score at or below 80.1.
    """
    results = await batch_factcheck_recent(hours=hours)
    return {"checked": len(results)}


@router.post("/admin/factcheck/retroactive")
async def run_retroactive_factcheck(limit: int = 500) -> dict:
    """
    Backfill fact-checks for all articles that have never been
    checked or haven't been checked in the last 7 days.
    Run once after deploy, then leave to the scheduler.
    """
    results = await retroactive_factcheck_all(limit=limit)
    return {"checked": len(results)}
