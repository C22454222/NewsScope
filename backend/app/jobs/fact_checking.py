"""
NewsScope Fact-Checking Service.

CHANGES FROM v2:

1. CRITICAL FIX — ArticleResponse validation error:
   batch_factcheck_recent() was selecting only 4 columns
   (id, title, content, source) but ArticleResponse requires all columns.
   Changed to select("*") so Pydantic gets every field it needs.
   This was causing 100% of fact-checks to fail silently.

2. QUERY QUALITY — headline sent as direct query:
   build_search_queries() now sends the full headline (truncated to 100
   chars) as query 1. Fact-checkers index claims by subject matter — a
   direct headline match is far more likely to return a ClaimReview than
   a bag of 5 extracted keywords. Keywords are still used for queries 2
   and 3 as fallback coverage.

3. NAMED ENTITIES PRESERVED — _PROPER_NOUN_STOPWORDS slimmed down:
   The previous list stripped trump, iran, israel, biden, russia, china
   etc. — exactly the entities that appear in fact-checker databases.
   These have been removed from the stopword list so they survive into
   the keyword queries.

4. RULING SCORING — broader pattern matching:
   Added "rated false", "not true", "inaccurate", "exaggerated",
   "distorted" to the ruling normaliser to capture more fact-checker
   verdict formats beyond the most common English phrases.

5. BATCH SELECT — select("*") instead of select("id,title,content,source"):
   Full record needed for ArticleResponse instantiation.

Flake8: 0 errors/warnings.
"""

import asyncio
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from app.schemas import ArticleResponse
from app.db.supabase import supabase


_MAX_KEYWORD_QUERIES = 3

_FACTCHECK_BATCH_SIZE = 100

_GOOGLE_FACTCHECK_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
_GOOGLE_FACTCHECK_URL = (
    "https://factchecktools.googleapis.com/v1alpha1/claims:search"
)

_SOURCE_REPUTATION: Dict[str, float] = {
    "Reuters": 93,
    "Associated Press": 92,
    "AFP": 91,
    "BBC News": 88,
    "RTÉ News": 86,
    "NPR": 85,
    "The Irish Times": 84,
    "The Guardian": 82,
    "The Independent": 76,
    "CNN": 72,
    "Politico": 74,
    "Politico Europe": 74,
    "Fox News": 55,
    "GB News": 50,
    "Euronews": 72,
    "Sky News": 74,
    "AP News": 92,
    "Deutsche Welle": 80,
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "has", "have", "had", "will", "would", "could", "should", "may",
    "that", "this", "its", "it", "be", "been", "as", "after",
    "before", "over", "also", "said", "says", "say", "new", "he",
    "she", "they", "their", "his", "her", "our", "we", "who", "what",
    "when", "where", "how", "which", "than", "more", "not", "no",
    "up", "out", "about", "into", "than", "us", "can",
}

# Stripped down to only genuine noise — named entities (trump, iran,
# biden etc.) were removed because fact-checker databases index exactly
# these terms. Keeping them in was causing zero-match queries.
_PROPER_NOUN_STOPWORDS = {
    "says", "told", "year", "years", "also", "after", "first",
    "last", "time", "day", "week", "month",
    "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "one",
}

_BOILERPLATE_RE = re.compile(
    r"^("
    r"NEW\b|close\b|Video|Watch|Listen|"
    r"Advertisement|ADVERTISEMENT|"
    r"You can now listen|Follow on|Follow us|"
    r"Subscribe|Sign up|Click here|"
    r"Fox News Digital|Share this|Read more|"
    r"Send tips|Previous bylines|"
    r"Getty Images|Photograph:"
    r")",
    re.IGNORECASE,
)


# ── Content validation ────────────────────────────────────────────────────────


def _is_valid_content(text: str) -> bool:
    if not text or len(text) < 40:
        return False
    sample = text[:500]
    printable = sum(1 for c in sample if c.isprintable())
    return (printable / len(sample)) > 0.85


# ── Keyword extraction ────────────────────────────────────────────────────────


def _extract_keywords_from_text(text: str, max_words: int = 5) -> List[str]:
    words = re.findall(r"\b[A-Za-z][a-zA-Z'\-]{2,}\b|\b\d{4}\b", text)
    seen: Dict[str, int] = {}
    for w in words:
        lower = w.lower()
        if (
            lower not in _STOPWORDS and lower not in _PROPER_NOUN_STOPWORDS and len(lower) > 2
        ):
            seen[lower] = seen.get(lower, 0) + 1

    ranked = sorted(
        seen.keys(),
        key=lambda w: (-int(w[0].isupper()), -seen[w]),
    )
    return ranked[:max_words]


def build_search_queries(title: str, content: str) -> List[str]:
    """
    Build up to 3 queries for the Google Fact Check API.

    Query 1: Full headline (≤100 chars) — most likely to match a
             ClaimReview entry directly. Fact-checkers write reviews
             against specific claims or headlines, so sending the
             actual headline first gives the best match rate.

    Query 2: Top keywords from headline, including named entities
             (trump, iran, israel etc. are now preserved).

    Query 3: Combined title + content keywords for topic coverage.
    """
    queries: List[str] = []

    # Query 1: headline direct — best match against ClaimReview entries
    if title and title.strip():
        headline_q = title.strip()[:100]
        queries.append(headline_q)

    # Query 2: keyword extraction from headline
    if title and title.strip():
        title_kws = _extract_keywords_from_text(title, max_words=5)
        if title_kws:
            q2 = " ".join(title_kws)
            if q2 not in queries:
                queries.append(q2)

    # Query 3: combined title + content keywords
    if len(queries) < _MAX_KEYWORD_QUERIES:
        title_kws_short = _extract_keywords_from_text(title or "", max_words=3)
        content_kws = []
        if content and _is_valid_content(content):
            content_kws = _extract_keywords_from_text(
                content[:800], max_words=3
            )
        combined = list(dict.fromkeys(title_kws_short + content_kws))[:5]
        q3 = " ".join(combined)
        if q3 and q3 not in queries:
            queries.append(q3)

    return queries[:_MAX_KEYWORD_QUERIES]


# ── Google Fact Check Tools API ───────────────────────────────────────────────


async def _check_single_query(
    client: httpx.AsyncClient, query: str
) -> tuple[str, Dict[str, Any]]:
    for attempt in range(2):
        try:
            resp = await client.get(
                _GOOGLE_FACTCHECK_URL,
                params={
                    "query": query,
                    "key": _GOOGLE_FACTCHECK_KEY,
                    "languageCode": "en",
                    "pageSize": 3,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            fact_claims = data.get("claims", [])
            if fact_claims:
                for claim in fact_claims:
                    reviews = claim.get("claimReview", [])
                    if reviews:
                        top = reviews[0]
                        return query, {
                            "ruling": top.get("textualRating", "Unknown"),
                            "url": top.get("url"),
                            "publisher": top.get("publisher", {}).get(
                                "name", "Unknown"
                            ),
                            "matched_claim": claim.get("text", "")[:120],
                        }
            return query, {"ruling": "No match", "url": None}

        except Exception:
            if attempt == 0:
                await asyncio.sleep(2)
            else:
                return query, {"ruling": "API Error", "url": None}

    return query, {"ruling": "API Error", "url": None}


async def check_google_factclaims(queries: List[str]) -> Dict[str, Any]:
    """
    All queries run concurrently within a shared httpx.AsyncClient.
    If GOOGLE_FACTCHECK_API_KEY is not set, returns Unverified for all.
    """
    if not _GOOGLE_FACTCHECK_KEY:
        return {q: {"ruling": "Unverified", "url": None} for q in queries}

    async with httpx.AsyncClient(timeout=15.0) as client:
        results_list = await asyncio.gather(
            *[_check_single_query(client, q) for q in queries]
        )

    return dict(results_list)


# ── Ruling normalisation ──────────────────────────────────────────────────────


def ruling_to_score(ruling: str) -> Optional[float]:
    if not ruling:
        return None
    r = ruling.lower()

    if any(x in r for x in ("error", "no match", "unverified")):
        return None

    if r in ("true", "correct", "accurate", "verified"):
        return 1.0
    if any(x in r for x in (
        "mostly true", "largely true", "mostly correct", "mostly accurate"
    )):
        return 0.8
    if any(x in r for x in (
        "half true", "partially true", "mixed", "partially correct",
        "unproven", "unsubstantiated", "needs context", "exaggerated",
        "distorted", "misleading context",
    )):
        return 0.5
    if any(x in r for x in (
        "mostly false", "largely false", "misleading",
        "not accurate", "inaccurate", "not true", "rated false",
    )):
        return 0.2
    if any(x in r for x in (
        "false", "pants", "incorrect", "wrong", "fabricated",
        "scam", "hoax", "debunked",
    )):
        return 0.0

    return 0.5


# ── Credibility scoring ───────────────────────────────────────────────────────


async def compute_credibility_score(article: ArticleResponse) -> Dict[str, Any]:
    """
    Compute a credibility score (0–100) for a given article.

    Formula:
      base  = _SOURCE_REPUTATION.get(source, 65)
      adj   = (mean valid fact-check scores - 0.5) * 40  → ±20 pts
      score = clamp(base + adj, 10, 100)
    """
    source = getattr(article, "source", None) or ""
    base_score = _SOURCE_REPUTATION.get(source, 65.0)

    queries = build_search_queries(
        title=article.title or "",
        content=article.content or "",
    )

    if not queries:
        return {
            "score": float(base_score),
            "reason": (
                f"No queries could be built. "
                f"Score based on source reputation ({source or 'unknown'})."
            ),
            "fact_checks": {},
            "claims_checked": 0,
        }

    fact_checks = await check_google_factclaims(queries)

    scores = [
        s for s in
        (ruling_to_score(r["ruling"]) for r in fact_checks.values())
        if s is not None
    ]

    adjustment = (
        (sum(scores) / len(scores) - 0.5) * 40 if scores else 0.0
    )
    score = round(max(10.0, min(100.0, base_score + adjustment)), 2)

    positive = sum(1 for s in scores if s >= 0.7)
    negative = sum(1 for s in scores if s < 0.3)
    total = len(queries)
    matched = len(scores)

    if score >= 70:
        verdict = "reliable"
    elif score >= 40:
        verdict = "mixed"
    else:
        verdict = "low credibility"

    if scores:
        neg_str = f", {negative} false" if negative else ""
        reason = (
            f"{matched}/{total} queries matched fact-checks. "
            f"{positive} verified true{neg_str}. "
            f"Source: {source or 'unknown'}. Rated {verdict}."
        )
    else:
        reason = (
            f"No fact-checks found for topic. "
            f"Score based on source reputation ({source or 'unknown'}). "
            f"Rated {verdict}."
        )

    return {
        "score": score,
        "fact_checks": fact_checks,
        "claims_checked": total,
        "reason": reason,
    }


# ── Batch re-check ────────────────────────────────────────────────────────────


async def batch_factcheck_recent(hours: int = 48) -> List[Dict[str, Any]]:
    """
    Fact-check articles published in the last `hours` hours
    whose credibility score is at or below 80.1.

    CRITICAL FIX: changed .select("id, title, content, source") to
    .select("*") — ArticleResponse is a Pydantic model that requires
    all columns. Selecting only 4 fields caused a validation error on
    every single article, producing 0/100 scored every cycle.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat()

    recent: Optional[list] = (
        supabase.table("articles")
        .select("*")          # ← was select("id, title, content, source")
        .gte("published_at", cutoff)
        .lte("credibility_score", 80.1)
        .order("published_at", desc=True)
        .limit(_FACTCHECK_BATCH_SIZE)
        .execute()
        .data
    )

    if not recent:
        print("No articles require fact-checking.")
        return []

    print(f"Fact-checking {len(recent)} articles (48h window)...")
    results: List[Dict[str, Any]] = []

    for art in recent:
        try:
            cred = await compute_credibility_score(ArticleResponse(**art))
            supabase.table("articles").update(
                {
                    "credibility_score": cred["score"],
                    "fact_checks": cred["fact_checks"],
                    "claims_checked": cred["claims_checked"],
                    "credibility_reason": cred["reason"],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            ).eq("id", art["id"]).execute()
            results.append({"id": art["id"], **cred})
        except Exception as exc:
            print(f"Fact-check failed [{art['id']}]: {exc}")
            continue

    print(f"Fact-check cycle complete. {len(results)}/{len(recent)} scored.")
    return results
