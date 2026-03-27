"""
NewsScope Fact-Checking Service.

CHANGES FROM v3:

1. structured_checks — compute_credibility_score() now returns a
   'structured_checks' list of normalised dicts ready for insertion
   into the fact_checks table. Previously only the raw JSONB blob
   was returned; rows were never written to the relational table.

2. credibility_updated_at — batch_factcheck_recent() now sets
   credibility_updated_at on every article it processes so the
   retroactive job can identify stale vs. never-checked articles.

3. fact_checks table writes — batch_factcheck_recent() deletes old
   rows for each article then inserts fresh structured_checks rows
   into the fact_checks table alongside the JSONB blob update.

4. retroactive_factcheck_all() — new function that queries articles
   where credibility_updated_at IS NULL (never checked) or older
   than _STALE_DAYS days (stale), and re-runs credibility scoring.
   Called by the /admin/factcheck/retroactive endpoint.

5. STALE_DAYS raised to 14 — article pool is 7 days, so articles
   near the pool edge need a second check window. Fact-checkers
   publish across a 1–14 day range; 7 days was too narrow to catch
   slower organisations and policy-claim reviews.

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

# Articles not checked in the last 14 days are considered stale.
# Raised from 7 — fact-checkers publish across a 1–14 day window
# and articles near the 7-day pool edge need a second pass.
_STALE_DAYS = 14

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

    if title and title.strip():
        headline_q = title.strip()[:100]
        queries.append(headline_q)

    if title and title.strip():
        title_kws = _extract_keywords_from_text(title, max_words=5)
        if title_kws:
            q2 = " ".join(title_kws)
            if q2 not in queries:
                queries.append(q2)

    if len(queries) < _MAX_KEYWORD_QUERIES:
        title_kws_short = _extract_keywords_from_text(
            title or "", max_words=3
        )
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


async def compute_credibility_score(
    article: ArticleResponse,
) -> Dict[str, Any]:
    """
    Compute a credibility score (0–100) for a given article.

    Formula:
      base  = _SOURCE_REPUTATION.get(source, 65)
      adj   = (mean valid fact-check scores - 0.5) * 40  → ±20 pts
      score = clamp(base + adj, 10, 100)

    Returns a dict including 'structured_checks' — a list of normalised
    rows ready for insertion into the fact_checks table.
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
            "structured_checks": [],
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

    # Build normalised rows for the fact_checks relational table.
    # Skips non-matches so only genuine ClaimReview hits are stored.
    now = datetime.now(timezone.utc).isoformat()
    structured_checks = []
    for query, result in fact_checks.items():
        ruling = result.get("ruling", "Unknown")
        if ruling in ("No match", "API Error", "Unverified"):
            continue
        structured_checks.append({
            "claim": result.get("matched_claim") or query,
            "rating": ruling,
            "source": result.get("publisher", "Unknown"),
            "link": result.get("url"),
            "checked_at": now,
        })

    return {
        "score": score,
        "fact_checks": fact_checks,
        "structured_checks": structured_checks,
        "claims_checked": total,
        "reason": reason,
    }


# ── Shared DB write helper ────────────────────────────────────────────────────


def _persist_credibility(
    article_id: str,
    cred: Dict[str, Any],
    now: str,
) -> None:
    """
    Write credibility results to the articles row and insert normalised
    rows into the fact_checks table. Deletes stale fact_checks rows first
    to avoid duplicates on re-checks.
    """
    supabase.table("articles").update({
        "credibility_score": cred["score"],
        "fact_checks": cred["fact_checks"],
        "claims_checked": cred["claims_checked"],
        "credibility_reason": cred["reason"],
        "credibility_updated_at": now,
        "updated_at": now,
    }).eq("id", article_id).execute()

    structured = cred.get("structured_checks", [])
    if structured:
        supabase.table("fact_checks") \
            .delete() \
            .eq("article_id", article_id) \
            .execute()
        supabase.table("fact_checks").insert([
            {**row, "article_id": article_id}
            for row in structured
        ]).execute()


# ── Batch re-check (three-stage decay) ───────────────────────────────────────


async def batch_factcheck_recent(hours: int = 48) -> List[Dict[str, Any]]:
    """
    Fact-check articles published in the last `hours` hours
    whose credibility score is at or below 80.1.

    Called by the scheduler at three frequencies to match the
    real-world fact-checker publication curve:

      Every 6h  → hours=24   catches same-day viral claim checks
      Every 24h → hours=72   catches peak 1–3 day check window
      Every 72h → retroactive_factcheck_all() for stale articles

    Writes results to both the articles row (JSONB blob + scalar
    fields) and the normalised fact_checks table.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat()

    recent: Optional[list] = (
        supabase.table("articles")
        .select("*")
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

    print(f"Fact-checking {len(recent)} articles ({hours}h window)...")
    results: List[Dict[str, Any]] = []

    for art in recent:
        try:
            cred = await compute_credibility_score(ArticleResponse(**art))
            now = datetime.now(timezone.utc).isoformat()
            _persist_credibility(art["id"], cred, now)
            results.append({"id": art["id"], **cred})
        except Exception as exc:
            print(f"Fact-check failed [{art['id']}]: {exc}")
            continue

    print(f"Fact-check cycle complete. {len(results)}/{len(recent)} scored.")
    return results


# ── Retroactive / stale re-check ─────────────────────────────────────────────


async def retroactive_factcheck_all(
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    Re-check all articles that have never been fact-checked
    (credibility_updated_at IS NULL) or haven't been checked in
    the last _STALE_DAYS (14) days.

    Scheduler calls this every 72h to catch slow fact-checkers
    and policy-claim reviews that publish up to two weeks later.
    Also used as a one-off backfill endpoint after deploy.
    """
    stale_cutoff = (
        datetime.now(timezone.utc) - timedelta(days=_STALE_DAYS)
    ).isoformat()

    never_checked = (
        supabase.table("articles")
        .select("*")
        .is_("credibility_updated_at", "null")
        .limit(limit)
        .execute()
        .data
    ) or []

    remaining = limit - len(never_checked)
    stale = []
    if remaining > 0:
        stale = (
            supabase.table("articles")
            .select("*")
            .lt("credibility_updated_at", stale_cutoff)
            .limit(remaining)
            .execute()
            .data
        ) or []

    articles = never_checked + stale
    if not articles:
        print("All articles are up to date.")
        return []

    print(
        f"Retroactive fact-check: {len(articles)} articles "
        f"({len(never_checked)} never checked, {len(stale)} stale)..."
    )
    results: List[Dict[str, Any]] = []

    for art in articles:
        try:
            cred = await compute_credibility_score(ArticleResponse(**art))
            now = datetime.now(timezone.utc).isoformat()
            _persist_credibility(art["id"], cred, now)
            results.append({"id": art["id"], **cred})
        except Exception as exc:
            print(f"Retroactive check failed [{art['id']}]: {exc}")
            continue

    print(
        f"Retroactive complete. {len(results)}/{len(articles)} scored."
    )
    return results
