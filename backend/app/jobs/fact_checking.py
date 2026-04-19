"""
NewsScope fact-checking service.

Scores each article's credibility (0-100) by combining a source
reputation baseline with results from the Google Fact Check Tools API.

Pipeline:
  1. Build up to 3 search queries from the article title and content.
  2. Query the Google Fact Check Tools API concurrently.
  3. Map textual rulings to numeric scores (0.0-1.0).
  4. Adjust the source reputation baseline by up to +/-20 points.
  5. Persist results to the articles row and the fact_checks table.
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

# Raised from 7 days -- fact-checkers publish across a 1-14 day window
# and articles near the pool edge need a second check pass.
_STALE_DAYS = 14

_GOOGLE_FACTCHECK_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
_GOOGLE_FACTCHECK_URL = (
    "https://factchecktools.googleapis.com/v1alpha1/claims:search"
)

# Source reputation baselines used when no fact-check results match.
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

# Common English stopwords excluded from keyword extraction.
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

# Temporal noise words filtered from queries. Named entities such as
# trump, iran, biden are intentionally kept -- fact-checker databases
# index those terms and removing them caused zero-match queries.
_PROPER_NOUN_STOPWORDS = {
    "says", "told", "year", "years", "also", "after", "first",
    "last", "time", "day", "week", "month",
    "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "one",
}

# Regex to strip common article boilerplate before processing content.
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


def _is_valid_content(text: str) -> bool:
    """Return True if the text is long enough and mostly printable."""
    if not text or len(text) < 40:
        return False
    sample = text[:500]
    printable = sum(1 for c in sample if c.isprintable())
    return (printable / len(sample)) > 0.85


def _extract_keywords_from_text(text: str, max_words: int = 5) -> List[str]:
    """
    Extract the most relevant keywords from a text string.

    Capitalised tokens (likely named entities) are prioritised over
    lowercase common words. Stopwords and temporal noise are excluded.
    """
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
    Build up to 3 queries for the Google Fact Check Tools API.

    Query 1: Full headline (up to 100 chars). Fact-checkers write
             ClaimReview entries against specific headlines so this
             gives the highest direct match rate.
    Query 2: Top keywords extracted from the headline, including named
             entities which are preserved for better index coverage.
    Query 3: Combined title and content keywords for broader topic
             coverage when the headline alone produces no match.
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


async def _check_single_query(
    client: httpx.AsyncClient, query: str
) -> tuple[str, Dict[str, Any]]:
    """
    Run a single query against the Google Fact Check Tools API.

    Retries once after a 2 second delay on any exception before
    returning an API Error result to the caller.
    """
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
    Run all queries concurrently within a shared httpx.AsyncClient.

    Returns a dict mapping each query string to its result dict.
    If GOOGLE_FACTCHECK_API_KEY is not configured, all queries
    return Unverified without making any network calls.
    """
    if not _GOOGLE_FACTCHECK_KEY:
        return {q: {"ruling": "Unverified", "url": None} for q in queries}

    async with httpx.AsyncClient(timeout=15.0) as client:
        results_list = await asyncio.gather(
            *[_check_single_query(client, q) for q in queries]
        )

    return dict(results_list)


def ruling_to_score(ruling: str) -> Optional[float]:
    """
    Map a textual fact-check ruling to a numeric score in [0.0, 1.0].

    Returns None for error, no-match, or unverified rulings so they
    are excluded from the credibility score adjustment calculation.
    """
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


async def compute_credibility_score(
    article: ArticleResponse,
) -> Dict[str, Any]:
    """
    Compute a credibility score (0-100) for a given article.

    Formula:
      base  = _SOURCE_REPUTATION.get(source, 65)
      adj   = (mean valid ruling scores - 0.5) * 40  -- max +/-20 pts
      score = clamp(base + adj, 10, 100)

    Also returns 'structured_checks': a list of normalised dicts
    ready for insertion into the fact_checks relational table.
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

    # Build normalised rows for the fact_checks table. Non-matches
    # and errors are skipped so only genuine ClaimReview hits are stored.
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


def _persist_credibility(
    article_id: str,
    cred: Dict[str, Any],
    now: str,
) -> None:
    """
    Write credibility results to the articles row and insert normalised
    rows into the fact_checks table. Deletes existing fact_checks rows
    first to prevent duplicates on subsequent re-checks.
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


async def batch_factcheck_recent(hours: int = 48) -> List[Dict[str, Any]]:
    """
    Fact-check articles published in the last `hours` hours whose
    credibility score is at or below 80.1.

    Called by the scheduler at three frequencies to match the
    real-world fact-checker publication curve:
      Every 6h  -- hours=24 catches same-day viral claim checks
      Every 24h -- hours=72 catches the peak 1-3 day check window
      Every 72h -- retroactive_factcheck_all handles stale articles

    Results are written to both the articles row (JSONB blob and scalar
    fields) and the normalised fact_checks relational table.
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


async def retroactive_factcheck_all(
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    Re-check all articles that have never been fact-checked or have not
    been checked within the last _STALE_DAYS (14) days.

    Called by the scheduler every 72 hours to catch slow fact-checkers
    and policy-claim reviews that publish up to two weeks after the
    original story. Also used as a one-off backfill endpoint post-deploy.
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


def extract_keywords(text, max_keywords=10):
    """
    Filter stopwords, prioritise capitalised tokens, return top keywords.
    Used by tests/unit/test_keywords.py.
    """
    if not text:
        return []
    words = text.split()
    filtered = [w.strip(".,;:!?\"'()[]{}") for w in words if w.strip()]
    filtered = [w for w in filtered if w and w.lower() not in _STOPWORDS]
    proper = [w for w in filtered if w and w[0].isupper()]
    rest = [w for w in filtered if w not in proper]
    seen = set()
    result = []
    for w in proper + rest:
        if w.lower() not in seen:
            seen.add(w.lower())
            result.append(w)
        if len(result) >= max_keywords:
            break
    return result


def compute_credibility_score_sync(source_name, rulings):
    """
    Sync wrapper of the credibility formula for unit testing.

    The production compute_credibility_score is async and takes an
    ArticleResponse. This wrapper exposes the underlying scoring logic
    directly against a source name and a list of ruling strings.
    Used by tests/unit/test_credibility.py.
    """
    base = _SOURCE_REPUTATION.get(source_name, 70.0)
    valid = [
        s for s in (ruling_to_score(r) for r in rulings)
        if s is not None
    ]
    if not valid:
        return int(round(base))
    mean_ruling = sum(valid) / len(valid)
    adjustment = (mean_ruling - 0.5) * 40
    score = base + adjustment
    return int(round(max(10.0, min(100.0, score))))
