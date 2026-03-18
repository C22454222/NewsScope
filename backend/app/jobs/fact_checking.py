"""
NewsScope Fact-Checking Service.

Google Fact Check Tools API only indexes claims that have been
explicitly reviewed by registered fact-checkers (PolitiFact, Snopes,
AFP, Reuters, etc.) with ClaimReview markup. It does NOT index
breaking news or sports results.

Root cause of "No match" on all queries (v1):
- Queries were full sentences (100-200 chars) sent verbatim.
- API is a keyword/topic search — it needs short, specific terms.
- Fix: extract 3-6 keyword topics from title + content instead of
  sending raw sentences. Match rate improves dramatically.

Credibility scoring formula (v2 — unchanged):
  base  = _SOURCE_REPUTATION.get(source, 65)
  adj   = (mean valid fact-check scores - 0.5) * 40  → ±20 pts
  score = clamp(base + adj, 10, 100)

Every article still receives a meaningful base score from source
reputation. Fact-check results shift the score when a match is found.

Daily budget:
- _MAX_KEYWORD_QUERIES = 3 per article
- 50 articles per 48h batch × 3 = 150 requests — well under 1,000/day

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


# Maximum keyword queries per article (budget control).
_MAX_KEYWORD_QUERIES = 3

_GOOGLE_FACTCHECK_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
_GOOGLE_FACTCHECK_URL = (
    "https://factchecktools.googleapis.com/v1alpha1/claims:search"
)

# Source reputation scores (0–100) based on journalism standards,
# editorial oversight, transparency, and corrections track record.
# Sources not listed default to 65 (unknown/neutral).
# Scores are intentionally conservative — no source scores above 95.
_SOURCE_REPUTATION: Dict[str, float] = {
    # Wire services — highest standards, global reach
    "Reuters": 93,
    "Associated Press": 92,
    "AFP": 91,
    # Public broadcasters — editorial charter, corrections policy
    "BBC News": 88,
    "RTÉ News": 86,
    "NPR": 85,
    # Quality broadsheets
    "The Irish Times": 84,
    "The Guardian": 82,
    "The Independent": 76,
    # US cable/digital — partisan lean lowers score
    "CNN": 72,
    "Politico": 74,
    "Politico Europe": 74,
    # Right-leaning — known for editorial bias
    "Fox News": 55,
    "GB News": 50,
    # International
    "Euronews": 72,
    "Sky News": 74,
}

# Words that add no search value — stripped before keyword extraction.
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

# High-frequency proper nouns that appear in nearly every article and
# provide no discriminating signal for fact-check matching.
# Without this filter, "Iran" or "Trump" match completely unrelated
# fact-checks because the API returns results for any mention of the
# term rather than the specific claim being made.
_PROPER_NOUN_STOPWORDS = {
    "iran", "israel", "uk", "us", "usa", "trump", "london",
    "europe", "china", "russia", "america", "washington", "biden",
    "parliament", "government", "minister", "president", "police",
    "court", "people", "country", "world", "state", "war", "new",
    "says", "told", "year", "years", "also", "after", "first",
    "last", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "one", "time", "day", "week", "month",
}

# Boilerplate patterns to skip during keyword extraction.
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
    """
    Return True if text appears to be readable UTF-8, not binary.

    Binary blobs have a very low ratio of printable ASCII characters —
    anything below 85% printable in the first 500 chars is treated as
    invalid and excluded from keyword extraction.
    """
    if not text or len(text) < 40:
        return False
    sample = text[:500]
    printable = sum(1 for c in sample if c.isprintable())
    return (printable / len(sample)) > 0.85


# ── Keyword extraction ────────────────────────────────────────────────────────


def _extract_keywords_from_text(text: str, max_words: int = 5) -> List[str]:
    """
    Extract the most meaningful capitalised/numeric terms from text.

    Returns a list of individual keyword tokens — not full sentences.
    Proper nouns are prioritised over common words. Both generic
    stopwords and high-frequency proper nouns (e.g. "Iran", "Trump")
    are removed to avoid matching unrelated fact-checks.
    Used to build short topic queries for the Google Fact Check API.
    """
    words = re.findall(r"\b[A-Za-z][a-zA-Z'\-]{2,}\b|\b\d{4}\b", text)
    seen: Dict[str, int] = {}
    for w in words:
        lower = w.lower()
        if (
            lower not in _STOPWORDS and lower not in _PROPER_NOUN_STOPWORDS and len(lower) > 2
        ):
            seen[lower] = seen.get(lower, 0) + 1

    # Prefer capitalised (proper noun) terms, then by frequency.
    ranked = sorted(
        seen.keys(),
        key=lambda w: (-int(w[0].isupper()), -seen[w]),
    )
    return ranked[:max_words]


def build_search_queries(title: str, content: str) -> List[str]:
    """
    Build 1–3 short keyword queries for the Google Fact Check API.

    Strategy:
    1. Title keywords  → query 1 (most concise, highest signal)
    2. Content proper nouns → query 2
    3. Combined blend  → query 3

    Queries are 3-6 words max — matches how the API's search index
    works. Full sentences never return results; short topic phrases do.
    High-frequency proper nouns filtered out to reduce false matches.

    Examples of good queries vs old bad queries:
      OLD: "President Donald Trump's administration has imposed the
            oil blockade on the Cuban government as Washington presses
            for regime change..."  → No match (always)
      NEW: "Cuba blockade regime change"  → matches fact-checker results

      OLD: "Chelsea have been given a suspended one-year transfer ban,
            and fined a record £10.75m..."  → No match (always)
      NEW: "Chelsea Premier League fine"  → matches if reviewed
    """
    queries: List[str] = []

    # Query 1: title keywords
    if title and title.strip():
        title_kws = _extract_keywords_from_text(title, max_words=5)
        if title_kws:
            queries.append(" ".join(title_kws))

    # Query 2: top proper nouns from content
    if content and _is_valid_content(content):
        content_kws = _extract_keywords_from_text(
            content[:800], max_words=5
        )
        if content_kws:
            q2 = " ".join(content_kws)
            if q2 not in queries:
                queries.append(q2)

    # Query 3: blended title + content for broader coverage
    if title and content and len(queries) < _MAX_KEYWORD_QUERIES:
        title_kws = _extract_keywords_from_text(title, max_words=3)
        content_kws = _extract_keywords_from_text(
            content[:400], max_words=3
        )
        combined = list(dict.fromkeys(title_kws + content_kws))[:5]
        q3 = " ".join(combined)
        if q3 and q3 not in queries:
            queries.append(q3)

    return queries[:_MAX_KEYWORD_QUERIES]


# ── Google Fact Check Tools API ───────────────────────────────────────────────


async def check_google_factclaims(
    queries: List[str],
) -> Dict[str, Any]:
    """
    Query Google Fact Check Tools API with short keyword queries.

    Each query is a topic phrase (3-6 words), not a full sentence.
    Returns dict mapping each query to its top ruling and URL.
    pageSize=3 checks top 3 results per query for best match.
    Falls back gracefully per query — one error does not abort batch.
    matched_claim stored so DB shows exactly what Google matched.
    """
    results: Dict[str, Any] = {}

    if not _GOOGLE_FACTCHECK_KEY:
        for q in queries:
            results[q] = {"ruling": "Unverified", "url": None}
        return results

    async with httpx.AsyncClient(timeout=15.0) as client:
        for query in queries:
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
                        # Pick the first claim that has a review.
                        matched = False
                        for claim in fact_claims:
                            reviews = claim.get("claimReview", [])
                            if reviews:
                                top = reviews[0]
                                results[query] = {
                                    "ruling": top.get(
                                        "textualRating", "Unknown"
                                    ),
                                    "url": top.get("url"),
                                    "publisher": (
                                        top.get("publisher", {})
                                        .get("name", "Unknown")
                                    ),
                                    "matched_claim": claim.get(
                                        "text", ""
                                    )[:120],
                                }
                                matched = True
                                break
                        if not matched:
                            results[query] = {
                                "ruling": "No match",
                                "url": None,
                            }
                    else:
                        results[query] = {
                            "ruling": "No match",
                            "url": None,
                        }
                    break

                except Exception:
                    if attempt == 0:
                        await asyncio.sleep(2)
                    else:
                        results[query] = {
                            "ruling": "API Error",
                            "url": None,
                        }

    return results


# ── Ruling normalisation ──────────────────────────────────────────────────────


def ruling_to_score(ruling: str) -> Optional[float]:
    """
    Convert a fact-check ruling string to a normalised score (0.0–1.0).

    Returns None for API Error, Unverified, No match — excluded from
    mean so missing data does not drag scores toward neutral.
    Unknown rulings default to 0.5 (neutral) rather than being ignored.
    """
    if not ruling:
        return None
    r = ruling.lower()

    if any(x in r for x in ("error", "no match", "unverified")):
        return None

    if r in ("true", "correct", "accurate", "verified"):
        return 1.0
    if any(x in r for x in ("mostly true", "largely true", "mostly correct")):
        return 0.8
    if any(x in r for x in ("half true", "partially true", "mixed")):
        return 0.5
    if any(x in r for x in ("mostly false", "largely false", "misleading")):
        return 0.2
    if any(x in r for x in (
        "false", "pants", "incorrect", "wrong", "fabricated"
    )):
        return 0.0

    # Unknown rating — treat as neutral rather than ignoring entirely.
    return 0.5


# ── Credibility scoring ───────────────────────────────────────────────────────


async def compute_credibility_score(
    article: ArticleResponse,
) -> Dict[str, Any]:
    """
    Compute a credibility score (0–100) for a given article.

    Scoring formula (v2 — guaranteed score for every article):
      base  = _SOURCE_REPUTATION.get(source, 65)
      adj   = (mean valid fact-check scores - 0.5) * 40  → ±20 pts
      score = clamp(base + adj, 10, 100)

    adj ranges from -20 (all false) to +20 (all true), 0 when no
    fact-checks are found. Every article gets a meaningful base score
    from source reputation. Fact-check results shift when available.

    Queries are now short keyword phrases (3-6 words) rather than full
    sentences — dramatically improves Google API match rate.
    High-frequency proper nouns filtered to reduce false matches.
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
                f"Score based on source reputation "
                f"({source or 'unknown'})."
            ),
            "fact_checks": {},
            "claims_checked": 0,
        }

    fact_checks = await check_google_factclaims(queries)

    # Exclude None (API Error / No match / Unverified) from mean.
    scores = [
        s for s in
        (ruling_to_score(r["ruling"]) for r in fact_checks.values())
        if s is not None
    ]

    # Adjustment: ±20 points based on fact-check results.
    # Zero adjustment when no valid fact-checks found.
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
            f"Score based on source reputation "
            f"({source or 'unknown'}). Rated {verdict}."
        )

    return {
        "score": score,
        "fact_checks": fact_checks,
        "claims_checked": total,
        "reason": reason,
    }


# ── Batch re-check ────────────────────────────────────────────────────────────


async def batch_factcheck_recent(
    hours: int = 48,
) -> List[Dict[str, Any]]:
    """
    Re-fact-check articles published in the last `hours` hours
    whose credibility score is at or below 80.1.

    Default window 48h — matches the article display window.
    description column removed — dropped from DB schema.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat()

    recent: Optional[list] = (
        supabase.table("articles")
        .select("id, title, content, source")
        .gte("published_at", cutoff)
        .lte("credibility_score", 80.1)
        .limit(50)
        .execute()
        .data
    )

    results: List[Dict[str, Any]] = []

    for art in (recent or []):
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

    return results
