"""
NewsScope Fact-Checking Service.

Integrates Google Fact Check Tools API with lightweight regex
claim extraction and source reputation scoring.

PolitiFact has been replaced with Google Fact Check Tools API:
- PolitiFact's public search API was blocking Render's shared IP
  range, returning connection errors on every request regardless
  of headers or retry strategy.
- Google Fact Check Tools API is a proper public REST API (free
  tier: 1,000 req/day) that aggregates fact-checks from PolitiFact,
  Reuters, AFP, Snopes, and others — richer than PolitiFact alone.
- No IP restrictions. API key auth only. No User-Agent tricks needed.
- Add GOOGLE_FACTCHECK_API_KEY to Render env vars (free at
  console.cloud.google.com → Fact Check Tools API → Credentials).

Daily request budget:
- _MAX_CLAIMS = 2 content claims + 1 title = 3 max per article.
- batch_factcheck_recent caps at 50 articles per 48h window.
- Worst case: 50 * 3 = 150 requests per 48h cycle — well under 1,000.

SOURCE REPUTATION SCORING:
- Every article receives a meaningful credibility score regardless
  of Google Fact Check coverage.
- Base score derived from known source reputation (journalism
  standards, editorial oversight, track record of corrections).
- Google fact-check results act as a ±20 point adjustment when
  a match is found — they do not replace the base score.
- Sources not in _SOURCE_REPUTATION default to 65 (neutral/unknown).
- This guarantees every article has a justified, non-arbitrary score.

spaCy has been removed entirely — it loaded 50MB+ into Render RAM
on first call, spiking memory when factcheck overlapped with analysis.
Claims are now extracted using simple sentence splitting and regex,
which has near-zero RAM cost and produces comparable quality input
for the fact-check API query.

Claim extraction strategy:
- Title is always included as claim 0 — it is the most concise and
  fact-checkable representation of the article.
- Up to _MAX_CLAIMS additional sentences are extracted from content
  using numeric and proper-noun heuristics.
- Total claims capped at _MAX_CLAIMS + 1 (title + extracted).
- Sentences matching _BOILERPLATE_RE are skipped to prevent Fox News
  and other site chrome leaking into fact-check queries.

Credibility scoring formula (v2):
  base  = _SOURCE_REPUTATION.get(source, 65)
  adj   = (mean valid fact-check scores - 0.5) * 40
          (ranges from -20 to +20, 0 when no fact-checks found)
  score = clamp(base + adj, 10, 100)

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


# Maximum claims extracted from content (excluding title).
# Title is always prepended, so total queries = _MAX_CLAIMS + 1 max.
_MAX_CLAIMS = 2

# Google Fact Check Tools API — free, reliable, no IP restrictions.
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

# Boilerplate patterns to skip during claim extraction.
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
    invalid and excluded from claim extraction.
    """
    if not text or len(text) < 40:
        return False
    sample = text[:500]
    printable = sum(1 for c in sample if c.isprintable())
    return (printable / len(sample)) > 0.85


# ── Claim extraction ──────────────────────────────────────────────────────────


def extract_claims(title: str, content: str) -> List[str]:
    """
    Extract up to _MAX_CLAIMS + 1 verifiable claims from an article.

    Title is always included as claim 0. Additional claims extracted
    from content using regex heuristics — no spaCy or NLP models.
    Sentences matching _BOILERPLATE_RE are skipped.
    """
    claims: List[str] = []

    if title and title.strip():
        claims.append(title.strip())

    if not content or not _is_valid_content(content):
        return claims

    raw_sentences = re.split(r"(?<=[.!?])\s+", content)
    extracted: List[str] = []

    for sent in raw_sentences:
        sent = sent.strip()

        if sent.endswith("?") or len(sent) < 40:
            continue
        if _BOILERPLATE_RE.match(sent):
            continue

        has_number = bool(re.search(r"\b\d+(?:[,.\d]+)?%?\b", sent))
        has_proper_noun = bool(
            re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", sent)
        )

        if has_number or has_proper_noun:
            extracted.append(sent)

        if len(extracted) >= _MAX_CLAIMS:
            break

    if not extracted:
        for sent in raw_sentences:
            sent = sent.strip()
            if (
                len(sent) >= 40 and not sent.endswith("?") and not _BOILERPLATE_RE.match(sent)
            ):
                extracted.append(sent)
            if len(extracted) >= _MAX_CLAIMS:
                break

    claims.extend(extracted)
    return claims


# ── Google Fact Check Tools API ───────────────────────────────────────────────


async def check_google_factclaims(
    claims: List[str],
) -> Dict[str, Any]:
    """
    Query Google Fact Check Tools API for each extracted claim.

    Returns a dict mapping each claim to its top ruling and URL.
    Fails gracefully per claim — one error does not abort the batch.
    Falls back to Unverified when key is missing or no match found.
    """
    results: Dict[str, Any] = {}

    if not _GOOGLE_FACTCHECK_KEY:
        for claim in claims:
            results[claim] = {"ruling": "Unverified", "url": None}
        return results

    async with httpx.AsyncClient(timeout=15.0) as client:
        for claim in claims:
            for attempt in range(2):
                try:
                    resp = await client.get(
                        _GOOGLE_FACTCHECK_URL,
                        params={
                            "query": claim[:200],
                            "key": _GOOGLE_FACTCHECK_KEY,
                            "languageCode": "en",
                            "pageSize": 1,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    fact_claims = data.get("claims", [])
                    if fact_claims:
                        reviews = fact_claims[0].get("claimReview", [])
                        if reviews:
                            top = reviews[0]
                            results[claim] = {
                                "ruling": top.get(
                                    "textualRating", "Unknown"
                                ),
                                "url": top.get("url"),
                                "publisher": (
                                    top.get("publisher", {})
                                    .get("name", "Unknown")
                                ),
                            }
                        else:
                            results[claim] = {
                                "ruling": "No match",
                                "url": None,
                            }
                    else:
                        results[claim] = {
                            "ruling": "No match",
                            "url": None,
                        }
                    break

                except Exception:
                    if attempt == 0:
                        await asyncio.sleep(2)
                    else:
                        results[claim] = {
                            "ruling": "API Error",
                            "url": None,
                        }

    return results


# ── Ruling normalisation ──────────────────────────────────────────────────────


def ruling_to_score(ruling: str) -> Optional[float]:
    """
    Convert a fact-check ruling string to a normalised score (0.0–1.0).

    Returns None for API Error, Unverified, No match — excluded from
    mean so missing data does not drag scores to neutral.
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
    if any(x in r for x in ("false", "pants", "incorrect", "wrong", "fabricated")):
        return 0.0

    return 0.5


# ── Credibility scoring ───────────────────────────────────────────────────────


async def compute_credibility_score(
    article: ArticleResponse,
) -> Dict[str, Any]:
    """
    Compute a credibility score (0–100) for a given article.

    Scoring formula (v2 — guaranteed score for every article):

      base  = _SOURCE_REPUTATION.get(source, 65)
      adj   = (mean valid fact-check scores - 0.5) * 40
      score = clamp(base + adj, 10, 100)

    adj ranges from -20 (all false) to +20 (all true), 0 when no
    fact-checks are found. This means:
    - Every article gets a meaningful base score from source reputation
    - Google fact-check results shift the score when available
    - Irish/regional sources without fact-check coverage still score
      correctly based on their editorial standards

    Falls back to 65.0 neutral default when no claims could be
    extracted and source is unknown.
    """
    source = getattr(article, "source", None) or ""
    base_score = _SOURCE_REPUTATION.get(source, 65.0)

    claims = extract_claims(
        title=article.title or "",
        content=article.content or "",
    )

    if not claims:
        return {
            "score": float(base_score),
            "reason": (
                f"No claims extracted. "
                f"Score based on source reputation ({source or 'unknown'})."
            ),
            "fact_checks": {},
            "claims_checked": 0,
        }

    fact_checks = await check_google_factclaims(claims)

    # Exclude None (API Error / No match / Unverified) from mean.
    scores = [
        s for s in
        (ruling_to_score(r["ruling"]) for r in fact_checks.values())
        if s is not None
    ]

    # Adjustment: ±20 points based on fact-check results.
    # Zero adjustment when no valid fact-checks found.
    if scores:
        fact_mean = sum(scores) / len(scores)
        adjustment = (fact_mean - 0.5) * 40
    else:
        adjustment = 0.0

    score = round(max(10.0, min(100.0, base_score + adjustment)), 2)

    positive = sum(1 for s in scores if s >= 0.7)
    negative = sum(1 for s in scores if s < 0.3)
    total = len(claims)

    if score >= 70:
        verdict = "reliable"
    elif score >= 40:
        verdict = "mixed"
    else:
        verdict = "low credibility"

    if scores:
        reason = (
            f"{positive}/{total} claims verified. "
            f"Source: {source or 'unknown'}. Rated {verdict}."
        )
        if negative > 0:
            reason = (
                f"{positive}/{total} claims verified true, "
                f"{negative} false. "
                f"Source: {source or 'unknown'}. Rated {verdict}."
            )
    else:
        reason = (
            f"No fact-checks found for claims. "
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
    Selects source field so compute_credibility_score can apply
    source reputation scoring to every article.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat()

    recent: Optional[list] = (
        supabase.table("articles")
        .select("id, title, content, description, source")
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
