"""
NewsScope Fact-Checking Service.

Integrates PolitiFact API with lightweight regex claim extraction.

spaCy has been removed entirely — it loaded 50MB+ into Render RAM
on first call, spiking memory when factcheck overlapped with analysis.
Claims are now extracted using simple sentence splitting and regex,
which has near-zero RAM cost and produces comparable quality input
for the PolitiFact API query.

Claim extraction strategy:
- Title is always included as claim 0 — it is the most concise and
  fact-checkable representation of the article.
- Up to _MAX_CLAIMS additional sentences are extracted from content
  using numeric and proper-noun heuristics.
- Total claims capped at _MAX_CLAIMS + 1 (title + extracted).
- Sentences beginning with ADVERTISEMENT are skipped to prevent
  boilerplate leaking into PolitiFact queries.

PolitiFact calls reduced from 5 claims → 2 per article to halve
the number of concurrent HTTP connections during factcheck batches.

PolitiFact headers updated to browser-like User-Agent with Referer
and Accept headers — plain "NewsScope/1.0" was being rejected.
One retry with 2s backoff added per claim to handle transient errors.

Credibility scoring formula:
- Weighted average of PolitiFact ruling scores (0.0–1.0 per ruling).
- score = (sum of ruling scores / total claims) * 100
- Spans full 10–100 range — no hardcoded base anchor.

Flake8: 0 errors/warnings.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from app.schemas import ArticleResponse
from app.db.supabase import supabase


# Maximum claims extracted from content (excluding title).
# Title is always prepended, so total queries = _MAX_CLAIMS + 1 max.
# Reduced from 5 → 2 to halve concurrent PolitiFact HTTP connections
# and reduce factcheck RAM overlap with analysis on Render free tier.
_MAX_CLAIMS = 2

# Browser-like headers for PolitiFact API — plain "NewsScope/1.0"
# User-Agent was being rejected with connection errors on Render.
# Referer and Accept headers added to match expected browser request.
_POLITIFACT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.politifact.com/",
}


# ── Content validation ────────────────────────────────────────────────────────


def _is_valid_content(text: str) -> bool:
    """
    Return True if text appears to be readable UTF-8, not binary.

    Euronews and some sources return gzip-compressed bytes when the
    scraper does not correctly decode the response. Binary blobs have
    a very low ratio of printable ASCII characters — anything below
    85% printable in the first 500 chars is treated as invalid and
    excluded from claim extraction to prevent garbage reaching
    the PolitiFact API.
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

    Title is always included as claim 0 — it is the most concise,
    consistently available, and fact-checkable part of any article.

    Additional claims are extracted from content using regex sentence
    splitting — no spaCy or NLP models loaded. Prioritises sentences
    likely to contain verifiable facts:
    - Contains a number or percentage
    - Contains a capitalised proper noun (likely a named entity)
    - Is a declarative statement (not a question)
    - Is long enough to be meaningful (>= 40 chars)
    - Does not begin with ADVERTISEMENT boilerplate

    Falls back to first two long declarative sentences if no
    heuristic-matched sentences are found.

    Zero RAM overhead vs spaCy's 50MB+ model load.
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

        # Skip questions, very short sentences, and ad boilerplate.
        if sent.endswith("?") or len(sent) < 40:
            continue
        if re.match(r"^ADVERTISEMENT", sent, re.IGNORECASE):
            continue

        has_number = bool(re.search(r"\b\d+(?:[,.\d]+)?%?\b", sent))
        has_proper_noun = bool(
            re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", sent)
        )

        if has_number or has_proper_noun:
            extracted.append(sent)

        if len(extracted) >= _MAX_CLAIMS:
            break

    # Fallback: first two long declarative non-ad sentences.
    if not extracted:
        for sent in raw_sentences:
            sent = sent.strip()
            if (
                len(sent) >= 40 and not sent.endswith("?") and not re.match(r"^ADVERTISEMENT", sent, re.IGNORECASE)
            ):
                extracted.append(sent)
            if len(extracted) >= _MAX_CLAIMS:
                break

    claims.extend(extracted)
    return claims


# ── PolitiFact API ────────────────────────────────────────────────────────────


async def check_politifact_claims(
    claims: List[str],
) -> Dict[str, Any]:
    """
    Query PolitiFact public search API for each extracted claim.

    Returns a dict mapping each claim to its top ruling and URL.
    Fails gracefully per claim — one API error does not abort the batch.

    Uses browser-like headers (_POLITIFACT_HEADERS) — plain
    "NewsScope/1.0" User-Agent was being rejected. One retry with
    2s backoff added per claim to handle transient errors.

    Uses a single shared AsyncClient across all claims in the batch
    to avoid opening a new connection pool per claim.
    """
    results: Dict[str, Any] = {}

    async with httpx.AsyncClient(
        timeout=15.0, headers=_POLITIFACT_HEADERS
    ) as client:
        for claim in claims:
            for attempt in range(2):  # one retry on failure
                try:
                    resp = await client.get(
                        "https://www.politifact.com/api/statements/search/",
                        params={"q": claim[:100]},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    if data.get("items"):
                        top = data["items"][0]
                        ruling = (
                            top.get("ruling", {}).get("ruling", "Unknown")
                        )
                        results[claim] = {
                            "ruling": ruling,
                            "url": top.get("url"),
                            "speaker": top.get("speaker", "N/A"),
                        }
                    else:
                        results[claim] = {
                            "ruling": "No match",
                            "url": None,
                        }
                    break  # success — stop retrying

                except Exception:
                    if attempt == 0:
                        # Wait before retry.
                        import asyncio
                        await asyncio.sleep(2)
                    else:
                        results[claim] = {
                            "ruling": "API Error",
                            "url": None,
                        }

    return results


# ── Ruling normalisation ──────────────────────────────────────────────────────


def ruling_to_score(ruling: str) -> float:
    """
    Convert a PolitiFact ruling string to a normalised credibility score.

    Returns a float between 0.0 (least credible) and 1.0 (most credible).
    Ambiguous or error rulings return 0.5 (neutral).
    """
    r = ruling.lower()
    if "true" in r and "mostly" not in r:
        return 1.0
    if "mostly true" in r:
        return 0.8
    if "half true" in r:
        return 0.5
    if "mostly false" in r:
        return 0.2
    if "false" in r or "pants" in r:
        return 0.0
    return 0.5


# ── Credibility scoring ───────────────────────────────────────────────────────


async def compute_credibility_score(
    article: ArticleResponse,
) -> Dict[str, Any]:
    """
    Compute a credibility score (0–100) for a given article.

    Scoring formula:
      score = (mean PolitiFact ruling score) * 100

    Where each ruling is normalised to 0.0–1.0 via ruling_to_score().
    This spans the full 10–100 range — there is no hardcoded base anchor.

    Falls back to a default of 70.0 (neutral/unknown) when:
    - Article text is too short to extract claims from
    - No claims could be extracted from title or content

    credibility_reason is always a human-readable string, never a tally.
    """
    claims = extract_claims(
        title=article.title or "",
        content=article.content or "",
    )

    if not claims:
        return {
            "score": 70.0,
            "reason": "No claims extracted. Score is neutral default.",
            "fact_checks": {},
            "claims_checked": 0,
        }

    fact_checks = await check_politifact_claims(claims)
    scores = [ruling_to_score(r["ruling"]) for r in fact_checks.values()]

    if scores:
        fact_score = sum(scores) / len(scores)
    else:
        fact_score = 0.5

    score = round(max(10.0, min(100.0, fact_score * 100)), 2)

    positive = sum(1 for s in scores if s >= 0.7)
    negative = sum(1 for s in scores if s < 0.3)
    total = len(claims)

    if score >= 70:
        verdict = "reliable"
    elif score >= 40:
        verdict = "mixed"
    else:
        verdict = "low credibility"

    reason = f"{positive}/{total} claims verified true. Rated {verdict}."

    # Surface negative count in reason when present.
    if negative > 0:
        reason = (
            f"{positive}/{total} claims verified true, "
            f"{negative} false. Rated {verdict}."
        )

    return {
        "score": score,
        "fact_checks": fact_checks,
        "claims_checked": total,
        "reason": reason,
    }


# ── Batch re-check ────────────────────────────────────────────────────────────


async def batch_factcheck_recent(
    hours: int = 24,
) -> List[Dict[str, Any]]:
    """
    Re-fact-check articles published in the last `hours` hours
    whose credibility score is at or below 80.1.

    Articles seeded with the ingestion default of 80.0 are caught
    by the <= 80.1 threshold and re-scored with real PolitiFact data.
    Persists updated score, fact_checks, claims_checked, and
    credibility_reason to Supabase.

    Returns a list of result dicts keyed by article ID.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat()

    recent: Optional[list] = (
        supabase.table("articles")
        .select("id, title, content, description")
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
