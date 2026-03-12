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
- Sentences matching _BOILERPLATE_RE are skipped to prevent Fox News
  and other site chrome leaking into PolitiFact queries.

PolitiFact calls reduced from 5 claims → 2 per article to halve
the number of concurrent HTTP connections during factcheck batches.

PolitiFact headers updated to browser-like User-Agent with Referer
and Accept headers — plain "NewsScope/1.0" was being rejected.
One retry with 2s backoff added per claim to handle transient errors.

Credibility scoring formula:
- Weighted average of valid PolitiFact ruling scores (0.0–1.0).
- API Error and No match rulings are excluded from the mean —
  they are not false claims, just unverifiable.
- score = (sum of valid scores / valid count) * 100
- Falls back to 50.0 neutral when all claims errored.
- Falls back to 70.0 default when no claims could be extracted.

Flake8: 0 errors/warnings.
"""

import asyncio
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

# Boilerplate patterns to skip during claim extraction.
# Covers Fox News UI chrome (NEW, close, Video), generic ad copy,
# photo captions (Photograph:, Getty Images) and common site furniture
# that leaks into scraped content from NewsAPI and RSS sources.
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
    - Does not match _BOILERPLATE_RE (Fox News chrome, ad copy, etc.)

    Falls back to first two long declarative non-boilerplate sentences
    if no heuristic-matched sentences are found.

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

        if sent.endswith("?") or len(sent) < 40:
            continue
        if _BOILERPLATE_RE.match(sent):
            continue

        has_number = bool(
            re.search(r"\b\d+(?:[,.\d]+)?%?\b", sent)
        )
        has_proper_noun = bool(
            re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", sent)
        )

        if has_number or has_proper_noun:
            extracted.append(sent)

        if len(extracted) >= _MAX_CLAIMS:
            break

    # Fallback: first two long declarative non-boilerplate sentences.
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
            for attempt in range(2):
                try:
                    resp = await client.get(
                        "https://www.politifact.com/api/statements/"
                        "search/",
                        params={"q": claim[:100]},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    if data.get("items"):
                        top = data["items"][0]
                        ruling = (
                            top.get("ruling", {})
                            .get("ruling", "Unknown")
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
    Convert a PolitiFact ruling string to a normalised credibility score.

    Returns float 0.0–1.0 for known rulings.
    Returns None for API Error and No match — these are excluded from
    the credibility mean so they do not drag all scores to 50.0 when
    PolitiFact is unreachable or returns no matching fact-check.
    """
    r = ruling.lower()
    if "error" in r or "no match" in r:
        return None
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
      score = (mean of valid PolitiFact ruling scores) * 100

    API Error and No match rulings return None from ruling_to_score()
    and are excluded from the mean — they are not counted as false
    claims, just unverifiable. Falls back to 50.0 neutral when all
    claims errored or no PolitiFact match was found.

    Falls back to 70.0 default when no claims could be extracted.
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

    # Exclude None (API Error / No match) from mean calculation.
    scores = [
        s for s in
        (ruling_to_score(r["ruling"]) for r in fact_checks.values())
        if s is not None
    ]

    fact_score = (sum(scores) / len(scores)) if scores else 0.5
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
