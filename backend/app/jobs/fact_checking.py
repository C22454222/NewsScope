"""
NewsScope Fact-Checking Service.

Integrates PolitiFact API with spaCy claim extraction.
spaCy is lazy-loaded on first use — not imported at module level —
to avoid loading 50MB+ into Render RAM on startup.

Flake8: 0 errors/warnings.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from app.schemas import ArticleResponse
from app.db.supabase import supabase


# ── spaCy lazy loader ─────────────────────────────────────────────────────────

_nlp = None


def _get_nlp():
    """
    Lazy-load spaCy en_core_web_sm on first call only.
    Prevents 50MB model from loading into Render RAM at startup.
    """
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ── Claim extraction ──────────────────────────────────────────────────────────


def extract_claims(text: str) -> List[str]:
    """
    Extract up to 5 verifiable claims from article text using spaCy.
    Filters out questions and very short sentences. Prioritises sentences
    containing named entities (dates, people, organisations) or numbers.
    """
    doc = _get_nlp()(text)
    claims: List[str] = []

    for sent in doc.sents:
        sent_text = sent.text.strip()

        if sent_text.endswith("?"):
            continue

        if len(sent) <= 5:
            continue

        has_entity = any(
            token.like_num or token.ent_type_ in ("DATE", "PERSON", "ORG")
            for token in sent
        )
        if has_entity:
            claims.append(sent_text)

        if len(claims) >= 5:
            break

    return claims


# ── PolitiFact API ────────────────────────────────────────────────────────────


async def check_politifact_claims(
    claims: List[str],
) -> Dict[str, Any]:
    """
    Query PolitiFact public search API for each extracted claim.
    Returns a dict mapping each claim to its top ruling and URL.
    Fails gracefully per claim — one API error does not abort the batch.
    """
    results: Dict[str, Any] = {}

    async with httpx.AsyncClient(timeout=10.0) as client:
        for claim in claims:
            try:
                resp = await client.get(
                    "https://www.politifact.com/api/statements/search/",
                    params={"q": claim[:100]},
                    headers={"User-Agent": "NewsScope/1.0"},
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("items"):
                    top = data["items"][0]
                    ruling = top.get("ruling", {}).get("ruling", "Unknown")
                    results[claim] = {
                        "ruling": ruling,
                        "url": top.get("url"),
                        "speaker": top.get("speaker", "N/A"),
                    }
                else:
                    results[claim] = {"ruling": "No match", "url": None}

            except Exception:
                results[claim] = {"ruling": "API Error", "url": None}

    return results


# ── Ruling normalisation ──────────────────────────────────────────────────────


def ruling_to_score(ruling: str) -> float:
    """
    Convert a PolitiFact ruling string to a normalised credibility score.
    Returns a float between 0.0 (least credible) and 1.0 (most credible).
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
    Compute a credibility score (0-100) for a given article.
    Uses claim extraction + PolitiFact verification as primary signal.
    Falls back to a default of 85.0 for short articles or when no
    verifiable claims are found.
    """
    text = " ".join(
        filter(None, [article.title, article.description, article.content])
    ).strip()

    if len(text) < 100:
        return {
            "score": 85.0,
            "reason": "Short article, credible source",
            "fact_checks": {},
            "claims_checked": 0,
        }

    claims = extract_claims(text)
    if not claims:
        return {
            "score": 85.0,
            "reason": "No claims extracted",
            "fact_checks": {},
            "claims_checked": 0,
        }

    fact_checks = await check_politifact_claims(claims)
    scores = [ruling_to_score(r["ruling"]) for r in fact_checks.values()]

    positive = sum(1 for s in scores if s >= 0.7)
    negative = sum(1 for s in scores if s < 0.3)

    base = 80.0
    bonus = (positive / len(claims)) * 15
    penalty = (negative / len(claims)) * 25
    score = max(10.0, round(base + bonus - penalty, 2))

    return {
        "score": score,
        "fact_checks": fact_checks,
        "claims_checked": len(claims),
        "reason": f"{positive}+/{negative}-/{len(claims)}",
    }


# ── Batch re-check ────────────────────────────────────────────────────────────


async def batch_factcheck_recent(
    hours: int = 24,
) -> List[Dict[str, Any]]:
    """
    Re-fact-check articles published in the last `hours` hours
    whose credibility score is at or below 80.1.
    Persists updated scores, fact_checks, and reason to Supabase.
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
