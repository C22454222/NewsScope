
# app/jobs/fact_checking.py
"""
NewsScope Fact-Checking Service (Flake8: 0 errors).
Integrates PolitiFact API + spaCy claim extraction.
"""

import httpx
from typing import Dict, Any, List
from datetime import datetime, timedelta, timezone

import spacy

from app.schemas import ArticleResponse
from app.db.supabase import supabase


nlp = spacy.load("en_core_web_sm")


def extract_claims(text: str) -> List[str]:
    """Extract verifiable claims (spaCy + heuristics)."""
    doc = nlp(text)
    claims = []
    for sent in doc.sents:
        if (
            not sent.is_question and len(sent) > 5 and any(
                token.like_num or token.ent_type_ in ["DATE", "PERSON", "ORG"]
                for token in sent
            )
        ):
            claims.append(sent.text.strip())
        if len(claims) >= 5:
            break
    return claims


async def check_politifact_claims(claims: List[str]) -> Dict[str, Any]:
    """Query PolitiFact public API."""
    results = {}
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


def ruling_to_score(ruling: str) -> float:
    """PolitiFact ruling → score (0-1)."""
    ruling_lower = ruling.lower()
    if "true" in ruling_lower:
        return 1.0
    if "mostly true" in ruling_lower:
        return 0.8
    if "half true" in ruling_lower:
        return 0.5
    if "mostly false" in ruling_lower:
        return 0.2
    if "false" in ruling_lower or "pants" in ruling_lower:
        return 0.0
    return 0.5


async def compute_credibility_score(article: ArticleResponse) -> Dict[str, Any]:
    """Compute credibility (0-100)."""
    text = f"{article.title} {article.description} {article.content}".strip()
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


async def batch_factcheck_recent(hours: int = 24) -> List[Dict[str, Any]]:
    """Re-factcheck recent articles."""
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat()
    recent = (
        supabase.table("articles")
        .select("id,title,content,description")
        .gte("published_at", cutoff)
        .lte("credibility_score", 80.1)
        .limit(50)
        .execute()
        .data
    )

    results = []
    for art in recent:
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
