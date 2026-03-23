"""
NewsScope FastAPI application entry point.

SCHEDULING OVERVIEW (v6 — chain-first, cron-as-safety-net):

  PIPELINE (chained, fires automatically each cycle):
    :00  Ingestion starts   (CronTrigger minute=0)
         Categorisation is inline — happens per-article during scraping.
         Completes in ~5 minutes on 512MB Render free tier.
         _heavy_job_lock held for entire duration.
         ↓ on completion
         Analysis fires immediately — no fixed wait, no wasted minutes.
         Completes in ~10-15 minutes (200 articles × 8 concurrent HF calls).
         ↓ on completion
         Pipeline done. Total wall time: ~15-20 minutes per hour.

  SAFETY NETS (cron — only fire if chained call failed/was skipped):
    :20  Analysis safety net  (CronTrigger minute=20)
         Catches articles that slipped through if ingestion chained
         analysis failed. analyze_unscored_articles() is idempotent —
         only picks up articles with sentiment_score IS NULL, so it
         never double-scores articles the chain already handled.

    :45  Fact-check           (CronTrigger minute=45)
         Pure async I/O (Google API). No heavy job lock needed.
         Runs after analysis is reliably complete (chained ~15-20 min,
         safety net at :20 adds another buffer). Was :40, moved to :45
         to give analysis 5 extra minutes on busy cycles.

    03:00 Archiving           (CronTrigger hour=3)
          Once daily. Lightweight DB-only operation.

WHY CHAINED NOT CRON FOR ANALYSIS?
  The previous CronTrigger at :20 was a timing guess — if ingestion
  somehow ran slow, analysis fired while scraping was still mid-chunk.
  The lock blocked it, causing silent skips and delayed scoring.
  Chaining guarantees analysis always starts the instant ingestion
  finishes, regardless of how long ingestion took. The :20 cron stays
  purely as a safety net for failure recovery.

MEMORY MODEL (512MB Render free tier):
  - Ingestion peak: 4 workers × ~15MB DOM = ~60MB per chunk of 5.
    Total ingestion footprint: ~100MB above baseline.
  - Analysis peak: 8 concurrent HF response buffers ≈ negligible.
    Real cost is the Supabase row fetch — capped to 48h × 200 rows.
  - Fact-check: httpx async client, one open connection. Negligible.
  - Combined baseline (FastAPI + Supabase + APScheduler): ~80MB.
  - Headroom on 512MB: ~280MB — sufficient for chained pipeline.
  - Analysis starts after ingestion RAM is freed (gc.collect() in
    run_ingestion_cycle finalizer), so no memory overlap.

Flake8: 0 errors/warnings.
"""

import asyncio
import json
import os
import threading
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import firebase_admin
from apscheduler.triggers.cron import CronTrigger
from dateutil import parser as dtparser
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from firebase_admin import auth, credentials

from app.core.scheduler import start_scheduler
from app.db.supabase import supabase
from app.jobs.analysis import analyze_unscored_articles
from app.jobs.archiving import archive_old_articles
from app.jobs.fact_checking import batch_factcheck_recent
from app.jobs.ingestion import run_ingestion_cycle, set_main_event_loop
from app.jobs.keep_alive import start_keep_alive
from app.routes import articles, sources, users
from app.schemas import (
    BiasProfile,
    ComparisonRequest,
    ComparisonResponse,
    ReadingHistoryCreate,
)


# ── Main event loop — captured in lifespan for sync bridges ──────────────────
_main_loop: Optional[asyncio.AbstractEventLoop] = None

# ── Job guards ────────────────────────────────────────────────────────────────
# _heavy_job_lock: atomic threading.Lock — prevents any two heavy jobs
# (ingestion, analysis) from running simultaneously.
# acquire(blocking=False) is the only safe pattern — boolean flags
# had a check-then-act race between APScheduler thread pool workers.
_ingestion_running = False
_analysis_running = False
_heavy_job_lock = threading.Lock()

# Redeploy guard: skip startup ingestion if DB updated < 30 min ago.
_REDEPLOY_GUARD_MINUTES = 30


def init_firebase() -> None:
    try:
        service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        if service_account:
            cred_dict = json.loads(service_account)
            cred = credentials.Certificate(cred_dict)
        else:
            cred = credentials.Certificate("firebase-service-account.json")
        firebase_admin.initialize_app(cred)
        print("Firebase Admin initialized")
    except ValueError:
        print("Firebase Admin already initialized")
    except Exception as exc:
        print(f"Firebase Admin init failed: {exc}")


init_firebase()


# ── Sync bridges for APScheduler ──────────────────────────────────────────────


def _sync_ingestion() -> None:
    """
    Sync bridge — runs _scheduled_ingestion on the main event loop.
    Fires at :00 via CronTrigger. After ingestion completes, analysis
    is chained immediately inside _scheduled_ingestion.
    Lock is held for full ingestion + analysis duration.
    """
    if not _heavy_job_lock.acquire(blocking=False):
        print("Heavy job in progress — skipping scheduled ingestion.")
        return
    if _main_loop is None or not _main_loop.is_running():
        _heavy_job_lock.release()
        print("Ingestion bridge: no running loop, skipping.")
        return
    future = asyncio.run_coroutine_threadsafe(
        _scheduled_ingestion(), _main_loop
    )
    try:
        future.result(timeout=3600)
    except Exception as exc:
        print(f"Scheduled ingestion error: {exc}")
    finally:
        _heavy_job_lock.release()


def _sync_analysis_safety_net() -> None:
    """
    Sync bridge — analysis safety net at :20.
    Only does real work if chained analysis (inside _scheduled_ingestion)
    failed or was skipped. analyze_unscored_articles() is idempotent —
    it only picks up articles with sentiment_score IS NULL.
    """
    if not _heavy_job_lock.acquire(blocking=False):
        print("Heavy job in progress — skipping analysis safety net.")
        return
    if _main_loop is None or not _main_loop.is_running():
        _heavy_job_lock.release()
        print("Analysis safety net: no running loop, skipping.")
        return
    future = asyncio.run_coroutine_threadsafe(
        _guarded_analysis(), _main_loop
    )
    try:
        future.result(timeout=600)
    except Exception as exc:
        print(f"Analysis safety net error: {exc}")
    finally:
        _heavy_job_lock.release()


def _sync_factcheck() -> None:
    """
    Sync bridge — runs batch_factcheck_recent on the main event loop.
    Fires at :45 via CronTrigger — after chained ingestion+analysis
    (~15-20 min) and the :20 analysis safety net.
    No heavy job lock — fact-checking is pure I/O (Google API calls).
    """
    if _main_loop is None or not _main_loop.is_running():
        print("Fact-check bridge: no running loop, skipping.")
        return
    future = asyncio.run_coroutine_threadsafe(
        batch_factcheck_recent(hours=48), _main_loop
    )
    try:
        future.result(timeout=1800)
    except Exception as exc:
        print(f"Scheduled fact-check error: {exc}")


def _sync_archive() -> None:
    """
    Sync bridge — runs archive_old_articles on the main event loop.
    No heavy job lock — archiving only reads/deletes old DB rows.
    """
    if _main_loop is None or not _main_loop.is_running():
        print("Archive bridge: no running loop, skipping.")
        return
    future = asyncio.run_coroutine_threadsafe(
        asyncio.to_thread(archive_old_articles), _main_loop
    )
    try:
        future.result(timeout=1800)
    except Exception as exc:
        print(f"Scheduled archiving error: {exc}")


# ── Async job wrappers ────────────────────────────────────────────────────────


async def _scheduled_ingestion() -> None:
    """
    Run ingestion, then immediately chain analysis on completion.

    This is the core of the pipeline — categorisation is inline during
    scraping (infer_category called per article in _scrape_one), so by
    the time ingestion finishes every article already has a category.
    Analysis fires the instant ingestion returns, with no fixed delay.

    The :20 cron safety net handles the case where this chain fails.
    """
    global _ingestion_running
    if _ingestion_running:
        print("Ingestion already running — skipping duplicate trigger.")
        return
    _ingestion_running = True
    try:
        print("Pipeline: starting ingestion...")
        await asyncio.to_thread(run_ingestion_cycle)
        print("Pipeline: ingestion complete — chaining analysis immediately.")
    except Exception as exc:
        print(f"Pipeline: ingestion failed: {exc}")
        return
    finally:
        _ingestion_running = False

    # Chain: analysis fires immediately after ingestion, no sleep needed.
    # gc.collect() already called at end of run_ingestion_cycle, so
    # ingestion RAM is freed before analysis begins.
    await _guarded_analysis()


async def _guarded_analysis() -> None:
    global _analysis_running
    if _analysis_running:
        print("Analysis already running — skipping duplicate trigger.")
        return
    _analysis_running = True
    try:
        await analyze_unscored_articles()
    finally:
        _analysis_running = False


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Scheduled job timing:
      :00  ingestion → chains analysis immediately on completion
      :20  analysis safety net — idempotent, only scores NULL articles
      :45  fact-check — pure I/O, no lock needed (moved from :40)
      03:00 archiving — once daily, lightweight

    On startup: ingestion runs once immediately (with redeploy guard),
    then analysis chains off it. Subsequent runs follow the cron schedule.
    """
    global _main_loop

    from app.core.scheduler import scheduler

    _main_loop = asyncio.get_event_loop()
    set_main_event_loop(_main_loop)

    start_scheduler()

    # Ingestion at top of every hour — chains analysis on completion.
    scheduler.add_job(
        _sync_ingestion,
        trigger=CronTrigger(minute=0),
        id="ingestion",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

    # Analysis safety net at :20 — only does work if chain failed.
    # analyze_unscored_articles() is idempotent (NULL filter).
    scheduler.add_job(
        _sync_analysis_safety_net,
        trigger=CronTrigger(minute=20),
        id="analysis_safety_net",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

    # Fact-check at :45 — after chained pipeline (~15-20 min) is done.
    # Pure I/O, no heavy job lock. Moved from :40 → :45 for extra buffer.
    scheduler.add_job(
        _sync_factcheck,
        trigger=CronTrigger(minute=45),
        id="factcheck",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

    # Archiving: 3am daily — lightweight, once per day is sufficient.
    scheduler.add_job(
        _sync_archive,
        trigger=CronTrigger(hour=3, minute=0),
        id="archiving",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

    env = os.getenv("ENVIRONMENT", "production")
    print(f"Environment: {env}")

    if env == "production":
        start_keep_alive()
        print("Keep-alive enabled (pings every 14 minutes)")
    else:
        print("Keep-alive disabled in development")

    asyncio.create_task(_run_startup_sequence())

    yield
    print("Server shutting down...")


# ── Startup helpers ───────────────────────────────────────────────────────────


def _ingestion_ran_recently() -> bool:
    try:
        data = (
            supabase.table("articles")
            .select("updated_at")
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
            .data
        )
        if not data:
            return False

        last_update = dtparser.parse(data[0]["updated_at"])
        if last_update.tzinfo is None:
            last_update = last_update.replace(tzinfo=timezone.utc)

        age_minutes = (
            datetime.now(timezone.utc) - last_update
        ).total_seconds() / 60

        if age_minutes < _REDEPLOY_GUARD_MINUTES:
            print(
                f"Redeploy guard: last ingestion {age_minutes:.1f} min ago "
                "— skipping startup ingestion to prevent double-OOM."
            )
            return True

    except Exception as exc:
        print(f"Redeploy guard check failed: {exc} — proceeding with ingestion.")

    return False


async def _run_startup_sequence() -> None:
    """
    Startup sequence: ingest → chain analysis immediately → done.

    If redeploy guard fires (ingestion ran < 30 min ago), skip ingestion
    and run analysis only — articles are already in DB, just need scoring.
    Subsequent runs follow the hourly cron schedule.
    """
    global _ingestion_running

    if _ingestion_ran_recently():
        print(
            "Startup ingestion skipped (redeploy guard). "
            "Running analysis on existing unscored articles..."
        )
        if _heavy_job_lock.acquire(blocking=False):
            try:
                await _guarded_analysis()
            finally:
                _heavy_job_lock.release()
        else:
            print("Lock held at startup analysis — skipping.")
        return

    print("Server startup: running ingestion pipeline...")
    if not _heavy_job_lock.acquire(blocking=False):
        print("Lock already held at startup — skipping.")
        return

    # _scheduled_ingestion handles the full chain: ingest → analyse.
    try:
        await _scheduled_ingestion()
        print("Startup pipeline complete (ingestion + analysis).")
    except Exception as exc:
        print(f"Startup pipeline failed: {exc}")
    finally:
        _heavy_job_lock.release()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="NewsScope API", version="1.0.0", lifespan=lifespan)


# ── Auth ──────────────────────────────────────────────────────────────────────


def get_current_user(
    authorization: Optional[str] = Header(None),
) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.replace("Bearer ", "")
    try:
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token["uid"]
        print(f"Authenticated user: {user_id}")
        return user_id
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    except auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Authentication token expired")
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {exc}")


# ── Core routes ───────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {
        "message": "Welcome to NewsScope API. Try /health to check status.",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.head("/")
def root_head():
    return {}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Internal + debug routes ───────────────────────────────────────────────────


@app.post("/internal/analyze-batch")
async def internal_analyze_batch(background_tasks: BackgroundTasks):
    if _analysis_running:
        return {"status": "skipped — analysis already running"}
    background_tasks.add_task(_guarded_analysis)
    return {"status": "ok"}


@app.post("/debug/ingest")
async def debug_ingest(background_tasks: BackgroundTasks):
    background_tasks.add_task(_scheduled_ingestion)
    return {"status": "ingestion triggered in background"}


@app.post("/debug/analyze")
async def debug_analyze(background_tasks: BackgroundTasks):
    background_tasks.add_task(_guarded_analysis)
    return {"status": "analysis triggered in background"}


@app.post("/debug/factcheck")
async def debug_factcheck(background_tasks: BackgroundTasks):
    background_tasks.add_task(batch_factcheck_recent, 48)
    return {"status": "fact-check triggered in background"}


@app.post("/debug/archive")
async def debug_archive(background_tasks: BackgroundTasks):
    background_tasks.add_task(archive_old_articles)
    return {"status": "archiving triggered in background"}


# ── User API ──────────────────────────────────────────────────────────────────


@app.post("/api/reading-history")
async def track_reading(
    data: ReadingHistoryCreate,
    user_id: str = Depends(get_current_user),
):
    """
    Track article reading time and snapshot scores for bias profile.
    Scores copied from article row at read time — stable after archiving.
    """
    try:
        article_data = (
            supabase.table("articles")
            .select("bias_score, sentiment_score, source, general_bias")
            .eq("id", data.article_id)
            .limit(1)
            .execute()
            .data
        )
        scores = article_data[0] if article_data else {}

        payload = {
            "user_id": user_id,
            "article_id": data.article_id,
            "time_spent_seconds": data.time_spent_seconds,
            "opened_at": datetime.utcnow().isoformat(),
            "bias_score": scores.get("bias_score"),
            "sentiment_score": scores.get("sentiment_score"),
            "source": scores.get("source"),
            "general_bias": scores.get("general_bias"),
        }
        response = supabase.table("reading_history").upsert(
            payload, on_conflict="user_id,article_id"
        ).execute()
        print(f"Reading tracked: {data.article_id} ({data.time_spent_seconds}s)")
        return {"success": True, "data": response.data}
    except Exception as exc:
        print(f"Error tracking reading: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/bias-profile", response_model=BiasProfile)
async def get_bias_profile(user_id: str = Depends(get_current_user)):
    """
    Calculate user's bias profile from reading history snapshot columns.
    Reads directly from reading_history — stable after article archiving.
    """
    try:
        response = (
            supabase.table("reading_history")
            .select("time_spent_seconds, bias_score, sentiment_score, source")
            .eq("user_id", user_id)
            .execute()
        )
        history = response.data

        if not history:
            return BiasProfile(
                avg_bias=0.0,
                avg_sentiment=0.0,
                total_articles_read=0,
                left_count=0,
                center_count=0,
                right_count=0,
                most_read_source="N/A",
                bias_distribution={"left": 0.0, "center": 0.0, "right": 0.0},
                reading_time_total_minutes=0,
                positive_count=0,
                neutral_count=0,
                negative_count=0,
                source_breakdown={},
            )

        total_time = sum(h["time_spent_seconds"] for h in history)

        bias_terms = [
            (h["time_spent_seconds"], h["bias_score"])
            for h in history
            if h.get("bias_score") is not None
        ]
        sent_terms = [
            (h["time_spent_seconds"], h["sentiment_score"])
            for h in history
            if h.get("sentiment_score") is not None
        ]

        weighted_bias = (
            sum(t * b for t, b in bias_terms) / sum(t for t, _ in bias_terms)
            if bias_terms else 0.0
        )
        weighted_sentiment = (
            sum(t * s for t, s in sent_terms) / sum(t for t, _ in sent_terms)
            if sent_terms else 0.0
        )

        left = sum(
            1 for h in history
            if h.get("bias_score") is not None and h["bias_score"] < -0.3
        )
        right = sum(
            1 for h in history
            if h.get("bias_score") is not None and h["bias_score"] > 0.3
        )
        center = len(history) - left - right

        pos = sum(
            1 for h in history
            if h.get("sentiment_score") is not None and h["sentiment_score"] > 0.3
        )
        neg = sum(
            1 for h in history
            if h.get("sentiment_score") is not None and h["sentiment_score"] < -0.3
        )
        neu = len(history) - pos - neg

        sources = [h["source"] for h in history if h.get("source")]
        source_counter = Counter(sources)
        most_read = source_counter.most_common(1)[0][0] if source_counter else "N/A"
        source_breakdown = dict(source_counter.most_common(12))

        total_reads = len(history)
        distribution = {
            "left": round(left / total_reads * 100, 1) if total_reads else 0.0,
            "center": round(center / total_reads * 100, 1) if total_reads else 0.0,
            "right": round(right / total_reads * 100, 1) if total_reads else 0.0,
        }

        return BiasProfile(
            avg_bias=round(weighted_bias, 3),
            avg_sentiment=round(weighted_sentiment, 3),
            total_articles_read=total_reads,
            left_count=left,
            center_count=center,
            right_count=right,
            most_read_source=most_read,
            bias_distribution=distribution,
            reading_time_total_minutes=round(total_time / 60),
            positive_count=pos,
            neutral_count=neu,
            negative_count=neg,
            source_breakdown=source_breakdown,
        )
    except Exception as exc:
        print(f"Error fetching bias profile: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/articles/compare", response_model=ComparisonResponse)
async def compare_articles(request: ComparisonRequest):
    try:
        query = (
            supabase.table("articles")
            .select("*")
            .not_.is_("bias_score", "null")
            .order("published_at", desc=True)
        )
        if request.topic:
            query = query.or_(
                f"title.ilike.%{request.topic}%,"
                f"content.ilike.%{request.topic}%"
            )
        if request.category:
            query = query.eq("category", request.category)
        if request.source:
            query = query.eq("source", request.source)

        articles_data = query.limit(10000).execute().data

        left = [a for a in articles_data if a.get("bias_score", 0) < -0.3]
        center = [a for a in articles_data if -0.3 <= a.get("bias_score", 0) <= 0.3]
        right = [a for a in articles_data if a.get("bias_score", 0) > 0.3]

        return ComparisonResponse(
            topic=request.topic or "",
            left_articles=left,
            center_articles=center,
            right_articles=right,
            total_found=len(articles_data),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(articles.router, prefix="/articles", tags=["articles"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(sources.router, prefix="/sources", tags=["sources"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/favicon.ico")
async def favicon():
    file_path = os.path.join(BASE_DIR, "..", "static", "favicon.ico")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"status": "no icon"}
