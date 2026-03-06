"""
NewsScope FastAPI application entry point.

Lifespan manages scheduler startup, job registration, keep-alive,
and a single startup ingestion run. Analysis is deliberately delayed
180 seconds after startup ingestion to prevent simultaneous memory
pressure from both jobs on Render free tier (512MB limit).

Redeploy guard in _run_startup_sequence checks DB for recent ingestion
(< 15 minutes ago) and skips startup ingestion on zero-downtime redeploys
where the old instance already ran it — prevents double-ingestion OOM.

Heavy job lock (_heavy_job_running) prevents ingestion and analysis
from ever running simultaneously — the root cause of ConnectionTerminated
errors and OOM crashes on the Render free tier.

Flake8: 0 errors/warnings.
"""

import asyncio
import json
import os
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

import firebase_admin
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dateutil import parser as dtparser
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from firebase_admin import auth, credentials

from app.core.scheduler import start_scheduler
from app.db.supabase import supabase
from app.jobs.analysis import analyze_unscored_articles
from app.jobs.archiving import archive_old_articles
from app.jobs.ingestion import run_ingestion_cycle, set_main_event_loop
from app.jobs.keep_alive import start_keep_alive
from app.routes import articles, sources, users
from app.schemas import (
    BiasProfile,
    ComparisonRequest,
    ComparisonResponse,
    FactCheck,
    ReadingHistoryCreate,
)


# ── Main event loop — captured in lifespan for sync bridges ──────────────────
_main_loop: Optional[asyncio.AbstractEventLoop] = None

# ── Job guards ────────────────────────────────────────────────────────────────
# _ingestion_running : prevents duplicate ingestion triggers
# _heavy_job_running : shared lock — prevents ingestion + analysis overlap
#                      which causes ConnectionTerminated + OOM on free tier
_ingestion_running = False
_heavy_job_running = False

# ── Startup analysis delay ────────────────────────────────────────────────────
# Increased from 60 → 180 — chunked ingestion takes longer to fully
# flush RAM; 60s was insufficient clearance before analysis fired.
_ANALYSIS_STARTUP_DELAY_SECONDS = 180

# ── Redeploy guard ────────────────────────────────────────────────────────────
# Increased from 10 → 15 to cover slower chunked ingestion cycles.
_REDEPLOY_GUARD_MINUTES = 15


def init_firebase() -> None:
    """Initialise Firebase Admin SDK from env var or local file."""
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
    Skips if any heavy job is already running.
    """
    if _heavy_job_running:
        print("Heavy job in progress — skipping scheduled ingestion.")
        return
    if _main_loop is None or not _main_loop.is_running():
        print("Ingestion bridge: no running loop, skipping.")
        return
    future = asyncio.run_coroutine_threadsafe(
        _scheduled_ingestion(), _main_loop
    )
    try:
        future.result(timeout=3600)
    except Exception as exc:
        print(f"Scheduled ingestion error: {exc}")


def _sync_analysis() -> None:
    """
    Sync bridge — runs _guarded_analysis on the main event loop.
    Fires every 5 minutes. Skips if any heavy job is running.
    """
    if _heavy_job_running:
        print("Heavy job in progress — skipping scheduled analysis.")
        return
    if _main_loop is None or not _main_loop.is_running():
        print("Analysis bridge: no running loop, skipping.")
        return
    future = asyncio.run_coroutine_threadsafe(
        _guarded_analysis(), _main_loop
    )
    try:
        future.result(timeout=600)
    except Exception as exc:
        print(f"Scheduled analysis error: {exc}")


def _sync_archive() -> None:
    """
    Sync bridge — runs archive_old_articles on the main event loop.
    archive_old_articles is synchronous — wrapped in to_thread.
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


# ── Guarded analysis wrapper ──────────────────────────────────────────────────


async def _guarded_analysis() -> None:
    """
    Sets _heavy_job_running around analyze_unscored_articles so that
    ingestion bridges cannot fire concurrently.
    All analysis calls must go through this — never call
    analyze_unscored_articles() directly from scheduled paths.
    """
    global _heavy_job_running

    if _heavy_job_running:
        print("Heavy job already running — skipping analysis trigger.")
        return

    _heavy_job_running = True
    try:
        await analyze_unscored_articles()
    finally:
        _heavy_job_running = False


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Startup order:
      1. Capture main event loop for sync bridges
      2. Start APScheduler with ingestion, analysis, archiving jobs
      3. Start keep-alive pinger (production only)
      4. Run startup sequence in background task:
         a. Check DB — skip ingestion if run < 15 min ago (redeploy guard)
         b. Set _heavy_job_running during ingestion to block analysis
         c. Wait 180s after ingestion, then run first analysis

    Shutdown:
      - Logs shutdown message (scheduler stops automatically)
    """
    global _main_loop

    from app.core.scheduler import scheduler

    _main_loop = asyncio.get_event_loop()
    set_main_event_loop(_main_loop)

    start_scheduler()

    scheduler.add_job(
        _sync_ingestion,
        trigger=CronTrigger(minute=0),
        id="ingestion",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

    scheduler.add_job(
        _sync_analysis,
        trigger=IntervalTrigger(minutes=5),
        id="analysis",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

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
    """
    Check whether ingestion ran within _REDEPLOY_GUARD_MINUTES.

    Queries the most recently updated article. If updated less than
    _REDEPLOY_GUARD_MINUTES ago, the old instance already ran ingestion
    during a redeploy — skip to avoid double-ingestion OOM.
    Returns True if ingestion should be skipped, False otherwise.
    """
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
        print(
            f"Redeploy guard check failed: {exc} "
            "— proceeding with ingestion."
        )

    return False


async def _scheduled_ingestion() -> None:
    """
    Async ingestion wrapper with overlap guard.
    Sets _heavy_job_running to block analysis during scraping.
    """
    global _ingestion_running, _heavy_job_running

    if _ingestion_running or _heavy_job_running:
        print("Heavy job already running — skipping ingestion trigger.")
        return

    _ingestion_running = True
    _heavy_job_running = True
    try:
        await asyncio.to_thread(run_ingestion_cycle)
    finally:
        _ingestion_running = False
        _heavy_job_running = False


async def _run_startup_sequence() -> None:
    """
    Run ingestion on startup, then wait for RAM to clear before
    triggering the first analysis run.

    Redeploy guard: if the DB shows articles updated within
    _REDEPLOY_GUARD_MINUTES, skip ingestion and wait the full
    _ANALYSIS_STARTUP_DELAY_SECONDS before analysis — same delay
    as the normal path to avoid immediate post-redeploy OOM.

    Heavy job lock: _heavy_job_running is held during ingestion so
    the APScheduler analysis bridge cannot fire simultaneously.
    """
    global _ingestion_running, _heavy_job_running

    if _ingestion_ran_recently():
        print(
            "Startup ingestion skipped (redeploy guard). "
            f"Waiting {_ANALYSIS_STARTUP_DELAY_SECONDS}s "
            "then running analysis..."
        )
        await asyncio.sleep(_ANALYSIS_STARTUP_DELAY_SECONDS)
        await _guarded_analysis()
        return

    print("Server startup: Running ingestion...")
    _ingestion_running = True
    _heavy_job_running = True
    try:
        await asyncio.to_thread(run_ingestion_cycle)
        print("Startup ingestion complete")
    except Exception as exc:
        print(f"Startup ingestion failed: {exc}")
    finally:
        _ingestion_running = False
        _heavy_job_running = False

    print(
        f"Waiting {_ANALYSIS_STARTUP_DELAY_SECONDS}s before first "
        "analysis run to allow ingestion RAM to clear..."
    )
    await asyncio.sleep(_ANALYSIS_STARTUP_DELAY_SECONDS)
    print("Starting post-startup analysis run...")
    await _guarded_analysis()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="NewsScope API", version="1.0.0", lifespan=lifespan)


# ── Auth ──────────────────────────────────────────────────────────────────────


def get_current_user(
    authorization: Optional[str] = Header(None),
) -> str:
    """Extract and verify user ID from Firebase JWT token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.replace("Bearer ", "")

    try:
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token["uid"]
        print(f"Authenticated user: {user_id}")
        return user_id
    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=401, detail="Invalid authentication token"
        )
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=401, detail="Authentication token expired"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=401, detail=f"Authentication failed: {exc}"
        )


# ── Core routes ───────────────────────────────────────────────────────────────


@app.get("/")
def root():
    """API root — returns version and docs link."""
    return {
        "message": (
            "Welcome to NewsScope API. Try /health to check status."
        ),
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.head("/")
def root_head():
    """HEAD handler for uptime monitors."""
    return {}


@app.get("/health")
def health():
    """Instant in-memory response — never touches DB or models."""
    return {"status": "ok"}


# ── Internal + debug routes ───────────────────────────────────────────────────


@app.post("/internal/analyze-batch")
async def internal_analyze_batch(background_tasks: BackgroundTasks):
    """
    Manual or external analysis trigger. Keep-alive no longer calls
    this — analysis is handled exclusively by APScheduler every 5
    minutes. Retained for debug use only.
    Uses BackgroundTasks to avoid asyncio.create_task race condition
    where the task could slip past the _heavy_job_running check.
    """
    if _heavy_job_running:
        return {"status": "skipped — heavy job in progress"}
    background_tasks.add_task(_guarded_analysis)
    return {"status": "ok"}


@app.post("/debug/ingest")
async def debug_ingest(background_tasks: BackgroundTasks):
    """Manually trigger an ingestion cycle."""
    background_tasks.add_task(_scheduled_ingestion)
    return {"status": "ingestion triggered in background"}


@app.post("/debug/analyze")
async def debug_analyze(background_tasks: BackgroundTasks):
    """Manually trigger an analysis cycle."""
    background_tasks.add_task(_guarded_analysis)
    return {"status": "analysis triggered in background"}


@app.post("/debug/archive")
async def debug_archive(background_tasks: BackgroundTasks):
    """Manually trigger an archiving cycle."""
    background_tasks.add_task(archive_old_articles)
    return {"status": "archiving triggered in background"}


# ── User API ──────────────────────────────────────────────────────────────────


@app.post("/api/reading-history")
async def track_reading(
    data: ReadingHistoryCreate,
    user_id: str = Depends(get_current_user),
):
    """Track article reading time for bias profile calculation."""
    try:
        payload = {
            "user_id": user_id,
            "article_id": data.article_id,
            "time_spent_seconds": data.time_spent_seconds,
            "opened_at": datetime.utcnow().isoformat(),
        }
        response = supabase.table("reading_history").upsert(
            payload, on_conflict="user_id,article_id"
        ).execute()
        print("Reading tracked successfully")
        return {"success": True, "data": response.data}
    except Exception as exc:
        print(f"Error tracking reading: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/bias-profile", response_model=BiasProfile)
async def get_bias_profile(user_id: str = Depends(get_current_user)):
    """Calculate user's bias profile from reading history."""
    try:
        response = (
            supabase.table("reading_history")
            .select("time_spent_seconds, articles(*)")
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
                bias_distribution={
                    "left": 0.0, "center": 0.0, "right": 0.0
                },
                reading_time_total_minutes=0,
                positive_count=0,
                neutral_count=0,
                negative_count=0,
            )

        total_time = sum(h["time_spent_seconds"] for h in history)

        bias_terms = [
            (h["time_spent_seconds"], h["articles"].get("bias_score"))
            for h in history
            if h["articles"].get("bias_score") is not None
        ]
        sent_terms = [
            (h["time_spent_seconds"], h["articles"].get("sentiment_score"))
            for h in history
            if h["articles"].get("sentiment_score") is not None
        ]

        weighted_bias = (
            sum(t * b for t, b in bias_terms) / sum(t for t, _ in bias_terms)
            if bias_terms
            else 0.0
        )
        weighted_sentiment = (
            sum(t * s for t, s in sent_terms) / sum(t for t, _ in sent_terms)
            if sent_terms
            else 0.0
        )

        left = sum(
            1
            for h in history
            if (b := h["articles"].get("bias_score")) is not None and b < -0.3
        )
        right = sum(
            1
            for h in history
            if (b := h["articles"].get("bias_score")) is not None and b > 0.3
        )
        center = len(history) - left - right

        pos = sum(
            1
            for h in history
            if (s := h["articles"].get("sentiment_score")) and s > 0.3
        )
        neg = sum(
            1
            for h in history
            if (s := h["articles"].get("sentiment_score")) and s < -0.3
        )
        neu = len(history) - pos - neg

        sources = [
            h["articles"]["source"]
            for h in history
            if h["articles"].get("source")
        ]
        most_read = (
            Counter(sources).most_common(1)[0][0] if sources else "N/A"
        )

        total_reads = len(history)
        distribution = {
            "left": (
                round(left / total_reads * 100, 1) if total_reads else 0.0
            ),
            "center": (
                round(center / total_reads * 100, 1) if total_reads else 0.0
            ),
            "right": (
                round(right / total_reads * 100, 1) if total_reads else 0.0
            ),
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
        )
    except Exception as exc:
        print(f"Error fetching bias profile: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/articles/compare", response_model=ComparisonResponse)
async def compare_articles(request: ComparisonRequest):
    """Find articles on the same topic grouped by political bias."""
    try:
        response = (
            supabase.table("articles")
            .select("*")
            .or_(
                f"title.ilike.%{request.topic}%,"
                f"content.ilike.%{request.topic}%"
            )
            .not_.is_("bias_score", "null")
            .order("published_at", desc=True)
            .execute()
        )
        articles_data = response.data

        left = [
            a for a in articles_data if a.get("bias_score", 0) < -0.3
        ]
        center = [
            a for a in articles_data
            if -0.3 <= a.get("bias_score", 0) <= 0.3
        ]
        right = [
            a for a in articles_data if a.get("bias_score", 0) > 0.3
        ]

        return ComparisonResponse(
            topic=request.topic,
            left_articles=left,
            center_articles=center,
            right_articles=right,
            total_found=len(articles_data),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/fact-checks/{article_id}", response_model=List[FactCheck])
async def get_article_fact_checks(article_id: str):
    """Return stored fact-checks for a given article."""
    try:
        response = (
            supabase.table("fact_checks")
            .select("*")
            .eq("article_id", article_id)
            .execute()
        )
        return response.data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(articles.router, prefix="/articles", tags=["articles"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(sources.router, prefix="/sources", tags=["sources"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon if present, otherwise return empty response."""
    file_path = os.path.join(BASE_DIR, "..", "static", "favicon.ico")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"status": "no icon"}
