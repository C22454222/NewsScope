"""
NewsScope FastAPI application entry point.

Lifespan manages scheduler startup, job registration, keep-alive,
and a single startup ingestion run. Analysis is deliberately delayed
60 seconds after startup ingestion to prevent simultaneous memory
pressure from both jobs on Render free tier (512MB limit).

Flake8: 0 errors/warnings.
"""

import os
import json
import asyncio
from contextlib import asynccontextmanager
from collections import Counter
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, Depends
from fastapi.responses import FileResponse

import firebase_admin
from firebase_admin import credentials, auth

from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from app.routes import articles, users, sources
from app.jobs.ingestion import run_ingestion_cycle, set_main_event_loop
from app.jobs.analysis import analyze_unscored_articles
from app.jobs.archiving import archive_old_articles
from app.jobs.keep_alive import start_keep_alive
from app.db.supabase import supabase
from app.schemas import (
    ReadingHistoryCreate,
    BiasProfile,
    FactCheck,
    ComparisonRequest,
    ComparisonResponse,
)
from app.core.scheduler import start_scheduler


# ── Main event loop — captured in lifespan for sync bridges ──────────────────
_main_loop: Optional[asyncio.AbstractEventLoop] = None

# ── Job guards — prevent concurrent overlap ───────────────────────────────────
_ingestion_running = False
_analysis_running = False

# ── Startup analysis delay — lets ingestion RAM clear before analysis runs ────
_ANALYSIS_STARTUP_DELAY_SECONDS = 60


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
    """Sync bridge — runs _scheduled_ingestion on the main event loop."""
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
    Sync bridge — runs analyze_unscored_articles on the main event loop.
    Fires every 5 minutes so even brief stable windows score articles.
    """
    if _main_loop is None or not _main_loop.is_running():
        print("Analysis bridge: no running loop, skipping.")
        return
    future = asyncio.run_coroutine_threadsafe(
        analyze_unscored_articles(), _main_loop
    )
    try:
        future.result(timeout=600)
    except Exception as exc:
        print(f"Scheduled analysis error: {exc}")


def _sync_archive() -> None:
    """Sync bridge — runs archive_old_articles on the main event loop."""
    if _main_loop is None or not _main_loop.is_running():
        print("Archive bridge: no running loop, skipping.")
        return
    future = asyncio.run_coroutine_threadsafe(
        archive_old_articles(), _main_loop
    )
    try:
        future.result(timeout=1800)
    except Exception as exc:
        print(f"Scheduled archiving error: {exc}")


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Startup order:
      1. Capture main event loop for sync bridges
      2. Start APScheduler with ingestion, analysis, archiving jobs
      3. Start keep-alive pinger (production only)
      4. Run startup ingestion in background
      5. Delay 60s then run first analysis — prevents simultaneous
         ingestion + analysis memory pressure on Render free tier

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


async def _scheduled_ingestion() -> None:
    """Async ingestion wrapper with overlap guard."""
    global _ingestion_running

    if _ingestion_running:
        print("Ingestion already running — skipping duplicate trigger.")
        return

    _ingestion_running = True
    try:
        await asyncio.to_thread(run_ingestion_cycle)
    finally:
        _ingestion_running = False


async def _run_startup_sequence() -> None:
    """
    Run ingestion on startup, then wait for RAM to clear before
    triggering the first analysis run. Prevents simultaneous memory
    pressure from both jobs crashing the Render free tier instance.
    """
    global _ingestion_running

    print("Server startup: Running ingestion...")
    _ingestion_running = True
    try:
        await asyncio.to_thread(run_ingestion_cycle)
        print("Startup ingestion complete")
    except Exception as exc:
        print(f"Startup ingestion failed: {exc}")
    finally:
        _ingestion_running = False

    print(
        f"Waiting {_ANALYSIS_STARTUP_DELAY_SECONDS}s before first "
        "analysis run to allow ingestion RAM to clear..."
    )
    await asyncio.sleep(_ANALYSIS_STARTUP_DELAY_SECONDS)
    print("Starting post-startup analysis run...")
    await analyze_unscored_articles()


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
async def internal_analyze_batch():
    """
    Additional analysis trigger called by keep-alive every 14 minutes.
    Skips silently if analysis is already running.
    """
    asyncio.create_task(analyze_unscored_articles())
    return {"status": "ok"}


@app.post("/debug/ingest")
async def debug_ingest(background_tasks: BackgroundTasks):
    """Manually trigger an ingestion cycle."""
    background_tasks.add_task(_scheduled_ingestion)
    return {"status": "ingestion triggered in background"}


@app.post("/debug/analyze")
async def debug_analyze(background_tasks: BackgroundTasks):
    """Manually trigger an analysis cycle."""
    background_tasks.add_task(analyze_unscored_articles)
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
