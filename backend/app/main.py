"""
NewsScope FastAPI application entry point.

SCHEDULING OVERVIEW (v8 — chain-first, FCM notifications):

  PIPELINE (chained, fires automatically each cycle):
    :00  Ingestion starts (CronTrigger minute=0).
         Categorisation is inline — happens per-article during scraping.
         Completes in ~5 minutes on 512 MB Render free tier.
         _heavy_job_lock held for the entire duration.
      -> on completion: Analysis fires immediately — no fixed wait,
         no wasted idle minutes. Completes in ~10-15 minutes
         (200 articles × 8 concurrent HF calls). LIME runs inside
         analysis for high-confidence articles (~20-30 min).
      -> on completion: FCM notification sent to the news_updates topic
         if new articles were inserted during this cycle.
         Count is measured from articles.created_at within the last
         _NOTIFICATION_WINDOW_MINUTES minutes.
      -> Pipeline done. Total wall time: ~25-40 minutes per hour.

  SAFETY NETS (cron — only fire if the chained call failed/was skipped):
    :20  Analysis safety net  (CronTrigger minute=20)
    :50  Fact-check           (CronTrigger minute=50)
    03:00 Archiving           (CronTrigger hour=3)
"""

import asyncio
import json
import os
import threading
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import firebase_admin
from apscheduler.triggers.cron import CronTrigger
from dateutil import parser as dtparser
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from firebase_admin import auth, credentials, messaging

from app.core.scheduler import start_scheduler
from app.db.supabase import supabase
from app.jobs.analysis import analyze_unscored_articles
from app.jobs.archiving import archive_old_articles
from app.jobs.fact_checking import batch_factcheck_recent
from app.jobs.ingestion import run_ingestion_cycle, set_main_event_loop
from app.jobs.keep_alive import keep_alive, start_keep_alive
from app.routes import articles, sources, users
from app.routes.articles import CATEGORY_GROUP_MAP
from app.schemas import (
    BiasProfile,
    ComparisonRequest,
    ComparisonResponse,
    ReadingHistoryCreate,
)

# Captured in lifespan and passed to ingestion so APScheduler threads
# can schedule coroutines onto the main event loop via run_coroutine_threadsafe.
_main_loop: Optional[asyncio.AbstractEventLoop] = None

# Guards that prevent overlapping ingestion/analysis cycles.
_ingestion_running = False
_analysis_running = False
_heavy_job_lock = threading.Lock()

# Minimum age of the most recent article (minutes) required before
# a startup cycle is allowed to run — prevents bursting immediately
# after a cold Render redeploy.
_REDEPLOY_GUARD_MINUTES = 45

# Look-back window used to count newly inserted articles for FCM.
_NOTIFICATION_WINDOW_MINUTES = 30


def init_firebase() -> None:
    """
    Initialise Firebase Admin SDK.

    Reads credentials from the FIREBASE_SERVICE_ACCOUNT environment
    variable (JSON string) if present, otherwise falls back to a local
    firebase-service-account.json file. ValueError is swallowed — it
    means the SDK was already initialised (e.g. during hot-reload).
    """
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


def _political_bias_score(label: Optional[str]) -> Optional[float]:
    """Map a RoBERTa political_bias label to a [-1, +1] numeric score."""
    if not label:
        return None
    upper = label.upper()
    if upper == "LEFT":
        return -1.0
    if upper == "RIGHT":
        return 1.0
    if upper in ("CENTER", "CENTRE"):
        return 0.0
    return None


def _count_recently_inserted_articles() -> int:
    """Count articles inserted in the last _NOTIFICATION_WINDOW_MINUTES."""
    try:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=_NOTIFICATION_WINDOW_MINUTES)
        ).isoformat()
        response = (
            supabase.table("articles")
            .select("id", count="exact")
            .gte("created_at", cutoff)
            .execute()
        )
        return response.count or 0
    except Exception as exc:
        print(f"Failed to count new articles: {exc}")
        return 0


def _notify_new_articles() -> None:
    """
    Send an FCM push notification to the news_updates topic if new
    articles were inserted during this ingestion cycle.

    Skipped silently when the count is zero. FCM failures are logged
    and swallowed — a notification failure must never crash the pipeline.
    """
    count = _count_recently_inserted_articles()
    if count <= 0:
        print("FCM notification skipped — no new articles in last cycle.")
        return

    body = (
        f"{count} new article{'s' if count != 1 else ''} ready to read"
    )

    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title="NewsScope",
                body=body,
            ),
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    channel_id="news_updates",
                    priority="high",
                    default_sound=True,
                ),
            ),
            topic="news_updates",
        )
        message_id = messaging.send(message)
        print(
            f"FCM notification sent to news_updates topic: {count} new "
            f"article{'s' if count != 1 else ''} "
            f"(message_id={message_id})"
        )
    except Exception as exc:
        print(f"FCM publish failed: {exc}")


# ── APScheduler sync bridges ──────────────────────────────────────────────────
# APScheduler runs jobs in background threads. Each bridge acquires the
# heavy-job lock, then schedules the async coroutine onto the main loop via
# run_coroutine_threadsafe. The lock is always released in a finally block.

def _sync_ingestion() -> None:
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
    if not _heavy_job_lock.acquire(blocking=False):
        print("Heavy job in progress — skipping scheduled fact-check.")
        return
    if _main_loop is None or not _main_loop.is_running():
        _heavy_job_lock.release()
        print("Fact-check bridge: no running loop, skipping.")
        return
    future = asyncio.run_coroutine_threadsafe(
        batch_factcheck_recent(hours=48), _main_loop
    )
    try:
        future.result(timeout=1800)
    except Exception as exc:
        print(f"Scheduled fact-check error: {exc}")
    finally:
        _heavy_job_lock.release()


def _sync_archive() -> None:
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


# ── Async pipeline wrappers ───────────────────────────────────────────────────

async def _scheduled_ingestion() -> None:
    """
    Run ingestion then immediately chain analysis and FCM notification.

    Guard flag prevents duplicate runs if the cron fires while a prior
    cycle is still executing. The chain fires on completion so analysis
    always starts with a fresh article pool and no idle wait is wasted.
    """
    global _ingestion_running
    if _ingestion_running:
        print("Ingestion already running — skipping duplicate trigger.")
        return
    _ingestion_running = True
    try:
        print("Pipeline: starting ingestion...")
        await asyncio.to_thread(run_ingestion_cycle)
        print(
            "Pipeline: ingestion complete — "
            "chaining analysis immediately."
        )
    except Exception as exc:
        print(f"Pipeline: ingestion failed: {exc}")
        return
    finally:
        _ingestion_running = False

    await _guarded_analysis()
    await asyncio.to_thread(_notify_new_articles)


async def _guarded_analysis() -> None:
    """
    Run analysis with a guard that prevents overlapping cycles.

    Called directly by _scheduled_ingestion after ingestion completes,
    and also by the :20 safety net when ingestion was skipped or failed.
    """
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
    FastAPI lifespan handler.

    Captures the main event loop, starts APScheduler with four jobs
    (ingestion, analysis safety net, fact-check, archiving), and
    launches keep-alive in production. Schedules a startup sequence
    that respects the redeploy guard before running the first cycle.
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
        _sync_analysis_safety_net,
        trigger=CronTrigger(minute=20),
        id="analysis_safety_net",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

    scheduler.add_job(
        _sync_factcheck,
        trigger=CronTrigger(minute=50),
        id="factcheck",
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
        print("Keep-alive enabled (pings every 10 minutes)")
        asyncio.create_task(asyncio.to_thread(keep_alive))
    else:
        print("Keep-alive disabled in development")

    asyncio.create_task(_run_startup_sequence())

    yield
    print("Server shutting down...")


# ── Startup helpers ───────────────────────────────────────────────────────────

def _ingestion_ran_recently() -> bool:
    """
    Return True if the most recent article update was within
    _REDEPLOY_GUARD_MINUTES, indicating a pipeline ran before this
    deploy and a full startup cycle should be skipped.

    Falls back to False (i.e., proceed with ingestion) on any error
    so a DB outage doesn't permanently suppress startup cycles.
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
                f"Redeploy guard: last ingestion {age_minutes:.1f} min "
                "ago — skipping startup ingestion."
            )
            return True

    except Exception as exc:
        print(
            f"Redeploy guard check failed: {exc} "
            "— proceeding with ingestion."
        )

    return False


async def _run_startup_sequence() -> None:
    """
    Run on startup: skip the full pipeline if the redeploy guard fires,
    otherwise acquire the lock and run the full chained pipeline
    (ingestion → analysis → FCM notification).

    When the guard fires, waits 180 s then runs analysis alone so
    freshly deployed scoring improvements are applied to existing articles.
    """
    if _ingestion_ran_recently():
        print(
            "Startup ingestion skipped (redeploy guard). "
            "Waiting 180s then running analysis..."
        )
        await asyncio.sleep(180)
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

    try:
        await _scheduled_ingestion()
        print(
            "Startup pipeline complete "
            "(ingestion + analysis + notify)."
        )
    except Exception as exc:
        print(f"Startup pipeline failed: {exc}")
    finally:
        _heavy_job_lock.release()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="NewsScope API", version="1.0.0", lifespan=lifespan)


# ── Auth dependency ───────────────────────────────────────────────────────────

def get_current_user(
    authorization: Optional[str] = Header(None),
) -> str:
    """
    Validate the Firebase ID token from the Authorization header.

    Returns the authenticated Firebase UID on success. Raises HTTP 401
    for missing, invalid, or expired tokens.
    """
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
    return {
        "message": (
            "Welcome to NewsScope API. Try /health to check status."
        ),
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


@app.post("/debug/notify")
async def debug_notify():
    await asyncio.to_thread(_notify_new_articles)
    return {"status": "notification attempted"}


# ── User API ──────────────────────────────────────────────────────────────────

@app.post("/api/reading-history")
async def track_reading(
    data: ReadingHistoryCreate,
    user_id: str = Depends(get_current_user),
):
    """
    Record that the current user read an article and snapshot the
    article's bias/credibility scores onto the history row.

    Uses an update-if-greater strategy on an existing row: if the client
    sends a shorter duration than what is already stored (e.g. a 5 s
    auto-ping after the user already read the full article), the stored
    value is kept. This means multiple calls from a single reading session
    (5 s auto-track, exit hook, app backgrounding) always converge to the
    longest observed duration without creating duplicate rows.
    """
    try:
        article_data = (
            supabase.table("articles")
            .select(
                "bias_score, sentiment_score, source, general_bias, "
                "credibility_score, political_bias"
            )
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
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "bias_score": scores.get("bias_score"),
            "sentiment_score": scores.get("sentiment_score"),
            "source": scores.get("source"),
            "general_bias": scores.get("general_bias"),
            "credibility_score": scores.get("credibility_score"),
            "political_bias": scores.get("political_bias"),
        }

        existing = (
            supabase.table("reading_history")
            .select("id, time_spent_seconds")
            .eq("user_id", user_id)
            .eq("article_id", data.article_id)
            .limit(1)
            .execute()
            .data
        )

        if existing:
            stored_time = existing[0].get("time_spent_seconds", 0) or 0
            if data.time_spent_seconds > stored_time:
                response = (
                    supabase.table("reading_history")
                    .update(payload)
                    .eq("id", existing[0]["id"])
                    .execute()
                )
                print(
                    f"Reading updated: {data.article_id} "
                    f"({data.time_spent_seconds}s, was {stored_time}s)"
                )
                return {"success": True, "data": response.data}
            return {"success": True, "data": existing}

        response = (
            supabase.table("reading_history").insert(payload).execute()
        )
        print(
            f"Reading tracked: {data.article_id} "
            f"({data.time_spent_seconds}s)"
        )
        return {"success": True, "data": response.data}
    except Exception as exc:
        print(f"Error tracking reading: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/bias-profile", response_model=BiasProfile)
async def get_bias_profile(user_id: str = Depends(get_current_user)):
    """
    Calculate and return the bias profile for the current user.

    All scores are read from snapshot columns on reading_history so
    the profile remains accurate after articles are archived and purged
    from the articles table. Nothing is persisted — computed live on
    every request.
    """
    try:
        response = (
            supabase.table("reading_history")
            .select(
                "time_spent_seconds, bias_score, sentiment_score, "
                "source, general_bias, credibility_score, "
                "political_bias"
            )
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
                source_breakdown={},
                avg_credibility=None,
                article_left_count=0,
                article_center_count=0,
                article_right_count=0,
                article_bias_distribution={
                    "left": 0.0, "center": 0.0, "right": 0.0
                },
                avg_article_bias=0.0,
                biased_count=0,
                unbiased_count=0,
            )

        total_time = sum(h["time_spent_seconds"] for h in history)

        # Outlet-level bias — time-weighted average of bias_score snapshot.
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
        most_read = (
            source_counter.most_common(1)[0][0]
            if source_counter else "N/A"
        )
        source_breakdown = dict(source_counter.most_common(12))

        total_reads = len(history)
        distribution = {
            "left": (
                round(left / total_reads * 100, 1)
                if total_reads else 0.0
            ),
            "center": (
                round(center / total_reads * 100, 1)
                if total_reads else 0.0
            ),
            "right": (
                round(right / total_reads * 100, 1)
                if total_reads else 0.0
            ),
        }

        cred_values = [
            h["credibility_score"]
            for h in history
            if h.get("credibility_score") is not None
        ]
        avg_credibility = (
            round(sum(cred_values) / len(cred_values), 1)
            if cred_values else None
        )

        # Article-level bias — time-weighted average of political_bias snapshot.
        art_left = 0
        art_center = 0
        art_right = 0
        article_pb_terms = []

        for h in history:
            pb_label = h.get("political_bias")
            pb_score = _political_bias_score(pb_label)

            if pb_label:
                upper = pb_label.upper()
                if upper == "LEFT":
                    art_left += 1
                elif upper == "RIGHT":
                    art_right += 1
                elif upper in ("CENTER", "CENTRE"):
                    art_center += 1

            if pb_score is not None:
                article_pb_terms.append(
                    (h["time_spent_seconds"], pb_score)
                )

        avg_article_bias = (
            sum(t * s for t, s in article_pb_terms) / sum(t for t, _ in article_pb_terms)
            if article_pb_terms else 0.0
        )

        art_total = art_left + art_center + art_right
        article_bias_distribution = {
            "left": (
                round(art_left / art_total * 100, 1)
                if art_total else 0.0
            ),
            "center": (
                round(art_center / art_total * 100, 1)
                if art_total else 0.0
            ),
            "right": (
                round(art_right / art_total * 100, 1)
                if art_total else 0.0
            ),
        }

        # General bias — count of BIASED / UNBIASED labels from DistilRoBERTa.
        biased_count = 0
        unbiased_count = 0
        for h in history:
            gb = h.get("general_bias")
            if not gb:
                continue
            upper = gb.upper()
            if upper == "BIASED":
                biased_count += 1
            elif upper == "UNBIASED":
                unbiased_count += 1

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
            avg_credibility=avg_credibility,
            article_left_count=art_left,
            article_center_count=art_center,
            article_right_count=art_right,
            article_bias_distribution=article_bias_distribution,
            avg_article_bias=round(avg_article_bias, 3),
            biased_count=biased_count,
            unbiased_count=unbiased_count,
        )
    except Exception as exc:
        print(f"Error fetching bias profile: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/articles/compare", response_model=ComparisonResponse)
async def compare_articles(request: ComparisonRequest):
    """
    Group articles by political leaning for the Compare screen.

    Filters the full article pool (no date cutoff) by topic (ilike on
    title and content), category (sub-category expansion via
    CATEGORY_GROUP_MAP), and source. Splits the result into left
    (bias_score < -0.3), center (-0.3 to 0.3), and right (> 0.3) bands.
    """
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
            related = [
                k for k, v in CATEGORY_GROUP_MAP.items()
                if v == request.category
            ] + [request.category]
            query = query.in_("category", related)

        if request.source:
            query = query.eq("source", request.source)

        articles_data = query.limit(10000).execute().data

        left = [
            a for a in articles_data
            if a.get("bias_score", 0) < -0.3
        ]
        center = [
            a for a in articles_data
            if -0.3 <= a.get("bias_score", 0) <= 0.3
        ]
        right = [
            a for a in articles_data
            if a.get("bias_score", 0) > 0.3
        ]

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

app.include_router(
    articles.router, prefix="/articles", tags=["articles"]
)
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(sources.router, prefix="/sources", tags=["sources"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/favicon.ico")
async def favicon():
    file_path = os.path.join(BASE_DIR, "..", "static", "favicon.ico")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"status": "no icon"}
