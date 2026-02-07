# app/main.py
import os
import json
import asyncio
from contextlib import asynccontextmanager
from collections import Counter
from datetime import datetime
from typing import Optional, List

from fastapi import (
    FastAPI,
    BackgroundTasks,
    HTTPException,
    Header,
    Depends
)
from fastapi.responses import FileResponse

# Firebase Admin imports
import firebase_admin
from firebase_admin import credentials, auth

from app.routes import articles, users, sources
from app.core.scheduler import start_scheduler
from app.jobs.ingestion import run_ingestion_cycle
from app.jobs.analysis import analyze_unscored_articles
from app.jobs.archiving import archive_old_articles
from app.jobs.keep_alive import start_keep_alive
from app.schemas import (
    ReadingHistoryCreate,
    BiasProfile,
    FactCheck,
    ComparisonRequest,
    ComparisonResponse
)
from app.db.supabase import supabase


def init_firebase():
    """Initialize Firebase Admin SDK."""
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
        print("ℹFirebase Admin already initialized")
    except Exception as e:
        print(f"Firebase Admin init failed: {e}")


init_firebase()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from apscheduler.triggers.cron import CronTrigger
    from app.core.scheduler import scheduler

    start_scheduler()

    scheduler.add_job(
        run_ingestion_cycle,
        trigger=CronTrigger(minute=0),
        id="ingestion",
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        analyze_unscored_articles,
        trigger=CronTrigger(minute=15),
        id="analysis",
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        archive_old_articles,
        trigger=CronTrigger(hour=3, minute=0),
        id="archiving",
        max_instances=1,
        coalesce=True,
    )

    env = os.getenv("ENVIRONMENT", "production")
    print(f"Environment: {env}")

    if env == "production":
        start_keep_alive()
        print("Keep-alive enabled (pings every 14 minutes)")
    else:
        print("Keep-alive disabled in development")

    asyncio.create_task(_run_startup_jobs())

    yield

    print("Server shutting down...")


async def _run_startup_jobs():
    print("Server startup: Running ingestion + analysis...")

    try:
        await asyncio.to_thread(run_ingestion_cycle)
        print("Startup ingestion complete")

        await asyncio.to_thread(analyze_unscored_articles)
        print("Startup analysis complete")
    except Exception as e:
        print(f"Startup jobs failed: {e}")


app = FastAPI(
    title="NewsScope API",
    version="1.0.0",
    lifespan=lifespan,
)


def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Extract user ID from Firebase JWT token.
    Used for authenticated endpoints.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
        )

    token = authorization.replace("Bearer ", "")

    try:
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token["uid"]

        print(f"Authenticated user: {user_id}")
        return user_id

    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
        )
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Authentication token expired",
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {e}",
        )


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


@app.post("/debug/ingest")
async def debug_ingest(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_ingestion_cycle)
    return {"status": "ingestion triggered in background"}


@app.post("/debug/analyze")
async def debug_analyze(background_tasks: BackgroundTasks):
    background_tasks.add_task(analyze_unscored_articles)
    return {"status": "analysis triggered in background"}


@app.post("/debug/archive")
async def debug_archive(background_tasks: BackgroundTasks):
    background_tasks.add_task(archive_old_articles)
    return {"status": "archiving triggered in background"}


# --- Reading history & bias profile ---


@app.post("/api/reading-history")
async def track_reading(
    data: ReadingHistoryCreate,
    user_id: str = Depends(get_current_user),
):
    """
    Track article reading time for bias profile calculation.
    Called when user exits article view in mobile app.

    NOTE: For archiving-safe stats you can snapshot bias/sentiment/source
    into reading_history here. This keeps the profile stable even after
    articles are archived.
    """
    try:
        print(
            f"Tracking reading: user={user_id}, "
            f"article={data.article_id}, time={data.time_spent_seconds}s"
        )

        payload = {
            "user_id": user_id,
            "article_id": data.article_id,
            "time_spent_seconds": data.time_spent_seconds,
            "opened_at": datetime.utcnow().isoformat(),
        }

        response = supabase.table("reading_history").upsert(
            payload,
            on_conflict="user_id,article_id",
        ).execute()

        print("Reading tracked successfully")
        return {"success": True, "data": response.data}

    except Exception as e:
        print(f"Error tracking reading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bias-profile", response_model=BiasProfile)
async def get_bias_profile(user_id: str = Depends(get_current_user)):
    """
    Calculate user's bias profile based on reading history.
    Weighted by time spent on each article.
    """
    try:
        print(f"Fetching bias profile for user: {user_id}")

        response = (
            supabase.table("reading_history")
            .select("time_spent_seconds, articles(*)")
            .eq("user_id", user_id)
            .execute()
        )

        history = response.data
        print(f"Found {len(history)} articles in reading history")

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
                    "left": 0.0,
                    "center": 0.0,
                    "right": 0.0,
                },
                reading_time_total_minutes=0,
                positive_count=0,
                neutral_count=0,
                negative_count=0,
            )

        # Total reading time across all entries
        total_time = sum(h["time_spent_seconds"] for h in history)

        # Weighted average bias (only when bias_score is not None)
        bias_terms = [
            (h["time_spent_seconds"], h["articles"].get("bias_score"))
            for h in history
            if h["articles"].get("bias_score") is not None
        ]
        if bias_terms:
            bias_time = sum(t for t, _ in bias_terms)
            weighted_bias = (
                sum(t * b for t, b in bias_terms) / bias_time
                if bias_time > 0
                else 0.0
            )
        else:
            weighted_bias = 0.0

        # Weighted average sentiment (only when sentiment_score is not None)
        sent_terms = [
            (h["time_spent_seconds"], h["articles"].get("sentiment_score"))
            for h in history
            if h["articles"].get("sentiment_score") is not None
        ]
        if sent_terms:
            sent_time = sum(t for t, _ in sent_terms)
            weighted_sentiment = (
                sum(t * s for t, s in sent_terms) / sent_time
                if sent_time > 0
                else 0.0
            )
        else:
            weighted_sentiment = 0.0

        # Count by political leaning (handle 0.0 correctly)
        left = 0
        center = 0
        right = 0
        for h in history:
            bias = h["articles"].get("bias_score")
            if bias is None:
                continue
            if bias < -0.3:
                left += 1
            elif bias > 0.3:
                right += 1
            else:
                center += 1

        # Sentiment band counts
        pos_count = 0
        neu_count = 0
        neg_count = 0
        for h in history:
            s = h["articles"].get("sentiment_score")
            if s is None:
                continue
            if s > 0.3:
                pos_count += 1
            elif s < -0.3:
                neg_count += 1
            else:
                neu_count += 1

        # Most read source
        sources = [
            h["articles"]["source"]
            for h in history
            if h["articles"].get("source")
        ]
        most_common = Counter(sources).most_common(1)
        most_read = most_common[0][0] if most_common else "N/A"

        # Bias distribution in percentages (based on reading events)
        total_reads = len(history)
        distribution = {
            "left": round(left / total_reads * 100, 1)
            if total_reads > 0
            else 0.0,
            "center": round(center / total_reads * 100, 1)
            if total_reads > 0
            else 0.0,
            "right": round(right / total_reads * 100, 1)
            if total_reads > 0
            else 0.0,
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
            positive_count=pos_count,
            neutral_count=neu_count,
            negative_count=neg_count,
        )

    except Exception as e:
        print(f"Error fetching bias profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Comparison View (NO LIMIT) ---


@app.post("/api/articles/compare", response_model=ComparisonResponse)
async def compare_articles(request: ComparisonRequest):
    """
    Find articles from different outlets covering same topic.
    Used for Comparison View feature.

    Returns ALL matching articles grouped by bias band.
    """
    try:
        # Fetch all matching articles (no limit on total)
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

        articles = response.data

        # Group by bias rating (no per-band limit)
        left = [
            a for a in articles
            if a.get("bias_score", 0) < -0.3
        ]

        center = [
            a for a in articles
            if -0.3 <= a.get("bias_score", 0) <= 0.3
        ]

        right = [
            a for a in articles
            if a.get("bias_score", 0) > 0.3
        ]

        return ComparisonResponse(
            topic=request.topic,
            left_articles=left,
            center_articles=center,
            right_articles=right,
            total_found=len(articles),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/fact-checks/{article_id}",
    response_model=List[FactCheck],
)
async def get_article_fact_checks(article_id: str):
    try:
        response = (
            supabase.table("fact_checks")
            .select("*")
            .eq("article_id", article_id)
            .execute()
        )
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
