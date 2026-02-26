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

from app.routes import articles, users, sources
from app.jobs.ingestion import run_ingestion_cycle
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
        print("ℹ Firebase Admin already initialized")
    except Exception as e:
        print(f"Firebase Admin init failed: {e}")


init_firebase()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run ingestion, analysis, and archiving jobs on schedule."""
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

    # Run ingestion only at startup (not analysis to save memory)
    asyncio.create_task(_run_startup_ingestion())

    yield
    print("Server shutting down...")


async def _run_startup_ingestion():
    """Run ingestion once on startup."""
    print("Server startup: Running ingestion...")
    try:
        await asyncio.to_thread(run_ingestion_cycle)
        print("Startup ingestion complete")
    except Exception as e:
        print(f"Startup ingestion failed: {e}")


app = FastAPI(title="NewsScope API", version="1.0.0", lifespan=lifespan)


def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """Extract user ID from Firebase JWT token."""
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
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {e}")


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


@app.post("/api/reading-history")
async def track_reading(data: ReadingHistoryCreate, user_id: str = Depends(get_current_user)):
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
    except Exception as e:
        print(f"Error tracking reading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
                bias_distribution={"left": 0.0, "center": 0.0, "right": 0.0},
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

        left = sum(1 for h in history if (b := h["articles"].get("bias_score")) is not None and b < -0.3)
        right = sum(1 for h in history if (b := h["articles"].get("bias_score")) is not None and b > 0.3)
        center = len(history) - left - right

        pos = sum(1 for h in history if (s := h["articles"].get("sentiment_score")) and s > 0.3)
        neg = sum(1 for h in history if (s := h["articles"].get("sentiment_score")) and s < -0.3)
        neu = len(history) - pos - neg

        sources = [
            h["articles"]["source"]
            for h in history
            if h["articles"].get("source")
        ]
        most_read = Counter(sources).most_common(1)[0][0] if sources else "N/A"

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
        )
    except Exception as e:
        print(f"Error fetching bias profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/articles/compare", response_model=ComparisonResponse)
async def compare_articles(request: ComparisonRequest):
    """Find articles on the same topic grouped by political bias."""
    try:
        response = (
            supabase.table("articles")
            .select("*")
            .or_(f"title.ilike.%{request.topic}%,content.ilike.%{request.topic}%")
            .not_.is_("bias_score", "null")
            .order("published_at", desc=True)
            .execute()
        )
        articles_data = response.data

        left = [a for a in articles_data if a.get("bias_score", 0) < -0.3]
        center = [a for a in articles_data if -0.3 <= a.get("bias_score", 0) <= 0.3]
        right = [a for a in articles_data if a.get("bias_score", 0) > 0.3]

        return ComparisonResponse(
            topic=request.topic,
            left_articles=left,
            center_articles=center,
            right_articles=right,
            total_found=len(articles_data),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fact-checks/{article_id}", response_model=List[FactCheck])
async def get_article_fact_checks(article_id: str):
    """Return stored fact-checks for an article."""
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
