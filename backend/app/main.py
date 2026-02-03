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
        # Try environment variable first (for Render)
        service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        if service_account:
            cred_dict = json.loads(service_account)
            cred = credentials.Certificate(cred_dict)
        else:
            # Fallback to file (for local development)
            cred = credentials.Certificate("firebase-service-account.json")
        
        firebase_admin.initialize_app(cred)
        print("✅ Firebase Admin initialized")
    except ValueError:
        # Already initialized
        print("ℹ️  Firebase Admin already initialized")
    except Exception as e:
        print(f"⚠️  Firebase Admin init failed: {e}")


# Initialize Firebase before creating the app
init_firebase()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown of background jobs.
    """
    from apscheduler.triggers.cron import CronTrigger
    from app.core.scheduler import scheduler

    # Start the APScheduler
    start_scheduler()

    # Schedule ingestion every hour at :00
    scheduler.add_job(
        run_ingestion_cycle,
        trigger=CronTrigger(minute=0),
        id="ingestion",
        max_instances=1,
        coalesce=True
    )

    # Schedule analysis every hour at :15 (15 min after ingestion)
    scheduler.add_job(
        analyze_unscored_articles,
        trigger=CronTrigger(minute=15),
        id="analysis",
        max_instances=1,
        coalesce=True
    )

    # Schedule archiving daily at 3 AM
    scheduler.add_job(
        archive_old_articles,
        trigger=CronTrigger(hour=3, minute=0),
        id="archiving",
        max_instances=1,
        coalesce=True
    )

    # Start keep-alive in production
    env = os.getenv("ENVIRONMENT", "production")
    print(f"🌍 Environment: {env}")

    if env == "production":
        start_keep_alive()
        print("✅ Keep-alive enabled (pings every 14 minutes)")
    else:
        print("ℹ️ Keep-alive disabled in development")

    # Run startup jobs asynchronously
    asyncio.create_task(_run_startup_jobs())

    yield

    # Shutdown logic
    print("🛑 Server shutting down...")


async def _run_startup_jobs():
    """
    Run critical jobs immediately when server starts.
    Ensures fresh data is available on deployment.
    """
    print("🚀 Server startup: Running ingestion + analysis...")

    try:
        # Run ingestion to fetch latest news
        await asyncio.to_thread(run_ingestion_cycle)
        print("✅ Startup ingestion complete")

        # Run analysis on any unscored articles
        await asyncio.to_thread(analyze_unscored_articles)
        print("✅ Startup analysis complete")
    except Exception as e:
        print(f"⚠️ Startup jobs failed: {e}")


app = FastAPI(
    title="NewsScope API",
    version="1.0.0",
    lifespan=lifespan
)


def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Extract user ID from Firebase JWT token.
    Used for authenticated endpoints.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )

    token = authorization.replace("Bearer ", "")

    try:
        # Verify Firebase ID token
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token['uid']
        
        print(f"✅ Authenticated user: {user_id}")
        return user_id
        
    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Authentication token expired"
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {e}"
        )


# ============================================================
# Core Endpoints
# ============================================================

@app.get("/")
def root():
    """Root endpoint with welcome message."""
    return {
        "message": (
            "Welcome to NewsScope API. "
            "Try /health to check status."
        ),
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.head("/")
def root_head():
    """HEAD request for root (used by monitoring tools)."""
    return {}


@app.get("/health")
def health():
    """
    Health check endpoint for uptime monitoring.
    Used by Render and keep-alive scheduler.
    """
    return {"status": "ok"}


# ============================================================
# Debug Endpoints
# ============================================================

@app.post("/debug/ingest")
async def debug_ingest(background_tasks: BackgroundTasks):
    """Manually trigger news ingestion job."""
    background_tasks.add_task(run_ingestion_cycle)
    return {"status": "ingestion triggered in background"}


@app.post("/debug/analyze")
async def debug_analyze(background_tasks: BackgroundTasks):
    """Manually trigger article analysis job."""
    background_tasks.add_task(analyze_unscored_articles)
    return {"status": "analysis triggered in background"}


@app.post("/debug/archive")
async def debug_archive(background_tasks: BackgroundTasks):
    """Manually trigger archiving job (>30 days old)."""
    background_tasks.add_task(archive_old_articles)
    return {"status": "archiving triggered in background"}


# ============================================================
# User Reading History & Bias Profile
# ============================================================

@app.post("/api/reading-history")
async def track_reading(
    data: ReadingHistoryCreate,
    user_id: str = Depends(get_current_user)
):
    """
    Track article reading time for bias profile calculation.
    Called when user exits article view in mobile app.
    """
    try:
        print(f"📊 Tracking reading: user={user_id}, article={data.article_id}, time={data.time_spent_seconds}s")
        
        response = supabase.table("reading_history").upsert({
            "user_id": user_id,
            "article_id": data.article_id,
            "time_spent_seconds": data.time_spent_seconds,
            "opened_at": datetime.utcnow().isoformat()
        }, on_conflict="user_id,article_id").execute()

        print(f"✅ Reading tracked successfully")
        return {"success": True, "data": response.data}

    except Exception as e:
        print(f"❌ Error tracking reading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bias-profile", response_model=BiasProfile)
async def get_bias_profile(user_id: str = Depends(get_current_user)):
    """
    Calculate user's bias profile based on reading history.
    Weighted by time spent on each article.
    """
    try:
        print(f"📊 Fetching bias profile for user: {user_id}")
        
        response = (
            supabase.table("reading_history")
            .select("time_spent_seconds, articles(*)")
            .eq("user_id", user_id)
            .execute()
        )

        history = response.data
        print(f"📚 Found {len(history)} articles in reading history")

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
                    "right": 0.0
                },
                reading_time_total_minutes=0
            )

        # Calculate weighted averages
        total_time = sum(h["time_spent_seconds"] for h in history)

        weighted_bias = sum(
            h["time_spent_seconds"] * (h["articles"]["bias_score"] or 0)
            for h in history
        ) / total_time if total_time > 0 else 0

        weighted_sentiment = sum(
            h["time_spent_seconds"] * (h["articles"]["sentiment_score"] or 0)
            for h in history
        ) / total_time if total_time > 0 else 0

        # Count by political leaning
        left = sum(
            1 for h in history
            if h["articles"].get("bias_score") and h["articles"]["bias_score"] < -0.3
        )
        center = sum(
            1 for h in history
            if h["articles"].get("bias_score") and
            -0.3 <= h["articles"]["bias_score"] <= 0.3
        )
        right = sum(
            1 for h in history
            if h["articles"].get("bias_score") and h["articles"]["bias_score"] > 0.3
        )

        # Most read source
        sources = [
            h["articles"]["source"]
            for h in history
            if h["articles"].get("source")
        ]
        most_common = Counter(sources).most_common(1)
        most_read = most_common[0][0] if most_common else "N/A"

        # Bias distribution
        total = len(history)
        distribution = {
            "left": round(
                left / total * 100, 1
            ) if total > 0 else 0.0,
            "center": round(
                center / total * 100, 1
            ) if total > 0 else 0.0,
            "right": round(
                right / total * 100, 1
            ) if total > 0 else 0.0
        }

        return BiasProfile(
            avg_bias=round(weighted_bias, 3),
            avg_sentiment=round(weighted_sentiment, 3),
            total_articles_read=len(history),
            left_count=left,
            center_count=center,
            right_count=right,
            most_read_source=most_read,
            bias_distribution=distribution,
            reading_time_total_minutes=round(total_time / 60)
        )

    except Exception as e:
        print(f"❌ Error fetching bias profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Comparison View
# ============================================================

@app.post("/api/articles/compare", response_model=ComparisonResponse)
async def compare_articles(request: ComparisonRequest):
    """
    Find articles from different outlets covering same topic.
    Used for Comparison View feature.
    """
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
            .limit(request.limit * 3)
            .execute()
        )

        articles = response.data

        # Group by bias rating
        left = [
            a for a in articles
            if a.get("bias_score", 0) < -0.3
        ][:request.limit]

        center = [
            a for a in articles
            if -0.3 <= a.get("bias_score", 0) <= 0.3
        ][:request.limit]

        right = [
            a for a in articles
            if a.get("bias_score", 0) > 0.3
        ][:request.limit]

        return ComparisonResponse(
            topic=request.topic,
            left_articles=left,
            center_articles=center,
            right_articles=right,
            total_found=len(articles)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Fact-Checking
# ============================================================

@app.get(
    "/api/fact-checks/{article_id}",
    response_model=List[FactCheck]
)
async def get_article_fact_checks(article_id: str):
    """
    Get fact-checks associated with an article.
    """
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


# ============================================================
# Include API Routers
# ============================================================

app.include_router(
    articles.router,
    prefix="/articles",
    tags=["articles"]
)
app.include_router(
    users.router,
    prefix="/users",
    tags=["users"]
)
app.include_router(
    sources.router,
    prefix="/sources",
    tags=["sources"]
)


# ============================================================
# Static Files
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon if available."""
    file_path = os.path.join(BASE_DIR, "..", "static", "favicon.ico")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"status": "no icon"}
