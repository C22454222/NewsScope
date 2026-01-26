# app/main.py
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse

from app.routes import articles, users, sources
from app.core.scheduler import start_scheduler
from app.jobs.ingestion import run_ingestion_cycle
from app.jobs.analysis import analyze_unscored_articles
from app.jobs.archiving import archive_old_articles
from app.jobs.keep_alive import start_keep_alive


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


app = FastAPI(title="NewsScope API", lifespan=lifespan)


@app.get("/")
def root():
    """Root endpoint with welcome message."""
    return {
        "message": (
            "Welcome to NewsScope API. Try /health to check status."
        ),
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.head("/")
def root_head():
    """HEAD request for root endpoint (used by monitoring tools)."""
    return {}


@app.get("/health")
def health():
    """
    Health check endpoint for uptime monitoring.
    Used by Render and keep-alive scheduler.
    """
    return {"status": "ok"}


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


# Include API routers
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon if available."""
    file_path = os.path.join(BASE_DIR, "..", "static", "favicon.ico")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"status": "no icon"}
