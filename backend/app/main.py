# app/main.py
from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
import os
import asyncio

from app.routes import articles, users, sources
from app.core.scheduler import start_scheduler, add_job
from app.jobs.ingestion import run_ingestion_cycle
from app.jobs.analysis import analyze_unscored_articles
from app.jobs.archiving import archive_old_articles


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context.

    Used to hook into FastAPI startup/shutdown events so that
    background jobs (ingestion, analysis, archiving) can be
    scheduled when the service starts.
    """
    # Startup logic
    start_scheduler()

    # Fetch new articles on a fixed interval
    add_job(run_ingestion_cycle, minutes=30)

    # Enrich articles with sentiment and bias scores
    add_job(analyze_unscored_articles, minutes=60)

    # Archive older content daily
    add_job(archive_old_articles, minutes=1440)

    # --- CRITICAL FIX FOR RENDER FREE TIER ---
    # Run analysis IMMEDIATELY on startup in the background.
    # This ensures that if the server slept and just woke up,
    # it processes any pending articles right away instead of waiting 60 mins.
    asyncio.create_task(_run_startup_analysis())

    # Yield control back to FastAPI
    yield

    # Shutdown logic (kept simple for this prototype)
    # scheduler.shutdown(wait=False)


async def _run_startup_analysis():
    """Helper to run analysis asynchronously on startup without blocking."""
    print("🚀 Server Startup: Checking for unscored articles immediately...")
    # We wrap this because analyze_unscored_articles might be synchronous or blocking
    try:
        analyze_unscored_articles()
    except Exception as e:
        print(f"⚠️ Startup analysis failed: {e}")


# Main FastAPI application instance
app = FastAPI(title="NewsScope API", lifespan=lifespan)


@app.get("/")
def root():
    """
    Simple root endpoint used for smoke testing.

    Visible when opening the Render URL in a browser.
    """
    return {
        "message": "Welcome to NewsScope API. Try /health to check status."
    }


@app.head("/")
def root_head():
    """
    Lightweight HEAD endpoint for platform health probes.

    Render uses this to verify that the service is online.
    """
    return {}


@app.get("/health")
def health():
    """
    Health check endpoint for uptime monitoring.

    Called by cron-job.org to keep the Render free tier awake.
    """
    return {"status": "ok"}


@app.post("/debug/ingest")
async def debug_ingest(background_tasks: BackgroundTasks):
    """
    Manually trigger a single ingestion run.

    Useful during development and debugging without
    waiting for the scheduler interval.
    """
    # Run in background so the request doesn't time out
    background_tasks.add_task(run_ingestion_cycle)
    return {"status": "ingestion triggered in background"}


@app.post("/debug/analyze")
async def debug_analyze(background_tasks: BackgroundTasks):
    """
    Manually trigger the analysis job.
    Force-run bias/sentiment scoring immediately.
    """
    background_tasks.add_task(analyze_unscored_articles)
    return {"status": "analysis triggered in background"}


# Register route modules with path prefixes and tags
app.include_router(articles.router, prefix="/articles", tags=["articles"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(sources.router, prefix="/sources", tags=["sources"])


# Resolve path to static assets (e.g. favicon)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/favicon.ico")
async def favicon():
    """
    Serve a small favicon so browsers do not log 404s.
    """
    # Basic check to prevent errors if file is missing locally
    file_path = os.path.join(BASE_DIR, "..", "static", "favicon.ico")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"status": "no icon"}
