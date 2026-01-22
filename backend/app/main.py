# app/main.py
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse

from app.routes import articles, users, sources
from app.core.scheduler import start_scheduler, add_job
from app.jobs.ingestion import run_ingestion_cycle
from app.jobs.analysis import analyze_unscored_articles
from app.jobs.archiving import archive_old_articles
from app.jobs.keep_alive import start_keep_alive


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context with immediate startup jobs.
    """
    # Start the APScheduler for background jobs
    start_scheduler()

    # Add scheduled jobs
    add_job(run_ingestion_cycle, minutes=30)
    add_job(analyze_unscored_articles, minutes=5)
    add_job(archive_old_articles, minutes=1440)

    # Start keep-alive scheduler in production
    env = os.getenv("ENVIRONMENT", "development")
    print(f"🌍 Environment: {env}")
    if env == "production":
        start_keep_alive()
    else:
        print("ℹ️ Keep-alive disabled in development")

    # Run startup jobs
    asyncio.create_task(_run_startup_jobs())

    yield


async def _run_startup_jobs():
    """
    Run critical jobs immediately when server wakes up from sleep.
    This ensures Render free tier servers process backlog instantly.
    """
    print("🚀 Server startup: Running ingestion + analysis...")
    try:
        await asyncio.to_thread(run_ingestion_cycle)
        print("✅ Startup ingestion complete")

        await asyncio.to_thread(analyze_unscored_articles)
        print("✅ Startup analysis complete")
    except Exception as e:
        print(f"⚠️ Startup jobs failed: {e}")


app = FastAPI(title="NewsScope API", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "message": "Welcome to NewsScope API. Try /health to check status."
    }


@app.head("/")
def root_head():
    return {}


@app.get("/health")
def health():
    """
    Health check endpoint for uptime monitoring.
    """
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


app.include_router(
    articles.router,
    prefix="/articles",
    tags=["articles"]
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
