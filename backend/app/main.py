# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
import os

from app.routes import articles, users, sources
from app.core.scheduler import start_scheduler, add_job
from app.jobs.ingestion import run_ingestion_cycle
from app.jobs.analysis import analyze_unscored_articles
from app.jobs.archiving import archive_old_articles


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    start_scheduler()
    add_job(run_ingestion_cycle, minutes=30)            # fetch new articles
    add_job(analyze_unscored_articles, minutes=60)      # sentiment/bias analysis
    add_job(archive_old_articles, minutes=1440)         # daily archiving

    yield   # <-- control passes to the application

    # Shutdown logic (optional)
    # scheduler.shutdown(wait=False)


app = FastAPI(title="NewsScope API", lifespan=lifespan)


# Root route
@app.get("/")
def root():
    return {
        "message": "Welcome to NewsScope API. Try /health to check status."
    }


# HEAD route (for Render probes)
@app.head("/")
def root_head():
    return {}


# Health check route
@app.get("/health")
def health():
    return {"status": "ok"}


# Debug route to trigger ingestion manually
@app.post("/debug/ingest")
async def debug_ingest():
    run_ingestion_cycle()
    return {"status": "ingestion triggered"}


# Routers
app.include_router(articles.router, prefix="/articles", tags=["articles"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(sources.router, prefix="/sources", tags=["sources"])


# Favicon route
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join(BASE_DIR, "..", "static", "favicon.ico"))
