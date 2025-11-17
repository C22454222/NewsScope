# app/main.py
from fastapi import FastAPI
from app.routes import articles, users, sources
from app.core.scheduler import start_scheduler, add_job
from app.jobs.ingestion import run_ingestion_cycle
from app.jobs.analysis import analyze_unscored_articles
from app.jobs.archiving import archive_old_articles


app = FastAPI(title="NewsScope API")


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


# Routers
app.include_router(articles.router, prefix="/articles", tags=["articles"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(sources.router, prefix="/sources", tags=["sources"])

# Scheduler startup
@app.on_event("startup")
def startup_event():
    start_scheduler()
    add_job(run_ingestion_cycle, minutes=30)       # fetch new articles
    add_job(analyze_unscored_articles, minutes=60) # sentiment/bias analysis
    add_job(archive_old_articles, minutes=1440)    # daily archiving
