"""
NewsScope background scheduler.

Uses APScheduler BackgroundScheduler — runs jobs in a daemon thread
alongside the FastAPI/uvicorn event loop.

Flake8: 0 errors/warnings.
"""

from apscheduler.schedulers.background import BackgroundScheduler

# Global singleton — imported and started once in main.py lifespan
scheduler = BackgroundScheduler(
    job_defaults={
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": 60,
    }
)


def start_scheduler() -> None:
    """
    Start the global scheduler if not already running.

    Called once during FastAPI startup so that ingestion,
    analysis, and archiving jobs run on fixed intervals
    in the background without blocking the async event loop.

    Job defaults (coalesce=True, max_instances=1) are set at
    scheduler level so every job registered on this instance
    inherits them — prevents overlapping triggers automatically.
    """
    if not scheduler.running:
        scheduler.start()
