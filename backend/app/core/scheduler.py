"""
NewsScope background scheduler.

Uses APScheduler BackgroundScheduler to run periodic jobs in a
daemon thread alongside the FastAPI/uvicorn event loop.
"""

from apscheduler.schedulers.background import BackgroundScheduler

# Global singleton started once in main.py lifespan and shared by all
# job registration calls throughout the application.
scheduler = BackgroundScheduler(
    job_defaults={
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": 60,
    }
)


def start_scheduler() -> None:
    """
    Start the global scheduler if it is not already running.

    Called once during FastAPI startup so ingestion, analysis, and
    archiving jobs run on fixed intervals without blocking the async
    event loop. Job defaults (coalesce=True, max_instances=1) are
    applied at scheduler level so every registered job inherits them,
    preventing overlapping executions automatically.
    """
    if not scheduler.running:
        scheduler.start()
