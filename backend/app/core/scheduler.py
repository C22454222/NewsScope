# app/core/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler

# Global background scheduler for periodic jobs
scheduler = BackgroundScheduler()


def start_scheduler():
    """
    Start the global scheduler if it is not already running.

    This is called once during FastAPI startup so that
    ingestion, analysis, and archiving jobs can run on
    a fixed interval in the background.
    """
    if not scheduler.running:
        scheduler.start()
