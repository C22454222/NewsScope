# app/core/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

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


def add_job(func, minutes: int = 30):
    """
    Register a new recurring job with the scheduler.

    Args:
        func: Callable with no arguments to be executed on schedule.
        minutes: Interval between runs. Defaults to 30 minutes.

    max_instances=1 ensures only one copy of the job runs at a time.
    coalesce=True means missed runs are combined into a single execution.
    """
    scheduler.add_job(
        func,
        IntervalTrigger(minutes=minutes),
        max_instances=1,
        coalesce=True,
    )
