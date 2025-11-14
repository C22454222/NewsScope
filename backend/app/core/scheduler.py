# app/core/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger


scheduler = BackgroundScheduler()


def start_scheduler():
    if not scheduler.running:
        scheduler.start()


def add_job(func, minutes: int = 30):
    scheduler.add_job(func, IntervalTrigger(minutes=minutes), max_instances=1, coalesce=True)
