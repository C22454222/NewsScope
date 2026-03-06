"""
NewsScope keep-alive pinger.

Pings /health every 14 minutes to prevent Render free tier spin-down.
Analysis is handled exclusively by APScheduler (every 5 minutes) —
keep-alive no longer triggers analysis batches to avoid race conditions
with the _heavy_job_running lock during ingestion.

Flake8: 0 errors/warnings.
"""

import requests
from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import settings

BACKEND_URL = settings.RENDER_EXTERNAL_URL


def keep_alive() -> None:
    """
    Ping health endpoint to prevent spin-down.
    Response closed immediately to release socket memory.
    Analysis scheduling is handled by APScheduler — not triggered
    here to prevent race conditions with ingestion memory lock.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if response.status_code == 200:
            print("Keep-alive ping successful")
        else:
            print(f"Keep-alive ping failed: {response.status_code}")
        response.close()
    except Exception as exc:
        print(f"Keep-alive ping error: {exc}")


def start_keep_alive() -> None:
    """Start the background scheduler for keep-alive pings."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(keep_alive, "interval", minutes=14)
    scheduler.start()
    print("Keep-alive scheduler started (pinging every 14 minutes)")
