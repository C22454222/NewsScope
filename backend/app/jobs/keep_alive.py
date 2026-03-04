"""
NewsScope keep-alive pinger.

Pings /health every 14 minutes to prevent Render free tier spin-down.
Also triggers a mini analysis batch to keep scoring resumable across
instance rotations.

Flake8: 0 errors/warnings.
"""

import requests
from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import settings

BACKEND_URL = settings.RENDER_EXTERNAL_URL


def keep_alive() -> None:
    """
    Ping health endpoint to prevent spin-down, then trigger a
    small analysis batch (3 articles) so analysis is resumable
    across instance rotations.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if response.status_code == 200:
            print("Keep-alive ping successful")
        else:
            print(f"Keep-alive ping failed: {response.status_code}")
    except Exception as exc:
        print(f"Keep-alive ping error: {exc}")
        return

    # Trigger a mini analysis batch — fire-and-forget, 5s timeout
    # If instance was rotated mid-analysis, this resumes from DB state
    try:
        requests.post(
            f"{BACKEND_URL}/internal/analyze-batch",
            timeout=5,
        )
        print("Keep-alive: analysis batch triggered")
    except Exception as exc:
        print(f"Keep-alive: analysis batch trigger failed: {exc}")


def start_keep_alive() -> None:
    """Start the background scheduler for keep-alive pings."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(keep_alive, "interval", minutes=14)
    scheduler.start()
    print("Keep-alive scheduler started (pinging every 14 minutes)")
