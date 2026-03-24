"""
NewsScope keep-alive pinger.

Pings /health every 14 minutes to prevent Render free tier spin-down.
Pings all three HF Spaces every 10 minutes to prevent Gradio cold
starts — free Spaces sleep after ~15 minutes of inactivity, causing
30-90s cold start delays on the next analysis cycle.

All three Spaces are pinged in a single scheduled job so they share
the same 10-minute interval and a single scheduler thread.

Flake8: 0 errors/warnings.
"""

import requests
from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import settings

BACKEND_URL = settings.RENDER_EXTERNAL_URL
POLITICAL_BIAS_SPACE = settings.HF_POLITICAL_BIAS_SPACE
SENTIMENT_SPACE = settings.HF_SENTIMENT_SPACE
GENERAL_BIAS_SPACE = settings.HF_GENERAL_BIAS_SPACE

# Module-level guard — prevents duplicate schedulers if lifespan
# or Uvicorn worker init somehow calls start_keep_alive() twice.
_scheduler: BackgroundScheduler | None = None


def keep_alive() -> None:
    """
    Ping /health to prevent Render spin-down.
    Response closed immediately to release socket memory.
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


def ping_spaces() -> None:
    """
    Ping all three HF Spaces to prevent Gradio cold starts.

    Free Spaces sleep after ~15 minutes of inactivity — a cold start
    forces the Space to reload its model which takes 30-90s. Pinging
    every 10 minutes keeps all three warm so analysis calls complete
    in 2-5s instead of timing out or stalling.

    A simple GET to the Space root is enough — no inference triggered.
    Failures are silently swallowed; a sleeping Space will just wake
    on the next analysis cycle instead.
    """
    for name, url in (
        ("Political bias", POLITICAL_BIAS_SPACE),
        ("Sentiment", SENTIMENT_SPACE),
        ("General bias", GENERAL_BIAS_SPACE),
    ):
        if not url:
            continue
        try:
            response = requests.get(url, timeout=15)
            response.close()
            print(f"{name} Space ping successful")
        except Exception as exc:
            print(f"{name} Space ping error: {exc}")


def start_keep_alive() -> None:
    """
    Start the background scheduler for keep-alive pings.

    Guard prevents duplicate schedulers — safe to call multiple times;
    only the first call has any effect. Subsequent calls are no-ops.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        print(
            "Keep-alive scheduler already running — skipping duplicate start."
        )
        return

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(keep_alive, "interval", minutes=14)
    _scheduler.add_job(ping_spaces, "interval", minutes=10)
    _scheduler.start()
    print(
        "Keep-alive scheduler started "
        "(Render ping every 14 min, Space pings every 10 min)"
    )
