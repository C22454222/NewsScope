"""
NewsScope keep-alive pinger.

Pings /health every 14 minutes to prevent Render free tier spin-down.
Pings HF Political Bias Space every 10 minutes to prevent Gradio cold
starts — free Spaces sleep after ~15 minutes of inactivity, causing
30-90s cold start delays that exceed _SPACES_TIMEOUT in analysis.py.
Analysis is handled exclusively by APScheduler (every 5 minutes) —
keep-alive no longer triggers analysis batches to avoid race conditions
with the _heavy_job_running lock during ingestion.

Flake8: 0 errors/warnings.
"""

import requests
from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import settings

BACKEND_URL = settings.RENDER_EXTERNAL_URL
POLITICAL_BIAS_SPACE = settings.HF_POLITICAL_BIAS_SPACE

# Module-level guard — prevents duplicate schedulers if lifespan
# or Uvicorn worker init somehow calls start_keep_alive() twice.
_scheduler: BackgroundScheduler | None = None


def keep_alive() -> None:
    """
    Ping health endpoint to prevent Render spin-down.
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


def ping_political_bias_space() -> None:
    """
    GET the HF Space root to prevent Gradio cold starts.

    Free HF Spaces sleep after ~15 minutes of inactivity. A cold start
    forces the Space to reload the RoBERTa model (~1.3GB) which takes
    30-90s and exceeds _SPACES_TIMEOUT in analysis.py. Pinging every
    10 minutes keeps the Space warm so article scoring calls complete
    in 2-5s instead of timing out.

    A simple GET to the root is enough — no inference triggered.
    Failures are silently swallowed; a sleeping Space will just wake
    on the next analysis cycle instead.
    """
    try:
        response = requests.get(POLITICAL_BIAS_SPACE, timeout=10)
        response.close()
        print("Political bias Space ping successful")
    except Exception as exc:
        print(f"Political bias Space ping error: {exc}")


def start_keep_alive() -> None:
    """
    Start the background scheduler for keep-alive pings.

    Guard prevents duplicate schedulers — safe to call multiple times;
    only the first call has any effect. Subsequent calls are no-ops.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        print("Keep-alive scheduler already running — skipping duplicate start.")
        return

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(keep_alive, "interval", minutes=14)
    _scheduler.add_job(ping_political_bias_space, "interval", minutes=10)
    _scheduler.start()
    print(
        "Keep-alive scheduler started "
        "(Render ping every 14 min, Space ping every 10 min)"
    )
