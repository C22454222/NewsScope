"""
NewsScope keep-alive pinger.

Pings /health every 14 minutes to prevent Render free tier spin-down.
Pings all three HF Spaces every 10 minutes to prevent Gradio cold
starts — free Spaces sleep after ~15 minutes of inactivity, causing
30-90s cold start delays on the next analysis cycle.

All three Spaces are pinged in a single scheduled job so they share
the same 10-minute interval and a single scheduler thread.

Retry policy: up to 3 attempts with exponential back-off on 500/502/
503/504 responses. A non-200 after all retries is logged and swallowed
— it must never propagate or APScheduler will pause the job permanently.

Flake8: 0 errors/warnings.
"""

import time

import requests
from apscheduler.events import EVENT_JOB_ERROR
from apscheduler.schedulers.background import BackgroundScheduler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.core.config import settings


BACKEND_URL = settings.RENDER_EXTERNAL_URL
POLITICAL_BIAS_SPACE = settings.HF_POLITICAL_BIAS_SPACE
SENTIMENT_SPACE = settings.HF_SENTIMENT_SPACE
GENERAL_BIAS_SPACE = settings.HF_GENERAL_BIAS_SPACE


# Module-level guard — prevents duplicate schedulers if lifespan
# or Uvicorn worker init somehow calls start_keep_alive() twice.
_scheduler: BackgroundScheduler | None = None


# Retry policy shared by all pings.
# Retries up to 3 times on transient server errors (500/502/503/504)
# with exponential back-off: 2s → 4s → 8s between attempts.
# raise_on_status=False means exhausted retries return the last
# response rather than raising — caller decides what to log.
_RETRY_POLICY = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)


def _make_session() -> requests.Session:
    """
    Build a requests Session with the shared retry policy mounted on
    both http:// and https://. Using a context manager (with) in each
    job guarantees the socket is released after every ping attempt.
    """
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY_POLICY)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ── Ping jobs ────────────────────────────────────────────────────────────────


def keep_alive() -> None:
    """
    Ping /health to prevent Render spin-down.

    Up to 3 retries on 5xx via the session adapter before giving up.
    A non-200 after all retries is logged but never raised — raising
    would cause APScheduler to pause this job permanently.
    """
    try:
        with _make_session() as session:
            response = session.get(
                f"{BACKEND_URL}/health", timeout=10
            )
            response.close()
            if response.status_code == 200:
                print("Keep-alive ping successful")
            else:
                print(
                    f"Keep-alive ping returned {response.status_code} "
                    f"after retries — Render may still be starting up"
                )
    except Exception as exc:
        print(f"Keep-alive ping error (non-fatal): {exc}")


def ping_spaces() -> None:
    """
    Ping all three HF Spaces to prevent Gradio cold starts.

    Free Spaces sleep after ~15 minutes of inactivity — a cold start
    forces the Space to reload its model which takes 30-90s. Pinging
    every 10 minutes keeps all three warm so analysis calls complete
    in 2-5s instead of timing out or stalling.

    A simple GET to the Space root is enough — no inference triggered.
    A 1s gap between pings avoids hitting the same HF edge node twice
    in rapid succession. All failures are swallowed — a sleeping Space
    will just cold-start on the next analysis cycle instead.
    """
    for name, url in (
        ("Political bias", POLITICAL_BIAS_SPACE),
        ("Sentiment", SENTIMENT_SPACE),
        ("General bias", GENERAL_BIAS_SPACE),
    ):
        if not url:
            continue
        try:
            with _make_session() as session:
                response = session.get(url.rstrip("/"), timeout=30)
                response.close()
                print(f"{name} Space ping successful")
        except Exception as exc:
            print(f"{name} Space ping error (non-fatal): {exc}")
        time.sleep(1)


# ── APScheduler error listener ───────────────────────────────────────────────


def _job_error_listener(event) -> None:
    """
    Catch any exception that escapes a job function.

    Without this, APScheduler silently pauses the offending job after
    an unhandled exception — the job never fires again until the
    process restarts. This listener logs and discards the error so
    the scheduler keeps the job on its normal interval.
    """
    print(
        f"Keep-alive scheduler job error (non-fatal, "
        f"job will continue on next interval): {event.exception}"
    )


# ── Scheduler bootstrap ──────────────────────────────────────────────────────


def start_keep_alive() -> None:
    """
    Start the background scheduler for keep-alive pings.

    Job defaults:
      coalesce=True       — if a ping fires late, run it once not many times
      max_instances=1     — never run the same job twice concurrently
      misfire_grace_time  — if a trigger is missed by up to 60s, still fire it

    Guard prevents duplicate schedulers — safe to call multiple times;
    only the first call has any effect. Subsequent calls are no-ops.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        print(
            "Keep-alive scheduler already running — "
            "skipping duplicate start."
        )
        return

    _scheduler = BackgroundScheduler(
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 60,
        }
    )
    _scheduler.add_listener(_job_error_listener, EVENT_JOB_ERROR)
    _scheduler.add_job(keep_alive, "interval", minutes=14)
    _scheduler.add_job(ping_spaces, "interval", minutes=10)
    _scheduler.start()

    print(
        "Keep-alive scheduler started "
        "(Render ping every 14 min, Space pings every 10 min)"
    )
