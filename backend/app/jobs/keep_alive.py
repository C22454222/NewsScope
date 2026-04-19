"""
NewsScope keep-alive pinger.

Pings /health every 10 minutes to prevent Render free tier spin-down.
Pings all three HF Spaces every 8 minutes to prevent Gradio cold starts.
Free Spaces sleep after roughly 15 minutes of inactivity, causing
30-90 second cold start delays on the next analysis cycle.

Retry policy: up to 3 attempts with exponential back-off on 500/502/
503/504 responses. A non-200 after all retries is logged and swallowed
so APScheduler never pauses the job permanently.
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

# Module-level guard prevents duplicate schedulers if start_keep_alive
# is called more than once during lifespan or worker initialisation.
_scheduler: BackgroundScheduler | None = None

# Shared retry policy for all ping requests.
# Retries up to 3 times on transient 5xx errors with exponential
# back-off (2s, 4s, 8s). raise_on_status=False returns the last
# response on exhaustion rather than raising so callers decide.
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
    both http:// and https://. Using the session as a context manager
    in each job ensures the socket is released after every ping.
    """
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY_POLICY)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def keep_alive() -> None:
    """
    Ping /health to prevent Render spin-down.

    Up to 3 retries on 5xx responses via the session adapter. A
    non-200 after all retries is logged but never re-raised -- raising
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
                    f"after retries -- Render may still be starting up"
                )
    except Exception as exc:
        print(f"Keep-alive ping error (non-fatal): {exc}")


def ping_spaces() -> None:
    """
    Ping all three HF Spaces to prevent Gradio cold starts.

    Free Spaces sleep after roughly 15 minutes of inactivity. A cold
    start forces the Space to reload its model, taking 30-90 seconds.
    Pinging every 8 minutes keeps all three warm so analysis calls
    complete in 2-5 seconds rather than timing out.

    A plain GET to the Space root is sufficient -- no inference is
    triggered. A 1 second gap between pings avoids hitting the same
    HF edge node in rapid succession. All failures are swallowed.
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


def _job_error_listener(event) -> None:
    """
    Catch any exception that escapes a job function.

    Without this listener, APScheduler silently pauses the offending
    job after an unhandled exception and it never fires again until
    the process restarts. This logs and discards the error so the
    scheduler keeps the job on its normal interval.
    """
    print(
        f"Keep-alive scheduler job error (non-fatal, "
        f"job will continue on next interval): {event.exception}"
    )


def start_keep_alive() -> None:
    """
    Start the background scheduler for keep-alive pings.

    Job defaults applied at scheduler level:
      coalesce=True -- if a ping fires late, run it once not many times
      max_instances=1 -- never run the same job twice concurrently
      misfire_grace_time=60 -- fire a missed trigger if within 60s

    Idempotent -- safe to call multiple times; only the first call
    has any effect. Subsequent calls are no-ops.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        print(
            "Keep-alive scheduler already running -- "
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
    _scheduler.add_job(keep_alive, "interval", minutes=10)
    _scheduler.add_job(ping_spaces, "interval", minutes=8)
    _scheduler.start()

    print(
        "Keep-alive scheduler started "
        "(Render ping every 10 min, Space pings every 8 min)"
    )
