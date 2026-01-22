# app/jobs/keep_alive.py
import requests
import os
from apscheduler.schedulers.background import BackgroundScheduler

BACKEND_URL = os.getenv("RENDER_EXTERNAL_URL", "https://newsscope-backend.onrender.com")


def keep_alive():
    """Ping the health endpoint every 14 minutes to prevent spindown."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Keep-alive ping successful")
        else:
            print(f"❌ Keep-alive failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Keep-alive error: {e}")


def start_keep_alive():
    """Start the background scheduler for keep-alive pings."""
    scheduler = BackgroundScheduler()
    # Run every 14 minutes (840 seconds)
    scheduler.add_job(keep_alive, 'interval', minutes=14)
    scheduler.start()
    print("🚀 Keep-alive scheduler started (pinging every 14 minutes)")
