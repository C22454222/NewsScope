"""
NewsScope Supabase client.

Singleton client shared across all jobs and routes.
Instantiated once at import time — never recreated per request.

Flake8: 0 errors/warnings.
"""

from dotenv import load_dotenv
from supabase import Client, create_client

from app.core.config import settings

# Load .env for local development — no-op in production
load_dotenv()

# Shared singleton — all modules import this instance directly
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
