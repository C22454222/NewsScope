"""
NewsScope Supabase client.

Provides a singleton client shared across all background jobs and
routes. Instantiated once at import time and never recreated per
request to avoid redundant connection overhead.
"""

from dotenv import load_dotenv
from supabase import Client, create_client

from app.core.config import settings

# Load .env for local development. No-op in production where
# environment variables are already present.
load_dotenv()

# Shared singleton -- all modules import this instance directly.
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
