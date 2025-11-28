# app/core/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env into process environment
load_dotenv()


class Settings:
    # Base Supabase configuration used across the backend
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")
    # Simple identifier used in logs and metadata
    PROJECT_NAME: str = "NewsScope API"


# Single settings instance imported by other modules
settings = Settings()
