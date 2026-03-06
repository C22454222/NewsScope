"""
NewsScope application configuration.

All environment variables are centralised here via a pydantic-settings
Settings class. Import `settings` — never os.getenv() directly elsewhere.

Flake8: 0 errors/warnings.
"""

import os

from dotenv import load_dotenv

# Load .env for local development — no-op in production (vars already set)
load_dotenv()


class Settings:
    """
    Application-wide configuration loaded from environment variables.

    Raises ValueError at startup if required variables are missing so
    the service fails fast rather than crashing mid-request.
    """

    # ── Supabase ──────────────────────────────────────────────────────────────
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

    # ── NewsAPI ───────────────────────────────────────────────────────────────
    NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")

    # ── HuggingFace ───────────────────────────────────────────────────────────
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
    HF_SENTIMENT_MODEL: str = os.getenv(
        "HF_SENTIMENT_MODEL",
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
    HF_BIAS_MODEL: str = os.getenv(
        "HF_BIAS_MODEL",
        # MoritzLaurer/mDeBERTa-v3-base-mnli-xnli — confirmed on HF
        # Inference API. Zero-shot NLI, 0.3B params. Replaces
        # facebook/bart-large-mnli (0.4B) which caused OOM via 90s
        # socket stalls on cold start. Same candidate_labels payload
        # shape and response format. DeBERTa-v3 outperforms BART on
        # NLI benchmarks. Multilingual — handles RTÉ, Irish Times,
        # Euronews content better than English-only BART.
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )
    HF_GENERAL_BIAS_MODEL: str = os.getenv(
        "HF_GENERAL_BIAS_MODEL",
        # valurank/distilroberta-bias — confirmed on HF Inference API.
        # Returns BIASED / UNBIASED. 3.79k downloads.
        "valurank/distilroberta-bias",
    )

    # ── Firebase ──────────────────────────────────────────────────────────────
    FIREBASE_SERVICE_ACCOUNT: str = os.getenv(
        "FIREBASE_SERVICE_ACCOUNT", ""
    )

    # ── Archiving ─────────────────────────────────────────────────────────────
    ARCHIVE_DAYS: int = int(os.getenv("ARCHIVE_DAYS", "30"))
    ARCHIVE_BUCKET: str = os.getenv(
        "ARCHIVE_BUCKET", "articles-archive"
    )

    # ── App ───────────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "NewsScope API"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    RENDER_EXTERNAL_URL: str = os.getenv(
        "RENDER_EXTERNAL_URL",
        "https://newsscope-backend.onrender.com",
    )

    def __init__(self) -> None:
        if not self.SUPABASE_URL:
            raise ValueError("SUPABASE_URL not set")
        if not self.SUPABASE_KEY:
            raise ValueError("SUPABASE_KEY not set")


# Single settings instance imported by all other modules
settings = Settings()
