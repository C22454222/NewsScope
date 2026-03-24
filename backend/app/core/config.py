"""
NewsScope application configurations.

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

    # Zero-shot NLI — still used for topic/category classification during
    # ingestion. Called rarely so Inference API credit cost is negligible.
    HF_BIAS_MODEL: str = os.getenv(
        "HF_BIAS_MODEL",
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )

    # ── HuggingFace Spaces (free CPU, unlimited requests, no credits) ─────────
    HF_POLITICAL_BIAS_SPACE: str = os.getenv(
        "HF_POLITICAL_BIAS_SPACE",
        # Fine-tuned RoBERTa — C22454222/political-bias-roberta.
        # Trained on 37,554 AllSides articles. 87.3% accuracy / macro F1.
        # Gradio 5.x endpoint: /gradio_api/call/classify_bias
        "https://c22454222-political-bias-api.hf.space",
    )
    HF_SENTIMENT_SPACE: str = os.getenv(
        "HF_SENTIMENT_SPACE",
        # distilbert-base-uncased-finetuned-sst-2-english wrapped in Gradio.
        # Gradio 5.x endpoint: /gradio_api/call/lambda
        # Returns list of {label, score} for POSITIVE / NEGATIVE.
        "https://c22454222-sentiment.hf.space",
    )
    HF_GENERAL_BIAS_SPACE: str = os.getenv(
        "HF_GENERAL_BIAS_SPACE",
        # valurank/distilroberta-bias wrapped in Gradio.
        # Gradio 5.x endpoint: /gradio_api/call/lambda
        # Returns list of {label, score} for BIASED / UNBIASED.
        "https://c22454222-general-bias.hf.space",
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
