"""
NewsScope application configuration.

All environment variables are centralised here via a Settings class.
Import the module-level 'settings' instance everywhere -- never call
os.getenv() directly in other modules.
"""

import os

from dotenv import load_dotenv

# Load .env file for local development. No-op when variables are
# already present in the environment (e.g. on Render in production).
load_dotenv()


class Settings:
    """
    Application-wide configuration loaded from environment variables.

    Raises ValueError at startup if required variables are absent so
    the service fails fast rather than crashing mid-request.
    """

    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

    # NewsAPI
    NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")

    # HuggingFace Inference API
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")

    # Zero-shot NLI model used for topic/category classification at
    # ingestion time. Called infrequently so credit cost is negligible.
    HF_BIAS_MODEL: str = os.getenv(
        "HF_BIAS_MODEL",
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )

    # HuggingFace Spaces endpoints (free CPU, no credit consumption).
    # Fine-tuned RoBERTa trained on 37,554 AllSides articles.
    # 87.3% accuracy / macro F1. Gradio 5.x endpoint:
    # /gradio_api/call/classify_bias
    HF_POLITICAL_BIAS_SPACE: str = os.getenv(
        "HF_POLITICAL_BIAS_SPACE",
        "https://c22454222-political-bias-api.hf.space",
    )

    # distilbert-base-uncased-finetuned-sst-2-english wrapped in Gradio.
    # Gradio 5.x endpoint: /gradio_api/call/lambda
    # Returns list of {label, score} dicts for POSITIVE / NEGATIVE.
    HF_SENTIMENT_SPACE: str = os.getenv(
        "HF_SENTIMENT_SPACE",
        "https://c22454222-sentiment.hf.space",
    )

    # valurank/distilroberta-bias wrapped in Gradio.
    # Gradio 5.x endpoint: /gradio_api/call/lambda
    # Returns list of {label, score} dicts for BIASED / UNBIASED.
    HF_GENERAL_BIAS_SPACE: str = os.getenv(
        "HF_GENERAL_BIAS_SPACE",
        "https://c22454222-general-bias.hf.space",
    )

    # Firebase
    FIREBASE_SERVICE_ACCOUNT: str = os.getenv(
        "FIREBASE_SERVICE_ACCOUNT", ""
    )

    # Archiving
    ARCHIVE_DAYS: int = int(os.getenv("ARCHIVE_DAYS", "7"))
    ARCHIVE_BUCKET: str = os.getenv(
        "ARCHIVE_BUCKET", "articles-archive"
    )

    # Application metadata
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


# Single shared instance imported by all other modules.
settings = Settings()
