import os
from dotenv import load_dotenv


load_dotenv()


class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")
    PROJECT_NAME: str = "NewsScope API"


settings = Settings()
