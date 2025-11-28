# app/db/supabase.py
import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables so SUPABASE_URL and SUPABASE_KEY are available
load_dotenv()

# Connection details for the Supabase project
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Shared Supabase client used by the data access layer and jobs
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
