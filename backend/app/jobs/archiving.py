# app/jobs/archiving.py
import json
import os
from datetime import datetime, timedelta
from app.db.supabase import supabase


ARCHIVE_DAYS = int(os.getenv("ARCHIVE_DAYS", "30"))
BUCKET = os.getenv("ARCHIVE_BUCKET", "articles-archive")


def archive_old_articles():
    cutoff = (datetime.utcnow() - timedelta(days=ARCHIVE_DAYS)).isoformat()
    # fetch metadata of old articles
    rows = supabase.table("articles").select("*").lte("published_at", cutoff).execute().data
    if not rows:
        return
    # write batch snapshot to storage (one file per article for simplicity)
    for row in rows:
        key = f"{row['id']}.json"
        content = json.dumps(row, default=str)
        supabase.storage.from_(BUCKET).upload(key, content)  # ensure bucket exists
    # optional: delete from table or mark as archived
    # supabase.table("articles").delete().lte("published_at", cutoff).execute()
