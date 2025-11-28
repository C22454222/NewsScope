# app/jobs/archiving.py
import json
import os
from datetime import datetime, timedelta
from app.db.supabase import supabase

# Number of days after which articles are eligible for archiving
ARCHIVE_DAYS = int(os.getenv("ARCHIVE_DAYS", "30"))
# Supabase storage bucket used to store archived article snapshots
BUCKET = os.getenv("ARCHIVE_BUCKET", "articles-archive")


def archive_old_articles():
    """
    Archive older articles to Supabase storage.

    This job finds articles older than ARCHIVE_DAYS, writes a JSON
    snapshot of each record into object storage, and leaves a hook
    to optionally delete or mark rows as archived.
    """
    cutoff = (datetime.utcnow() - timedelta(days=ARCHIVE_DAYS)).isoformat()

    # Fetch all articles with a published_at older than the cutoff
    rows = (
        supabase.table("articles")
        .select("*")
        .lte("published_at", cutoff)
        .execute()
        .data
    )
    if not rows:
        return

    # Write one JSON file per article into the archive bucket
    for row in rows:
        key = f"{row['id']}.json"
        content = json.dumps(row, default=str)
        # Assumes the bucket exists and the service key can write to it
        supabase.storage.from_(BUCKET).upload(key, content)

    # Optional clean-up step to remove or flag archived rows
    # supabase.table("articles").delete().lte("published_at", cutoff).execute()
