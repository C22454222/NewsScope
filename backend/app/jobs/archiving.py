# app/jobs/archiving.py
import json
import os
from datetime import datetime, timedelta
from app.db.supabase import supabase

ARCHIVE_DAYS = int(os.getenv("ARCHIVE_DAYS", "30"))
BUCKET = os.getenv("ARCHIVE_BUCKET", "articles-archive")


def archive_old_articles():
    """
    Archive articles older than 30 days daily.
    Run at 2 AM UTC (off-peak).
    """
    cutoff = (
        datetime.utcnow() - timedelta(days=ARCHIVE_DAYS)
    ).isoformat()

    print(f"🗄️ Archiving articles published before {cutoff}...")

    rows = (
        supabase.table("articles")
        .select("*")
        .lte("published_at", cutoff)
        .execute()
        .data
    )

    if not rows:
        print("ℹ️ No old articles to archive.")
        return

    print(f"📦 Archiving {len(rows)} articles...")

    archived_count = 0
    for row in rows:
        key = f"{row['id']}.json"
        content_str = json.dumps(row, default=str)
        content_bytes = content_str.encode("utf-8")

        try:
            supabase.storage.from_(BUCKET).upload(
                path=key,
                file=content_bytes,
                file_options={
                    "content-type": "application/json",
                    "upsert": "true"
                }
            )
            archived_count += 1
        except Exception as e:
            print(f"❌ Failed to archive {key}: {e}")

    print(f"✅ Archived {archived_count}/{len(rows)} articles")
