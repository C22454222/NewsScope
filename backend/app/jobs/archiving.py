# app/jobs/archiving.py
import json
import os
from datetime import datetime, timedelta
from app.db.supabase import supabase


ARCHIVE_DAYS = int(os.getenv("ARCHIVE_DAYS", "30"))
BUCKET = os.getenv("ARCHIVE_BUCKET", "articles-archive")


def archive_old_articles():
    """
    Archive ALL articles older than 30 days and delete them.
    Handles unlimited articles with pagination.
    """
    cutoff = (
        datetime.utcnow() - timedelta(days=ARCHIVE_DAYS)
    ).isoformat()

    print(f"🗄️ Archiving articles published before {cutoff}...")

    # Count total articles to archive
    count_response = (
        supabase.table("articles")
        .select("id", count="exact")
        .lte("published_at", cutoff)
        .execute()
    )
    total = count_response.count

    if total == 0:
        print("ℹ️ No old articles to archive.")
        return

    print(f"📦 Found {total} articles to archive...")

    archived_count = 0
    archived_ids = []
    offset = 0
    batch_size = 500

    while offset < total:
        print(f"Processing batch {offset//batch_size + 1}...")

        rows = (
            supabase.table("articles")
            .select("*")
            .lte("published_at", cutoff)
            .range(offset, offset + batch_size - 1)
            .execute()
            .data
        )

        if not rows:
            break

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
                archived_ids.append(row['id'])
            except Exception as e:
                print(f"❌ Failed to archive {key}: {e}")

        offset += batch_size

    print(f"✅ Archived {archived_count}/{total} articles")

    # Delete archived articles in batches
    if archived_ids:
        print(f"🗑️ Deleting {len(archived_ids)} archived articles...")

        for i in range(0, len(archived_ids), 1000):
            batch = archived_ids[i:i + 1000]
            try:
                supabase.table("articles").delete().in_("id", batch).execute()
                print(f"Deleted batch {i//1000 + 1}: {len(batch)}")
            except Exception as e:
                print(f"❌ Failed to delete batch: {e}")

        print(f"✅ Deleted all {len(archived_ids)} archived articles")
