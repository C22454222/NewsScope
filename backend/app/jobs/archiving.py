# app/jobs/archiving.py
import json
import os
from datetime import datetime, timedelta, timezone
from app.db.supabase import supabase


ARCHIVE_DAYS = int(os.getenv("ARCHIVE_DAYS", "30"))
BUCKET = os.getenv("ARCHIVE_BUCKET", "articles-archive")


def archive_old_articles():
    """
    Archive ALL articles older than 30 days and delete them.
    Uses pagination to handle unlimited articles.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=ARCHIVE_DAYS)
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

    # Process in batches using pagination
    while True:
        batch_num = offset // batch_size + 1
        print(
            f"Processing batch {batch_num} "
            f"(offset {offset}, expecting ~{min(batch_size, total - offset)} articles)..."
        )

        # CRITICAL: Filter by date AND use range
        rows = (
            supabase.table("articles")
            .select("*")
            .lte("published_at", cutoff)
            .order("published_at", desc=False)  # Ensure consistent ordering
            .range(offset, offset + batch_size - 1)
            .execute()
            .data
        )

        # Stop if no more rows
        if not rows:
            print(f"No more articles found at offset {offset}")
            break

        print(f"Retrieved {len(rows)} articles in this batch")

        # Archive each article in this batch
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

        offset += len(rows)

        # Stop if we've processed all expected articles
        if archived_count >= total:
            print(f"Reached expected total of {total} articles")
            break

    print(f"✅ Archived {archived_count}/{total} articles to storage")

    # Delete archived articles from database in batches
    if archived_ids:
        print(
            f"🗑️ Deleting {len(archived_ids)} archived articles "
            f"from database..."
        )

        deleted_total = 0
        for i in range(0, len(archived_ids), 1000):
            batch = archived_ids[i:i + 1000]
            batch_num = i // 1000 + 1

            try:
                supabase.table("articles").delete().in_("id", batch).execute()
                deleted_total += len(batch)
                print(
                    f"✅ Deleted batch {batch_num}: {len(batch)} articles "
                    f"(total: {deleted_total})"
                )
            except Exception as e:
                print(f"❌ Failed to delete batch {batch_num}: {e}")

        print(
            f"✅ Cleanup complete! "
            f"Deleted {deleted_total}/{len(archived_ids)} articles"
        )
