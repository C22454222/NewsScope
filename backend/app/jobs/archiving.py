# app/jobs/archiving.py
import json
import os
from datetime import datetime, timedelta, timezone
from app.db.supabase import supabase


ARCHIVE_DAYS = int(os.getenv("ARCHIVE_DAYS", "30"))
BUCKET = os.getenv("ARCHIVE_BUCKET", "articles-archive")


def archive_old_articles():
    """
    Archive ALL articles older than 30 days and delete them from the database.
    Logic:
    - Today: 2026-01-23
    - Cutoff: 2025-12-24 (30 days ago)
    - Archives: All articles published BEFORE 2025-12-24
    Steps:
    1. Count articles older than 30 days
    2. Fetch them in batches of 500 (pagination)
    3. Upload each to Supabase storage bucket as JSON
    4. Delete successfully archived articles from database in batches of 1000
    """
    # Calculate cutoff date: exactly 30 days ago from now
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=ARCHIVE_DAYS)
    ).isoformat()

    print(f"🗄️ Archiving articles published before {cutoff}...")

    # Step 1: Count total articles to archive
    try:
        count_response = (
            supabase.table("articles")
            .select("id", count="exact")
            .lte("published_at", cutoff)
            .execute()
        )
        total = count_response.count
    except Exception as e:
        print(f"❌ Failed to count articles: {e}")
        return

    if total == 0:
        print("ℹ️ No old articles to archive.")
        return

    print(f"📦 Found {total} articles to archive...")

    archived_count = 0
    archived_ids = []
    offset = 0
    batch_size = 500

    # Step 2: Process articles in batches
    while True:
        batch_num = offset // batch_size + 1
        expected = min(batch_size, total - offset)
        print(
            f"📥 Processing batch {batch_num} "
            f"(offset {offset}, expecting ~{expected} articles)..."
        )

        try:
            # Fetch batch of old articles, ordered by date for consistency
            rows = (
                supabase.table("articles")
                .select("*")
                .lte("published_at", cutoff)
                .order("published_at", desc=False)
                .range(offset, offset + batch_size - 1)
                .execute()
                .data
            )
        except Exception as e:
            print(f"❌ Failed to fetch batch {batch_num}: {e}")
            break

        # Stop if no more rows
        if not rows:
            print(f"✅ No more articles at offset {offset}")
            break

        print(f"📄 Retrieved {len(rows)} articles in batch {batch_num}")

        # Step 3: Archive each article to storage
        for row in rows:
            article_id = row['id']
            key = f"{article_id}.json"
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
                archived_ids.append(article_id)
            except Exception as e:
                print(f"❌ Failed to archive article {article_id}: {e}")
                # Continue archiving other articles even if one fails

        offset += len(rows)

        # Stop if we've processed all expected articles
        if archived_count >= total:
            print(f"✅ Reached expected total of {total} articles")
            break

    print(f"✅ Archived {archived_count}/{total} articles to storage")

    # Step 4: Delete archived articles from database
    if archived_ids:
        print(
            f"🗑️ Deleting {len(archived_ids)} successfully archived articles "
            f"from database..."
        )

        deleted_total = 0

        # Delete in batches of 1000 (Supabase limit for .in_() operator)
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
                # Continue deleting other batches even if one fails

        print(
            f"✅ Cleanup complete! "
            f"Archived: {archived_count}, Deleted: {deleted_total}"
        )
    else:
        print("ℹ️ No articles were successfully archived, skipping deletion")
