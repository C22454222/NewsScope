# app/jobs/archiving.py
import json
import os
from datetime import datetime, timedelta, timezone
from app.db.supabase import supabase


ARCHIVE_DAYS = int(os.getenv("ARCHIVE_DAYS", "0"))
BUCKET = os.getenv("ARCHIVE_BUCKET", "articles-archive")


def archive_old_articles():
    """
    Archive ALL articles older than 30 days and delete them from the database.

    Example: If today is 2026-01-26, archives articles published
    BEFORE 2025-12-27 (exactly 30 days ago).
    """
    # Calculate cutoff date: 30 days ago from now
    now = datetime.now(timezone.utc)
    cutoff_datetime = now - timedelta(days=ARCHIVE_DAYS)
    cutoff = cutoff_datetime.isoformat()

    # DEBUG: Verify the calculation is working
    print(f"🔧 DEBUG: now={now}, cutoff_datetime={cutoff_datetime}, ARCHIVE_DAYS={ARCHIVE_DAYS}")

    print(f"🗄️ Today: {now.isoformat()}")
    print(f"🗄️ Cutoff (30 days ago): {cutoff}")
    print(f"🗄️ Archiving articles published BEFORE {cutoff}...")

    # Step 1: Count total articles to archive
    try:
        count_response = (
            supabase.table("articles")
            .select("id", count="exact")
            .lt("published_at", cutoff)
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
            # Fetch batch of old articles
            rows = (
                supabase.table("articles")
                .select("*")
                .lt("published_at", cutoff)
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
                archived_ids.append(str(article_id))
            except Exception as e:
                print(f"❌ Failed to archive article {article_id}: {e}")

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

        # Delete in smaller batches of 100
        for i in range(0, len(archived_ids), 100):
            batch = archived_ids[i:i + 100]
            batch_num = i // 100 + 1

            try:
                # Delete each article individually for reliability
                for article_id in batch:
                    supabase.table("articles").delete().eq("id", article_id).execute()

                deleted_total += len(batch)
                print(
                    f"✅ Deleted batch {batch_num}: {len(batch)} articles "
                    f"(total: {deleted_total}/{len(archived_ids)})"
                )
            except Exception as e:
                print(f"❌ Failed to delete batch {batch_num}: {e}")

        print(
            f"✅ Cleanup complete! "
            f"Archived: {archived_count}, Deleted: {deleted_total}"
        )
    else:
        print("ℹ️ No articles were successfully archived, skipping deletion")
