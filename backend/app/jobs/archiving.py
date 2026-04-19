"""
NewsScope archiving job.

Archives articles older than ARCHIVE_DAYS to Supabase Storage,
then deletes them from the articles table. Runs daily at 03:00.
"""

import json
from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.db.supabase import supabase

ARCHIVE_DAYS = settings.ARCHIVE_DAYS
BUCKET = settings.ARCHIVE_BUCKET


def archive_old_articles() -> None:
    """
    Archive all articles older than ARCHIVE_DAYS to Supabase Storage
    and delete them from the database.

    Processes in batches of 500 to avoid memory spikes on Render.
    Deletes only articles that were successfully uploaded to prevent
    data loss on partial failures.
    """
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=ARCHIVE_DAYS)).isoformat()

    print(f"Today: {now.isoformat()}")
    print(f"Cutoff ({ARCHIVE_DAYS} days ago): {cutoff}")
    print(f"Archiving articles published before {cutoff}...")

    # Step 1: Count total articles that need archiving.
    try:
        count_response = (
            supabase.table("articles")
            .select("id", count="exact")
            .lt("published_at", cutoff)
            .execute()
        )
        total = count_response.count
    except Exception as exc:
        print(f"Failed to count articles: {exc}")
        return

    if total == 0:
        print("No old articles to archive.")
        return

    print(f"Found {total} articles to archive...")

    archived_count = 0
    archived_ids = []
    offset = 0
    batch_size = 500

    # Step 2: Fetch and upload articles in batches.
    while True:
        batch_num = offset // batch_size + 1
        expected = min(batch_size, total - offset)

        print(
            f"Processing batch {batch_num} "
            f"(offset {offset}, expecting ~{expected} articles)..."
        )

        try:
            rows = (
                supabase.table("articles")
                .select("*")
                .lt("published_at", cutoff)
                .order("published_at", desc=False)
                .range(offset, offset + batch_size - 1)
                .execute()
                .data
            )
        except Exception as exc:
            print(f"Failed to fetch batch {batch_num}: {exc}")
            break

        if not rows:
            print(f"No more articles at offset {offset}")
            break

        print(f"Retrieved {len(rows)} articles in batch {batch_num}")

        # Step 3: Upload each article as a JSON blob to Storage.
        for row in rows:
            article_id = row["id"]
            key = f"{article_id}.json"
            content_bytes = json.dumps(row, default=str).encode("utf-8")

            try:
                supabase.storage.from_(BUCKET).upload(
                    path=key,
                    file=content_bytes,
                    file_options={
                        "content-type": "application/json",
                        "upsert": "true",
                    },
                )
                archived_count += 1
                archived_ids.append(str(article_id))
            except Exception as exc:
                print(f"Failed to archive article {article_id}: {exc}")

        offset += len(rows)

        if archived_count >= total:
            print(f"Reached expected total of {total} articles")
            break

    print(f"Archived {archived_count}/{total} articles to storage")

    # Step 4: Delete only the articles that were successfully archived.
    if not archived_ids:
        print("No articles successfully archived -- skipping deletion")
        return

    print(
        f"Deleting {len(archived_ids)} successfully archived "
        "articles from database..."
    )

    deleted_total = 0

    for i in range(0, len(archived_ids), 100):
        batch = archived_ids[i:i + 100]
        batch_num = i // 100 + 1

        try:
            for article_id in batch:
                supabase.table("articles").delete().eq(
                    "id", article_id
                ).execute()

            deleted_total += len(batch)
            print(
                f"Deleted batch {batch_num}: {len(batch)} articles "
                f"(total: {deleted_total}/{len(archived_ids)})"
            )
        except Exception as exc:
            print(f"Failed to delete batch {batch_num}: {exc}")

    print(
        f"Cleanup complete -- "
        f"archived: {archived_count}, deleted: {deleted_total}"
    )
