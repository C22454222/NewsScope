# app/jobs/backfill_categories.py
from app.db.supabase import supabase
from app.jobs.ingestion import infer_category  # FIXED import


def backfill_article_categories(batch_size: int = 200):
    offset = 0
    while True:
        resp = (
            supabase.table("articles")
            .select("id, url, title, category")
            .is_("category", None)
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break

        updates = []
        for row in rows:
            cat = infer_category(row.get("url"), row.get("title"))
            updates.append({"id": row["id"], "category": cat})

        if updates:
            supabase.table("articles").upsert(updates).execute()

        offset += batch_size
        if len(rows) < batch_size:
            break
