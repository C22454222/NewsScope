"""
NewsScope category backfill script.

Re-categorises articles with NULL or 'general' category using
the full 3-tier inference pipeline (URL -> title -> zero-shot).

Flake8: 0 errors/warnings.
"""

from app.core.categorisation import infer_category
from app.db.supabase import supabase


def backfill_article_categories(batch_size: int = 200) -> None:
    """
    Re-categorise articles that have no category set (NULL)
    or were previously stuck on 'general' due to missing content.

    Runs through all such articles in batches and applies the
    full 3-tier inference (URL -> title -> zero-shot model).
    """
    offset = 0
    total_updated = 0

    while True:
        resp = (
            supabase.table("articles")
            .select("id, url, title, content, category")
            .or_("category.is.null,category.eq.general")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break

        updates = []
        for row in rows:
            cat = infer_category(
                row.get("url"),
                row.get("title"),
                row.get("content"),
            )
            if cat != "general" or row.get("category") is None:
                updates.append({"id": row["id"], "category": cat})

        if updates:
            supabase.table("articles").upsert(updates).execute()
            total_updated += len(updates)
            print(
                f"  Backfilled {len(updates)} articles "
                f"(offset {offset})"
            )

        offset += batch_size
        if len(rows) < batch_size:
            break

    print(f"Backfill complete — {total_updated} articles updated")
