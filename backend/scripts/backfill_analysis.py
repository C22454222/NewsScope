"""
NewsScope analysis backfill script.

Runs analyze_unscored_articles() in a loop until all articles
in the database have sentiment and bias scores.

Run from backend/ directory:
    python -m scripts.backfill_analysis

Flake8: 0 errors/warnings.
"""

import asyncio
import os
import sys
import time

# Add backend root to path so app.* imports resolve
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from app.db.supabase import supabase  # noqa: E402
from app.jobs.analysis import analyze_unscored_articles  # noqa: E402


def run_backfill() -> None:
    """
    Loop until all articles have a sentiment score.
    Runs one batch per iteration with a 2s pause between
    batches to stay within HuggingFace API rate limits.
    """
    print("Starting full analysis backfill...")

    while True:
        count_response = (
            supabase.table("articles")
            .select("id", count="exact")
            .is_("sentiment_score", "null")
            .execute()
        )

        count = count_response.count
        if count == 0:
            print("All articles analyzed!")
            break

        print(f"Remaining unscored articles: {count}")

        asyncio.run(analyze_unscored_articles())

        time.sleep(2)


if __name__ == "__main__":
    run_backfill()
