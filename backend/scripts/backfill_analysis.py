# backend/scripts/backfill_analysis.py
import time
import os
import sys
from app.jobs.analysis import analyze_unscored_articles
from app.db.supabase import supabase

# Add backend to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_backfill():
    print("Starting full backfill of analysis...")
    while True:
        # Check how many are left
        count_response = supabase.table("articles")\
            .select("id", count="exact")\
            .is_("sentiment_score", "null")\
            .execute()

        count = count_response.count
        if count == 0:
            print("All articles analyzed!")
            break

        print(f"Remaining unscored articles: {count}")

        # Run one batch (5 articles)
        analyze_unscored_articles()

        # Wait a bit to be nice to Hugging Face API limits
        time.sleep(2)


if __name__ == "__main__":
    run_backfill()
