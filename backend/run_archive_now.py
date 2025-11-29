import os
import sys

# Add the backend directory to the python path so we can import 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force configuration to archive EVERYTHING from today backwards
os.environ["ARCHIVE_DAYS"] = "0"

print("--- Starting Manual Archive Run ---")
print(f"Target Bucket: {os.environ.get('ARCHIVE_BUCKET', 'articles-archive')}")

try:
    from app.jobs.archiving import archive_old_articles
    archive_old_articles()
    print("--- Archive Run Complete ---")
except Exception as e:
    print(f"!!! Error during archive: {e}")
