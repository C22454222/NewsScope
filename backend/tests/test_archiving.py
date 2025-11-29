# tests/test_archiving.py
from unittest.mock import patch
from app.jobs.archiving import archive_old_articles


def test_archive_old_articles_no_rows():
    """
    Test that the function exits gracefully when no articles need archiving.
    """
    # Mock supabase so we don't actually hit the database during tests
    with patch("app.jobs.archiving.supabase") as mock_supabase:
        # Simulate database returning empty list
        mock_supabase.table.return_value.select.return_value.lte.return_value.execute.return_value.data = []

        archive_old_articles()

        # Verify we tried to select, but did NOT try to upload anything
        mock_supabase.table.assert_called_with("articles")
        mock_supabase.storage.from_.assert_not_called()


def test_archive_old_articles_success():
    """
    Test that we convert content to bytes and upload when rows exist.
    """
    mock_article = {
        "id": "test-uuid",
        "title": "Old News",
        "published_at": "2020-01-01"
    }

    with patch("app.jobs.archiving.supabase") as mock_supabase:
        # Simulate database returning one article
        mock_supabase.table.return_value.select.return_value.lte.return_value.execute.return_value.data = [mock_article]

        archive_old_articles()

        # Verify we tried to upload to the correct bucket
        mock_supabase.storage.from_.assert_called_with("articles-archive")

        # Verify upload was called.
        mock_supabase.storage.from_.return_value.upload.assert_called_once()
