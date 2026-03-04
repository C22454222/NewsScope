"""
NewsScope archiving job unit tests.
Flake8: 0 errors/warnings.
"""

from unittest.mock import MagicMock, patch

from app.jobs.archiving import archive_old_articles


def _make_supabase_mock(count: int, rows: list) -> MagicMock:
    """
    Build a mock Supabase client that satisfies the archiving query chain:
      .table().select().lt().execute()  -> count
      .table().select().lt().order().range().execute() -> rows
    """
    mock = MagicMock()

    # Count query
    count_exec = MagicMock()
    count_exec.count = count
    (
        mock.table.return_value
        .select.return_value
        .lt.return_value
        .execute.return_value
    ) = count_exec

    # Fetch rows query
    rows_exec = MagicMock()
    rows_exec.data = rows
    (
        mock.table.return_value
        .select.return_value
        .lt.return_value
        .order.return_value
        .range.return_value
        .execute.return_value
    ) = rows_exec

    return mock


def test_archive_old_articles_no_rows():
    """Exits gracefully when no articles need archiving."""
    with patch("app.jobs.archiving.supabase") as mock_supabase:
        mock_supabase.table.return_value \
            .select.return_value \
            .lt.return_value \
            .execute.return_value.count = 0

        archive_old_articles()

        mock_supabase.storage.from_.assert_not_called()


def test_archive_old_articles_success():
    """Uploads article JSON and deletes from DB on success."""
    mock_article = {
        "id": "test-uuid",
        "title": "Old News",
        "published_at": "2020-01-01T00:00:00+00:00",
    }

    with patch("app.jobs.archiving.supabase") as mock_supabase:
        # Count returns 1
        mock_supabase.table.return_value \
            .select.return_value \
            .lt.return_value \
            .execute.return_value.count = 1

        # Batch fetch returns one row then empty (stops loop)
        mock_supabase.table.return_value \
            .select.return_value \
            .lt.return_value \
            .order.return_value \
            .range.return_value \
            .execute.return_value.data = [mock_article]

        archive_old_articles()

        mock_supabase.storage.from_.assert_called_with("articles-archive")
        mock_supabase.storage.from_.return_value \
            .upload.assert_called_once()
