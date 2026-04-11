"""Unit tests for URL-based deduplication."""
from app.jobs.ingestion import is_duplicate_url


def test_identical_url_is_duplicate():
    existing = {"https://bbc.co.uk/news/1"}
    assert is_duplicate_url("https://bbc.co.uk/news/1", existing) is True


def test_different_url_not_duplicate():
    existing = {"https://bbc.co.uk/news/1"}
    assert is_duplicate_url("https://bbc.co.uk/news/2", existing) is False


def test_trailing_slash_normalised():
    existing = {"https://bbc.co.uk/news/1"}
    assert is_duplicate_url("https://bbc.co.uk/news/1/", existing) is True


def test_query_params_distinct():
    existing = {"https://example.com/a"}
    assert is_duplicate_url("https://example.com/a?utm=1", existing) is False
