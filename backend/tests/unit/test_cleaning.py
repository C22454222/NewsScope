"""Unit tests for article text cleaning."""
from app.jobs.ingestion import clean_article_text


def test_strips_null_bytes():
    cleaned = clean_article_text("Hello\x00World")
    assert "\x00" not in cleaned


def test_strips_surrogates():
    cleaned = clean_article_text("Test\ud800text")
    assert "\ud800" not in cleaned


def test_normalises_whitespace():
    cleaned = clean_article_text("hello   world\n\n\nfoo")
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_removes_subscription_prompts():
    text = "Real content here. Subscribe to our newsletter for more!"
    cleaned = clean_article_text(text)
    assert "Real content" in cleaned


def test_empty_input_returns_empty():
    assert clean_article_text("") == ""


def test_unicode_preserved():
    assert "café" in clean_article_text("café")
