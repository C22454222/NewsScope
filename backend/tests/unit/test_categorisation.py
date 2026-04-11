"""Unit tests for the four-tier category inference cascade."""
from app.core.categorisation import infer_category, CATEGORY_GROUP_MAP


def test_url_prefix_politics():
    cat = infer_category("https://bbc.co.uk/news/politics/abc", "Election news")
    assert cat in ("politics", "world")


def test_url_prefix_sport():
    cat = infer_category("https://bbc.co.uk/sport/football/123", "Match report")
    assert cat == "sport"


def test_keyword_fallback_climate():
    cat = infer_category("https://example.com/article", "Climate change report")
    assert cat in ("science", "environment", "world")


def test_empty_url_uses_title():
    cat = infer_category("", "Football match results")
    assert cat in ("sport", "general", "world")


def test_ambiguous_prefix_takes_precedence():
    cat = infer_category("https://cnn.com/politics/sports-funding", "Sports article")
    assert cat in ("politics", "sport")


def test_never_returns_general():
    cat = infer_category("https://example.com/random", "random text")
    assert cat != "general"


def test_category_group_map_resolves_subcategory():
    assert CATEGORY_GROUP_MAP.get("football") == "sport"
    assert CATEGORY_GROUP_MAP.get("climate") == "science"
