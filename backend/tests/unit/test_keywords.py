"""Unit tests for keyword extraction."""
from app.jobs.fact_checking import extract_keywords


def test_filters_stopwords():
    kws = extract_keywords("The quick brown fox jumps over the lazy dog")
    assert "the" not in [k.lower() for k in kws]


def test_preserves_proper_nouns():
    kws = extract_keywords("Joe Biden visited Germany on Tuesday")
    assert any("biden" in k.lower() for k in kws)


def test_empty_input_safe():
    assert extract_keywords("") == []


def test_returns_top_n():
    kws = extract_keywords("politics economy climate trade war " * 10, max_keywords=5)
    assert len(kws) <= 5
