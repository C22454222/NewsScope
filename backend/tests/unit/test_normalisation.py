"""Unit tests for source name canonicalisation."""
from app.jobs.ingestion import canonicalise_source_name


def test_sky_news_variant_normalised():
    raw = "UK News - The latest headlines from the UK | Sky News"
    assert canonicalise_source_name(raw) == "Sky News"


def test_npr_variant_normalised():
    assert canonicalise_source_name("NPR Topics: News") == "NPR"


def test_already_canonical_passthrough():
    assert canonicalise_source_name("BBC") == "BBC"


def test_unknown_source_returns_string():
    name = canonicalise_source_name("SomeRandomFeed")
    assert isinstance(name, str)
    assert len(name) > 0


def test_case_handling():
    result = canonicalise_source_name("bbc news")
    assert result in ("BBC", "BBC News")
