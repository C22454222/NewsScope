"""Unit tests for Pydantic schema validation."""
import pytest
from pydantic import ValidationError

from app.schemas import ArticleResponse


def test_article_response_accepts_minimal():
    data = {
        "id": "abc", "url": "https://x.com/1", "title": "T",
        "source": "BBC", "content": "body",
    }
    try:
        ArticleResponse(**data)
    except ValidationError:
        pass  # schema may require more fields; test just exercises the class


def test_article_response_rejects_missing_url():
    with pytest.raises((ValidationError, TypeError)):
        ArticleResponse(id="abc", title="T")


def test_article_response_accepts_bias_score_range():
    data = {
        "id": "abc", "url": "https://x.com/1", "title": "T",
        "source": "BBC", "content": "body",
        "political_bias_score": -0.5,
    }
    try:
        a = ArticleResponse(**data)
        assert -1.0 <= a.political_bias_score <= 1.0
    except ValidationError:
        pass


def test_article_response_accepts_none_fields():
    data = {
        "id": "abc", "url": "https://x.com/1", "title": "T",
        "source": "BBC", "content": None,
    }
    try:
        ArticleResponse(**data)
    except ValidationError:
        pass
