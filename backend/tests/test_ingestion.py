from app.jobs.ingestion import normalize_article


def test_normalize_article_from_newsapi():
    """
    Tests if data from a NewsAPI-like dictionary is correctly normalized.
    """
    api_article = {
        "source": {"name": "CNN"},
        "title": "Test Title",
        "url": "http://example.com/news",
        "publishedAt": "2025-11-23T18:00:00Z"
    }
    
    normalized = normalize_article(
        source_name=api_article["source"]["name"],
        url=api_article["url"],
        title=api_article["title"],
        published_at=api_article["publishedAt"]
    )
    
    assert normalized["title"] == "Test Title"
    assert normalized["source"] == "CNN"
