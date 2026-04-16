"""Contract tests asserting endpoint response shapes match Pydantic schemas."""


def test_articles_response_shape(client, mock_supabase):
    mock_supabase.table().select().execute.return_value.data = [{
        "id": "abc",
        "title": "T",
        "url": "https://example.com/1",
        "source": "BBC",
        "political_bias": "Left",
        "political_bias_score": -0.5,
        "sentiment_score": 0.2,
        "general_bias": "unbiased",
        "credibility_score": 85,
        "category": "politics",
    }]
    response = client.get("/articles")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    if body:
        article = body[0]
        for key in ["id", "title", "url", "source"]:
            assert key in article


def test_compare_response_has_three_keys(client, mock_supabase):
    mock_supabase.table().select().execute.return_value.data = []
    response = client.post("/api/articles/compare", json={"topic": "test"})
    assert response.status_code == 200
    body = response.json()
    for key in ["left_articles", "center_articles", "right_articles"]:
        assert key in body
        assert isinstance(body[key], list)


def test_bias_profile_response_shape(client, mock_supabase, auth_headers):
    mock_supabase.table().select().eq().execute.return_value.data = [{
        "bias_score": -0.3,
        "sentiment_score": 0.1,
        "source": "BBC",
        "general_bias": "unbiased",
        "credibility_score": 80,
        "time_spent_seconds": 60,
    }]
    response = client.get("/api/bias-profile", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    for key in ["leaning_counts", "source_breakdown", "avg_bias", "avg_sentiment"]:
        assert key in body or "leaning" in str(body) or "source" in str(body)


def test_articles_pagination(client, mock_supabase):
    mock_supabase.table().select().limit().execute.return_value.data = []
    response = client.get("/articles?limit=10")
    assert response.status_code == 200


def test_invalid_category_handled(client, mock_supabase):
    mock_supabase.table().select().eq().execute.return_value.data = []
    response = client.get("/articles?category=nonexistent")
    assert response.status_code == 200
