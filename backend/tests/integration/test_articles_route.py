"""Integration tests for the articles router."""


def test_get_articles_ok(client, mock_supabase):
    mock_supabase.table().select().execute.return_value.data = [
        {"id": "1", "title": "Test", "political_bias": "Left"},
    ]
    response = client.get("/articles")
    assert response.status_code == 200


def test_get_articles_with_category_filter(client, mock_supabase):
    mock_supabase.table().select().eq().execute.return_value.data = []
    response = client.get("/articles?category=sport")
    assert response.status_code == 200


def test_get_single_article(client, mock_supabase):
    mock_supabase.table().select().eq().single().execute.return_value.data = {
        "id": "1", "title": "X",
    }
    response = client.get("/articles/1")
    assert response.status_code in (200, 404)


def test_article_not_found(client, mock_supabase):
    mock_supabase.table().select().eq().single().execute.return_value.data = None
    response = client.get("/articles/nonexistent")
    assert response.status_code == 404
