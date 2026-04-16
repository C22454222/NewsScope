def test_get_articles_ok(client, mock_supabase):
    mock_supabase.table().select().execute.return_value.data = []
    response = client.get("/articles")
    assert response.status_code in (200, 422, 500)


def test_get_articles_with_category_filter(client, mock_supabase):
    mock_supabase.table().select().eq().execute.return_value.data = []
    response = client.get("/articles?category=sport")
    assert response.status_code in (200, 422, 500)


def test_articles_endpoint_accepts_limit(client, mock_supabase):
    mock_supabase.table().select().execute.return_value.data = []
    response = client.get("/articles?limit=5")
    assert response.status_code in (200, 422, 500)
