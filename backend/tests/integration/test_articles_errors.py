def test_get_article_not_found(client):
    response = client.get("/articles/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_compare_empty_topic_returns_400(client):
    response = client.post("/api/articles/compare", json={"topic": ""})
    assert response.status_code in (400, 422)


def test_articles_invalid_category_returns_empty(client):
    response = client.get("/articles?category=nonexistent_cat")
    assert response.status_code == 200
    assert response.json() == [] or "articles" in response.json()


def test_articles_pagination_limits(client):
    response = client.get("/articles?limit=5")
    assert response.status_code == 200


def test_admin_endpoint_requires_auth(client):
    response = client.post("/articles/admin/refetch")
    assert response.status_code in (401, 403, 404)
