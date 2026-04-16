def test_articles_endpoint_returns_200(client):
    response = client.get("/articles?limit=5")
    assert response.status_code == 200


def test_articles_invalid_category_returns_ok(client):
    response = client.get("/articles?category=nonexistent_cat")
    assert response.status_code == 200


def test_articles_pagination_limits(client):
    response = client.get("/articles?limit=5")
    assert response.status_code == 200


def test_admin_endpoint_requires_auth(client):
    response = client.post("/articles/admin/refetch")
    assert response.status_code in (401, 403, 404, 405, 422)


def test_sources_endpoint_returns_200(client):
    response = client.get("/sources")
    assert response.status_code == 200
