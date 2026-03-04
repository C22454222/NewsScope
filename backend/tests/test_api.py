"""
NewsScope API smoke tests.
Flake8: 0 errors/warnings.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """Verify /health returns 200 and expected payload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_articles():
    """Verify /articles returns a list."""
    response = client.get("/articles")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_compare_articles():
    """Verify /api/articles/compare returns grouped bias response."""
    response = client.post(
        "/api/articles/compare",
        json={"topic": "climate", "limit": 5},
    )
    assert response.status_code == 200
    assert "left_articles" in response.json()
