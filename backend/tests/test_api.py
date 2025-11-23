from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health_check():
    """
    Tests the /health endpoint to ensure the API is alive.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
