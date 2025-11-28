# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

# Test client for exercising FastAPI endpoints
client = TestClient(app)


def test_health_check():
    """
    Verify that the /health endpoint is available and returns the expected payload.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
