"""Integration tests for authentication enforcement."""
from unittest.mock import patch


def test_no_token_rejected(client):
    response = client.get("/api/bias-profile")
    assert response.status_code == 401


def test_invalid_token_rejected(client):
    with patch("app.routes.users.verify_id_token", side_effect=Exception("invalid")):
        response = client.get(
            "/api/bias-profile",
            headers={"Authorization": "Bearer bad"},
        )
        assert response.status_code == 401


def test_public_endpoints_no_auth_required(client, mock_supabase):
    mock_supabase.table().select().execute.return_value.data = []
    response = client.get("/articles")
    assert response.status_code == 200
