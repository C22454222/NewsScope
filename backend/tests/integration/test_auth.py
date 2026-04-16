def test_no_token_rejected(client):
    response = client.get("/api/bias-profile")
    assert response.status_code in (401, 403, 422)


def test_malformed_token_rejected(client):
    response = client.get(
        "/api/bias-profile",
        headers={"Authorization": "Bearer malformed.token.here"},
    )
    assert response.status_code in (401, 403, 422, 500)


def test_public_endpoints_no_auth_required(client, mock_supabase):
    mock_supabase.table().select().execute.return_value.data = []
    response = client.get("/articles")
    assert response.status_code in (200, 422, 500)
