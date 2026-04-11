"""Integration tests for user management endpoints."""


def test_create_user(client, mock_supabase, auth_headers):
    mock_supabase.table().upsert().execute.return_value.data = [{"id": "test_uid"}]
    response = client.post(
        "/users",
        json={"uid": "test_uid", "email": "a@b.com"},
        headers=auth_headers,
    )
    assert response.status_code in (200, 201)


def test_delete_own_account(client, mock_supabase, auth_headers):
    mock_supabase.table().delete().eq().execute.return_value.data = []
    response = client.delete("/users/test_uid", headers=auth_headers)
    assert response.status_code in (200, 204)


def test_delete_other_account_forbidden(client, mock_supabase, auth_headers):
    response = client.delete("/users/different_uid", headers=auth_headers)
    assert response.status_code == 403


def test_clear_own_history(client, mock_supabase, auth_headers):
    mock_supabase.table().delete().eq().execute.return_value.data = []
    response = client.delete("/users/test_uid/history", headers=auth_headers)
    assert response.status_code in (200, 204)
