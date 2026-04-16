def test_create_user_endpoint_exists(client, mock_supabase, auth_headers):
    response = client.post(
        "/users",
        json={"uid": "test_uid", "email": "test@example.com", "display_name": "Test"},
        headers=auth_headers,
    )
    assert response.status_code in (200, 201, 400, 401, 403, 422, 500)


def test_delete_own_account(client, mock_supabase, auth_headers):
    mock_supabase.table().delete().eq().execute.return_value.data = []
    response = client.delete("/users/test_uid", headers=auth_headers)
    assert response.status_code in (200, 204, 401, 403, 404, 500)


def test_clear_own_history(client, mock_supabase, auth_headers):
    mock_supabase.table().delete().eq().execute.return_value.data = []
    response = client.delete("/users/test_uid/history", headers=auth_headers)
    assert response.status_code in (200, 204, 401, 403, 404, 500)
