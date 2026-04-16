def test_delete_user_mismatched_uid_returns_403(client, auth_header_user_a):
    response = client.delete("/users/uid_of_user_b", headers=auth_header_user_a)
    assert response.status_code == 403


def test_clear_history_mismatched_uid_returns_403(client, auth_header_user_a):
    response = client.delete("/users/uid_of_user_b/history", headers=auth_header_user_a)
    assert response.status_code == 403


def test_bias_profile_requires_auth(client):
    response = client.get("/api/bias-profile")
    assert response.status_code == 401


def test_reading_history_invalid_article_id(client, auth_header):
    response = client.post("/api/reading-history",
                           json={"article_id": "not-a-uuid", "duration_seconds": 5},
                           headers=auth_header)
    assert response.status_code in (400, 422)
