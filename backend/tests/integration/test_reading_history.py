"""Integration tests for reading history and bias profile."""


def test_track_reading_inserts_snapshot(client, mock_supabase, auth_headers):
    mock_supabase.table().select().eq().single().execute.return_value.data = {
        "id": "art1",
        "political_bias_score": 0.5,
        "sentiment_score": 0.2,
        "source": "BBC",
        "general_bias": "unbiased",
        "credibility_score": 85,
    }
    mock_supabase.table().insert().execute.return_value.data = [{}]
    response = client.post(
        "/api/reading-history",
        json={"article_id": "art1", "time_spent_seconds": 120},
        headers=auth_headers,
    )
    assert response.status_code in (200, 201)


def test_bias_profile_returns_aggregates(client, mock_supabase, auth_headers):
    mock_supabase.table().select().eq().execute.return_value.data = [
        {
            "bias_score": -0.5,
            "sentiment_score": 0.3,
            "source": "BBC",
            "general_bias": "unbiased",
            "credibility_score": 85,
            "time_spent_seconds": 60,
        },
    ]
    response = client.get("/api/bias-profile", headers=auth_headers)
    assert response.status_code == 200


def test_snapshot_pattern_survives_archive(client, mock_supabase, auth_headers):
    mock_supabase.table().select().eq().execute.return_value.data = [
        {
            "bias_score": -0.5,
            "sentiment_score": 0.3,
            "source": "BBC",
            "general_bias": "biased",
            "credibility_score": 80,
            "time_spent_seconds": 120,
            "article_id": None,
        },
    ]
    response = client.get("/api/bias-profile", headers=auth_headers)
    assert response.status_code == 200
