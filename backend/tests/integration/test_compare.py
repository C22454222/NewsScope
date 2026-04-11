"""Integration tests for the comparison endpoint."""


def test_compare_partitions_into_three_bands(client, mock_supabase):
    mock_supabase.table().select().execute.return_value.data = [
        {"id": "1", "political_bias_score": -0.6, "title": "L"},
        {"id": "2", "political_bias_score": 0.0, "title": "C"},
        {"id": "3", "political_bias_score": 0.7, "title": "R"},
    ]
    response = client.post("/api/articles/compare", json={"topic": "climate"})
    assert response.status_code == 200
    data = response.json()
    assert "left" in data
    assert "centre" in data
    assert "right" in data


def test_compare_empty_topic(client, mock_supabase):
    mock_supabase.table().select().execute.return_value.data = []
    response = client.post("/api/articles/compare", json={"topic": ""})
    assert response.status_code in (200, 400)
