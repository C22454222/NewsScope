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
    assert "left_articles" in data
    assert "center_articles" in data
    assert "right_articles" in data
