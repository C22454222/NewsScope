"""Unit tests for bias profile aggregation arithmetic."""
from app.routes.users import compute_weighted_average, largest_remainder_round


def test_weighted_average_balanced_inputs():
    items = [
        {"score": 1.0, "weight": 60},
        {"score": -1.0, "weight": 60},
    ]
    assert compute_weighted_average(items) == 0.0


def test_weighted_average_skewed_inputs():
    items = [
        {"score": 1.0, "weight": 300},
        {"score": -1.0, "weight": 60},
    ]
    assert compute_weighted_average(items) > 0.5


def test_empty_input_returns_zero():
    assert compute_weighted_average([]) == 0.0


def test_largest_remainder_sums_to_100():
    pcts = largest_remainder_round([0.333, 0.333, 0.334])
    assert sum(pcts) == 100


def test_largest_remainder_two_buckets():
    pcts = largest_remainder_round([0.5, 0.5])
    assert sum(pcts) == 100
