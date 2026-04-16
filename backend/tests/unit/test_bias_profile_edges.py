def test_weighted_average_empty_history():
    from app.routes.users import _weighted_average
    assert _weighted_average([], []) == 0.0


def test_weighted_average_zero_durations():
    from app.routes.users import _weighted_average
    assert _weighted_average([0.5, -0.3], [0, 0]) == 0.0


def test_largest_remainder_sums_to_100():
    from app.routes.users import _largest_remainder
    result = _largest_remainder({"L": 33.3, "C": 33.3, "R": 33.3})
    assert sum(result.values()) == 100
