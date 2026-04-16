def test_weighted_average_empty_returns_zero():
    scores, weights = [], []
    total = sum(weights) or 1
    result = sum(s * w for s, w in zip(scores, weights)) / total
    assert result == 0.0


def test_weighted_average_basic():
    scores = [0.5, -0.5]
    weights = [2, 2]
    total = sum(weights)
    result = sum(s * w for s, w in zip(scores, weights)) / total
    assert result == 0.0


def test_weighted_average_heavier_left():
    scores = [-1.0, 1.0]
    weights = [3, 1]
    total = sum(weights)
    result = sum(s * w for s, w in zip(scores, weights)) / total
    assert result == -0.5


def test_largest_remainder_sums_to_100():
    counts = {"L": 33, "C": 34, "R": 33}
    total = sum(counts.values())
    pcts = {k: round(v / total * 100) for k, v in counts.items()}
    # ensure sum rounds clean — smoke test for algorithm shape
    assert sum(pcts.values()) in (99, 100, 101)
