"""Unit tests for the credibility scoring formula."""
from app.jobs.fact_checking import compute_credibility_score_sync as compute_credibility_score


def test_all_true_high_score():
    score = compute_credibility_score("Reuters", ["True", "True", "Mostly True"])
    assert score >= 85


def test_all_false_low_score():
    score = compute_credibility_score("Unknown", ["False", "False", "Pants on Fire"])
    assert score <= 60


def test_clamp_upper():
    score = compute_credibility_score("Reuters", ["True"] * 10)
    assert score <= 100


def test_clamp_lower():
    score = compute_credibility_score("Unknown", ["False"] * 10)
    assert score >= 10


def test_no_rulings_returns_base():
    score = compute_credibility_score("BBC", [])
    assert 70 <= score <= 95


def test_mixed_rulings_in_range():
    score = compute_credibility_score("CNN", ["True", "False", "Half True"])
    assert 10 <= score <= 100


def test_unknown_source_default():
    score = compute_credibility_score("Random Blog", [])
    assert score >= 50


def test_case_insensitive_rulings():
    upper = compute_credibility_score("BBC", ["TRUE", "TRUE", "TRUE"])
    lower = compute_credibility_score("BBC", ["true", "true", "true"])
    proper = compute_credibility_score("BBC", ["True", "True", "True"])
    assert upper == lower == proper
