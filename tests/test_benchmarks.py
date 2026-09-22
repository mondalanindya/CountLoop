"""Tests for benchmark metrics computation."""

from benchmarks.evaluate import calculate_counting_metrics


def test_calculate_counting_metrics():
    pairs = [
        (30, 30),  # exact in 30-60 tier
        (35, 30),  # diff 5, relative 5/30 = 0.166
        (58, 60),  # diff 2, relative 2/60 = 0.033 <= 5%
        (98, 100), # diff 2, relative 2/100 = 0.02 <= 5% in 61-120 tier
        (135, 140),# diff 5, relative 5/140 = 0.035 <= 5% in 121-200 tier
    ]

    metrics = calculate_counting_metrics(pairs)

    assert metrics["total_evaluated"] == 5
    assert metrics["exact_rate"] == 20.0
    assert metrics["tol_5_rate"] == 80.0
    assert metrics["tol_10_rate"] == 80.0
    assert metrics["mae"] == (0 + 5 + 2 + 2 + 5) / 5.0

    # Check tiers exist
    assert "30-60" in metrics["tiers"]
    assert "61-120" in metrics["tiers"]
    assert "121-200" in metrics["tiers"]
