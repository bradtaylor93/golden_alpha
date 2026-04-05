"""Objective helpers for optimization."""


def maximize_inverse_metric(metric_value: float) -> float:
    """Convert a lower-is-better error metric into a score to maximize."""
    return 1.0 / (1.0 + max(metric_value, 0.0))
