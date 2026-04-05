"""Placeholder plotting hooks for notebook/export integration."""


def describe_plotting_capabilities() -> list[str]:
    """Return available plot families.

    TODO: Implement actual plotting backends.
    """

    return ["equity_curve", "rolling_metrics", "drift_over_time", "residual_distribution"]
