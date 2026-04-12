"""Model comparison helpers."""

from __future__ import annotations

import pandas as pd


def compare_model_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values(["metric", "value"], ascending=[True, False])
