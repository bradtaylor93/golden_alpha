"""Residual diagnostics utilities."""

from __future__ import annotations

import pandas as pd


def residual_std_by_asset(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    work = df.copy()
    work["residual"] = work[target_col] - work[prediction_col]
    out = work.groupby("asset", observed=True)["residual"].std().reset_index(name="residual_std_by_asset")
    if "outer_fold_id" in work.columns and "outer_fold_id" not in out.columns:
        out["outer_fold_id"] = int(work["outer_fold_id"].iloc[0])
    return out
