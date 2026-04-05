"""Subgroup analysis scaffolding."""

from __future__ import annotations

import pandas as pd


def by_asset_mean(df: pd.DataFrame, value_col: str = "prediction") -> pd.DataFrame:
    if "asset" not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    return df.groupby("asset", as_index=False)[value_col].mean().rename(columns={value_col: "mean_value"})
