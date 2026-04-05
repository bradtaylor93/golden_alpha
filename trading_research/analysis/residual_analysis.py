"""Residual analysis helpers."""

from __future__ import annotations

import pandas as pd


def residual_summary(df: pd.DataFrame) -> pd.DataFrame:
    if not {"target", "prediction"}.issubset(df.columns):
        raise ValueError("expected target and prediction columns")
    residual = df["target"] - df["prediction"]
    return pd.DataFrame({"residual_mean": [float(residual.mean())], "residual_std": [float(residual.std(ddof=0))]})
