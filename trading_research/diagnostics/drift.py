"""Feature drift diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_std_shift(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> float:
    if not columns:
        return 0.0
    shifts: list[float] = []
    for col in columns:
        train_mean = float(train[col].mean())
        test_mean = float(test[col].mean())
        train_std = float(train[col].std() or 1.0)
        shifts.append(abs(test_mean - train_mean) / max(train_std, 1e-8))
    return float(np.mean(shifts))


def feature_drift_summary(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Simple drift summary containing standardized mean and quantile shifts."""
    if not columns:
        return {"mean_std_shift": 0.0, "q50_shift": 0.0, "q90_shift": 0.0}
    mean_shift = mean_std_shift(train, test, columns)
    q50: list[float] = []
    q90: list[float] = []
    for c in columns:
        tr = train[c]
        te = test[c]
        denom = max(float(tr.std() or 1.0), 1e-8)
        q50.append(abs(float(te.quantile(0.5) - tr.quantile(0.5))) / denom)
        q90.append(abs(float(te.quantile(0.9) - tr.quantile(0.9))) / denom)
    return {
        "mean_std_shift": float(mean_shift),
        "q50_shift": float(np.mean(q50)),
        "q90_shift": float(np.mean(q90)),
    }
