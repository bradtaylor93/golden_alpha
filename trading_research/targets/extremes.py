"""Extreme move target definitions."""

from __future__ import annotations

import pandas as pd

from trading_research.targets.registry import PredictionTask


def max_upside(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    work = df.sort_values(["asset", "timestamp"]).copy()
    future_max = (
        work.groupby("asset")["high"]
        .rolling(horizon, min_periods=max(2, horizon // 4))
        .max()
        .shift(-horizon)
        .reset_index(level=0, drop=True)
    )
    out = work[["timestamp", "asset"]].copy()
    out["target"] = future_max / work["close"] - 1.0
    return out.dropna().reset_index(drop=True)


def max_drawdown(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    work = df.sort_values(["asset", "timestamp"]).copy()
    future_min = (
        work.groupby("asset")["low"]
        .rolling(horizon, min_periods=max(2, horizon // 4))
        .min()
        .shift(-horizon)
        .reset_index(level=0, drop=True)
    )
    out = work[["timestamp", "asset"]].copy()
    out["target"] = future_min / work["close"] - 1.0
    return out.dropna().reset_index(drop=True)


def make_max_upside_task(horizon: int) -> PredictionTask:
    return PredictionTask(
        name=f"max_upside_{horizon}",
        task_type="regression",
        horizon=horizon,
        build_target_fn=max_upside,
        metric_names=("mae", "rmse"),
        embargo=0,
    )


def make_max_drawdown_task(horizon: int) -> PredictionTask:
    return PredictionTask(
        name=f"max_drawdown_{horizon}",
        task_type="regression",
        horizon=horizon,
        build_target_fn=max_drawdown,
        metric_names=("mae", "rmse"),
        embargo=0,
    )
