"""Forward return target definitions."""

from __future__ import annotations

import pandas as pd

from trading_research.targets.registry import PredictionTask


def forward_return(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    work = df.sort_values(["asset", "timestamp"]).copy()
    grouped = work.groupby("asset")["close"]
    out = work[["timestamp", "asset"]].copy()
    out["target"] = grouped.shift(-horizon) / work["close"] - 1.0
    return out.dropna().reset_index(drop=True)


def make_forward_return_task(horizon: int) -> PredictionTask:
    return PredictionTask(
        name=f"forward_return_{horizon}",
        task_type="regression",
        horizon=horizon,
        build_target_fn=forward_return,
        metric_names=("mae", "rmse"),
        embargo=0,
    )
