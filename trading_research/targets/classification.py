"""Classification target definitions."""

from __future__ import annotations

import pandas as pd

from trading_research.targets.registry import PredictionTask


def up_down(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    work = df.sort_values(["asset", "timestamp"]).copy()
    fwd = work.groupby("asset")["close"].shift(-horizon) / work["close"] - 1.0
    out = work[["timestamp", "asset"]].copy()
    out["target"] = (fwd > 0.0).astype(float)
    return out.dropna().reset_index(drop=True)


def make_up_down_task(horizon: int) -> PredictionTask:
    return PredictionTask(
        name=f"up_down_{horizon}",
        task_type="classification",
        horizon=horizon,
        build_target_fn=up_down,
        metric_names=("accuracy", "logloss"),
        embargo=0,
    )
