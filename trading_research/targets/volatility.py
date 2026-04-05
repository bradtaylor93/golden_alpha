"""Realized volatility target definitions."""

from __future__ import annotations

import pandas as pd

from trading_research.targets.registry import PredictionTask


def forward_realized_volatility(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    work = df.sort_values(["asset", "timestamp"]).copy()
    ret = work.groupby("asset")["close"].pct_change().fillna(0.0)
    forward_std = (
        ret.groupby(work["asset"])
        .rolling(horizon, min_periods=max(2, horizon // 4))
        .std()
        .shift(-horizon)
        .reset_index(level=0, drop=True)
    )
    out = work[["timestamp", "asset"]].copy()
    out["target"] = forward_std
    return out.dropna().reset_index(drop=True)


def make_forward_realized_volatility_task(horizon: int) -> PredictionTask:
    return PredictionTask(
        name=f"forward_realized_volatility_{horizon}",
        task_type="regression",
        horizon=horizon,
        build_target_fn=forward_realized_volatility,
        metric_names=("mae", "rmse"),
        embargo=0,
    )


# Backward-compatible alias.
make_forward_realized_vol_task = make_forward_realized_volatility_task
