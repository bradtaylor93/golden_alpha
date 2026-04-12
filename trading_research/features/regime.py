"""Regime feature family."""

from __future__ import annotations

import pandas as pd


class RegimeFeatureFamily:
    """Features describing broad market regime state."""

    name = "regime"
    version = "1"

    def build(self, bars: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
        window = int(params.get("window", 20))
        df = bars.sort_values(["timestamp", "asset"]).copy()
        df["ret_1"] = df.groupby("asset", sort=False)["close"].pct_change().fillna(0.0)
        market = (
            df.groupby("timestamp", as_index=False)["ret_1"]
            .mean()
            .rename(columns={"ret_1": "market_ret_1"})
        )
        market["regime_vol"] = market["market_ret_1"].rolling(window=window, min_periods=1).std().fillna(0.0)
        market["regime_trend"] = market["market_ret_1"].rolling(window=window, min_periods=1).mean().fillna(0.0)
        out = df[["timestamp", "asset"]].merge(
            market[["timestamp", "regime_vol", "regime_trend"]],
            on="timestamp",
            how="left",
        )
        return out.fillna(0.0)
