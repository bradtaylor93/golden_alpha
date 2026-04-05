"""Baseline causal features derived directly from bars."""

from __future__ import annotations

import pandas as pd


class BaselineFeatureFamily:
    name = "baseline"
    version = "1"

    def build(self, bars: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
        lookbacks = params.get("lookbacks", [1, 5, 10])
        if not isinstance(lookbacks, list):
            raise TypeError("lookbacks must be a list[int]")

        df = bars.sort_values(["asset", "timestamp"]).copy()
        out = df.loc[:, ["timestamp", "asset"]].copy()
        grouped = df.groupby("asset", sort=False)
        ret1 = grouped["close"].pct_change()
        out["ret_1"] = ret1
        for lb in lookbacks:
            out[f"ret_{lb}"] = grouped["close"].pct_change(lb)
            out[f"vol_{lb}"] = grouped["close"].pct_change().rolling(lb).std().reset_index(
                level=0, drop=True
            )
        out["range"] = (df["high"] - df["low"]) / df["close"].replace(0, pd.NA)
        out["dollar_volume"] = df["close"] * df["volume"]
        return out.fillna(0.0)
