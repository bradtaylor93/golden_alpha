"""Richer feature pack for medium-horizon predictive tasks."""

from __future__ import annotations

import pandas as pd


class ResearchFeaturePackFamily:
    """Feature pack oriented for return/volatility modeling."""

    name = "research_pack"
    version = "1"

    def build(self, bars: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
        short = int(params.get("short_window", 6))
        medium = int(params.get("medium_window", 20))
        long = int(params.get("long_window", 60))

        df = bars.sort_values(["asset", "timestamp"]).copy()
        out = df[["timestamp", "asset"]].copy()
        grouped = df.groupby("asset", sort=False)
        ret1 = grouped["close"].pct_change().fillna(0.0)

        out["ret_1"] = ret1
        for w in [2, 4, 10, 20, 40]:
            out[f"ret_{w}"] = grouped["close"].pct_change(w)
            out[f"vol_{w}"] = ret1.groupby(df["asset"]).rolling(w, min_periods=1).std().reset_index(level=0, drop=True)
            out[f"mom_{w}"] = ret1.groupby(df["asset"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)

        ema_short = grouped["close"].transform(lambda s: s.ewm(span=short, adjust=False).mean())
        ema_medium = grouped["close"].transform(lambda s: s.ewm(span=medium, adjust=False).mean())
        ema_long = grouped["close"].transform(lambda s: s.ewm(span=long, adjust=False).mean())
        out["ema_short_over_medium"] = ema_short / ema_medium.replace(0, pd.NA) - 1.0
        out["ema_medium_over_long"] = ema_medium / ema_long.replace(0, pd.NA) - 1.0

        range_pct = (df["high"] - df["low"]) / df["close"].replace(0, pd.NA)
        out["range_pct"] = range_pct
        out["dollar_volume"] = df["close"] * df["volume"]
        out["dollar_volume_z20"] = (
            out["dollar_volume"] - out.groupby("asset", observed=True)["dollar_volume"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        ) / (
            out.groupby("asset", observed=True)["dollar_volume"].transform(lambda s: s.rolling(20, min_periods=1).std()).replace(0, pd.NA)
        )

        abs_ret = ret1.abs()
        out["abs_ret_rolling_short"] = abs_ret.groupby(df["asset"]).rolling(short, min_periods=1).mean().reset_index(level=0, drop=True)
        out["abs_ret_rolling_long"] = abs_ret.groupby(df["asset"]).rolling(long, min_periods=1).mean().reset_index(level=0, drop=True)
        out["range_vol_interaction"] = out["range_pct"] * out["vol_20"]

        return out.fillna(0.0)

