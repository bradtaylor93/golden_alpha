"""Baseline causal features derived directly from bars."""

from __future__ import annotations

import numpy as np
import pandas as pd


class BaselineFeatureFamily:
    name = "baseline"
    version = "2"

    def build(self, bars: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
        lookbacks = params.get("lookbacks", [1, 5, 10])
        if not isinstance(lookbacks, list):
            raise TypeError("lookbacks must be a list[int]")

        df = bars.sort_values(["asset", "timestamp"]).copy()
        out = df.loc[:, ["timestamp", "asset"]].copy()
        grouped = df.groupby("asset", sort=False)
        ret1 = grouped["close"].pct_change().fillna(0.0)
        range_pct = (df["high"] - df["low"]) / df["close"].replace(0, pd.NA)
        dollar_volume = df["close"] * df["volume"]
        out["ret_1"] = ret1
        for lb in lookbacks:
            min_obs = max(1, min(lb, max(2, lb // 2)))
            out[f"ret_{lb}"] = grouped["close"].pct_change(lb).fillna(0.0)
            roll_mean = (
                ret1.groupby(df["asset"])
                .rolling(lb, min_periods=min_obs)
                .mean()
                .reset_index(level=0, drop=True)
            )
            roll_std = (
                ret1.groupby(df["asset"])
                .rolling(lb, min_periods=min_obs)
                .std()
                .reset_index(level=0, drop=True)
                .replace(0, pd.NA)
            )
            out[f"ret_z_{lb}"] = ((ret1 - roll_mean) / roll_std).fillna(0.0)
            out[f"vol_{lb}"] = (
                ret1.groupby(df["asset"])
                .rolling(lb, min_periods=min_obs)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )
            out[f"range_mean_{lb}"] = (
                range_pct.groupby(df["asset"])
                .rolling(lb, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )
            vol_mean = (
                dollar_volume.groupby(df["asset"])
                .rolling(lb, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            vol_std = (
                dollar_volume.groupby(df["asset"])
                .rolling(lb, min_periods=min_obs)
                .std()
                .reset_index(level=0, drop=True)
                .replace(0, pd.NA)
            )
            out[f"dollar_volume_z_{lb}"] = ((dollar_volume - vol_mean) / vol_std).fillna(0.0)

            roll_high = grouped["high"].rolling(lb, min_periods=1).max().reset_index(level=0, drop=True)
            roll_low = grouped["low"].rolling(lb, min_periods=1).min().reset_index(level=0, drop=True)
            out[f"close_pos_{lb}"] = (
                (df["close"] - roll_low) / (roll_high - roll_low).replace(0, pd.NA)
            ).fillna(0.5)

        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        hour = ts.dt.hour.fillna(0).astype(float)
        dow = ts.dt.dayofweek.fillna(0).astype(float)
        out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

        out["range"] = range_pct
        out["dollar_volume"] = dollar_volume
        return out.fillna(0.0)
