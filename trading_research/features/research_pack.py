"""Richer feature pack for medium-horizon predictive tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ResearchFeaturePackFamily:
    """Feature pack oriented for return/volatility modeling."""

    name = "research_pack"
    version = "3"

    def build(self, bars: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
        short = int(params.get("short_window", 6))
        medium = int(params.get("medium_window", 20))
        long = int(params.get("long_window", 60))
        lookbacks = params.get("lookbacks", [2, 4, 10, 20, 40, 80, 120])
        if not isinstance(lookbacks, list):
            raise TypeError("lookbacks must be a list[int]")
        lookbacks = [int(v) for v in lookbacks]
        sr_windows = params.get("sr_windows", [20, 40, 80, 120])
        if not isinstance(sr_windows, list):
            raise TypeError("sr_windows must be a list[int]")
        sr_windows = [int(v) for v in sr_windows]
        bb_windows = params.get("bb_windows", [20, 40, 80])
        if not isinstance(bb_windows, list):
            raise TypeError("bb_windows must be a list[int]")
        bb_windows = [int(v) for v in bb_windows]
        hit_tolerance = float(params.get("hit_tolerance", 0.001))

        df = bars.sort_values(["asset", "timestamp"]).copy()
        out = df[["timestamp", "asset"]].copy()
        grouped = df.groupby("asset", sort=False)
        ret1 = grouped["close"].pct_change().fillna(0.0)
        log_ret1 = np.log1p(ret1.clip(lower=-0.999))

        out["ret_1"] = ret1
        for w in lookbacks:
            out[f"ret_{w}"] = grouped["close"].pct_change(w)
            out[f"vol_{w}"] = ret1.groupby(df["asset"]).rolling(w, min_periods=1).std().reset_index(level=0, drop=True)
            out[f"mom_{w}"] = ret1.groupby(df["asset"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            out[f"down_vol_{w}"] = (
                ret1.clip(upper=0.0).abs().groupby(df["asset"]).rolling(w, min_periods=1).std().reset_index(level=0, drop=True)
            )
            out[f"up_vol_{w}"] = (
                ret1.clip(lower=0.0).groupby(df["asset"]).rolling(w, min_periods=1).std().reset_index(level=0, drop=True)
            )
            out[f"log_ret_mean_{w}"] = (
                log_ret1.groupby(df["asset"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            )

        ema_short = grouped["close"].transform(lambda s: s.ewm(span=short, adjust=False).mean())
        ema_medium = grouped["close"].transform(lambda s: s.ewm(span=medium, adjust=False).mean())
        ema_long = grouped["close"].transform(lambda s: s.ewm(span=long, adjust=False).mean())
        out["ema_short_over_medium"] = ema_short / ema_medium.replace(0, pd.NA) - 1.0
        out["ema_medium_over_long"] = ema_medium / ema_long.replace(0, pd.NA) - 1.0
        out["trend_strength"] = (ema_short - ema_long) / ema_long.replace(0, pd.NA)
        out["ema_curvature"] = out["ema_short_over_medium"] - out["ema_medium_over_long"]

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
        vol_mid = "vol_20" if "vol_20" in out.columns else f"vol_{lookbacks[min(len(lookbacks) - 1, max(0, len(lookbacks) // 2))]}"
        vol_short = "vol_10" if "vol_10" in out.columns else f"vol_{lookbacks[min(len(lookbacks) - 1, 1)]}"
        vol_long = "vol_40" if "vol_40" in out.columns else f"vol_{lookbacks[-1]}"
        mom_short = "mom_4" if "mom_4" in out.columns else f"mom_{lookbacks[0]}"
        mom_long = "mom_20" if "mom_20" in out.columns else f"mom_{lookbacks[-1]}"
        ret_short = "ret_4" if "ret_4" in out.columns else f"ret_{lookbacks[0]}"
        out["range_vol_interaction"] = out["range_pct"] * out[vol_mid]
        out["vol_ratio_short_long"] = out[vol_short] / out[vol_long].replace(0, pd.NA)
        out["mom_accel_4_20"] = out[mom_short] - out[mom_long]
        out["ret_reversal_1_4"] = out["ret_1"] - out[ret_short]

        roll_high_20 = grouped["high"].rolling(20, min_periods=1).max().reset_index(level=0, drop=True)
        roll_low_20 = grouped["low"].rolling(20, min_periods=1).min().reset_index(level=0, drop=True)
        out["breakout_up_20"] = df["close"] / roll_high_20.replace(0, pd.NA) - 1.0
        out["breakout_down_20"] = df["close"] / roll_low_20.replace(0, pd.NA) - 1.0
        out["close_range_position_20"] = (
            (df["close"] - roll_low_20) / (roll_high_20 - roll_low_20).replace(0, pd.NA)
        )

        # Support/resistance hit features and asymmetry dynamics.
        for w in sr_windows:
            sr_high = grouped["high"].rolling(w, min_periods=1).max().reset_index(level=0, drop=True)
            sr_low = grouped["low"].rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
            dist_to_res = (sr_high - df["close"]) / df["close"].replace(0, pd.NA)
            dist_to_sup = (df["close"] - sr_low) / df["close"].replace(0, pd.NA)
            resistance_hit = (df["close"] >= sr_high * (1.0 - hit_tolerance)).astype(float)
            support_hit = (df["close"] <= sr_low * (1.0 + hit_tolerance)).astype(float)
            hit_asym = resistance_hit - support_hit
            out[f"resistance_hit_{w}"] = resistance_hit
            out[f"support_hit_{w}"] = support_hit
            out[f"hit_asym_{w}"] = hit_asym
            out[f"hit_asym_ema_{w}"] = (
                hit_asym.groupby(df["asset"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            )
            out[f"pct_to_resistance_{w}"] = (df["close"] / sr_high.replace(0, pd.NA) - 1.0)
            out[f"pct_to_support_{w}"] = (df["close"] / sr_low.replace(0, pd.NA) - 1.0)
            out[f"sr_distance_asym_{w}"] = dist_to_sup - dist_to_res

        # Price to EMA distance (more direct than EMA ratios for some models).
        out["pct_to_ema_short"] = df["close"] / ema_short.replace(0, pd.NA) - 1.0
        out["pct_to_ema_medium"] = df["close"] / ema_medium.replace(0, pd.NA) - 1.0
        out["pct_to_ema_long"] = df["close"] / ema_long.replace(0, pd.NA) - 1.0

        # Bollinger dynamics.
        for w in bb_windows:
            min_obs = max(5, w // 4)
            bb_mid = grouped["close"].rolling(w, min_periods=min_obs).mean().reset_index(level=0, drop=True)
            bb_std = (
                grouped["close"].rolling(w, min_periods=min_obs).std().reset_index(level=0, drop=True).replace(0, pd.NA)
            )
            bb_upper = bb_mid + 2.0 * bb_std
            bb_lower = bb_mid - 2.0 * bb_std
            bb_width = (bb_upper - bb_lower).replace(0, pd.NA)
            out[f"bb_z_{w}"] = (df["close"] - bb_mid) / bb_std
            out[f"bb_width_{w}"] = bb_width / bb_mid.replace(0, pd.NA)
            out[f"bb_pct_b_{w}"] = (df["close"] - bb_lower) / bb_width
            out[f"bb_breakout_up_{w}"] = (df["close"] > bb_upper).astype(float)
            out[f"bb_breakout_down_{w}"] = (df["close"] < bb_lower).astype(float)
            out[f"bb_breakout_asym_{w}"] = out[f"bb_breakout_up_{w}"] - out[f"bb_breakout_down_{w}"]

        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        hour = ts.dt.hour.fillna(0).astype(float)
        out["rp_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        out["rp_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

        return out.fillna(0.0)

