"""Evaluate VWAP-style signals for long-term reversal predictability.

This experiment tests three related signals over a 50-equity universe:
1) anchored VWAP distance (anchored at each calendar-year start),
2) rolling volume-weighted price (20-day VWAP),
3) rolling average price (20-day simple average of typical price).

Target:
    Long-term reversal event over 63 trading days:
    reversal_t = 1 if sign(past_63d_return) != sign(future_63d_return)
                 and |past_63d_return| >= trend_threshold

The study reports out-of-sample metrics on a strict chronological split.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data.vendors.yahoo import YahooMarketDataVendor

UNIVERSE_50 = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "TSLA",
    "BRK-B",
    "JPM",
    "JNJ",
    "V",
    "UNH",
    "HD",
    "PG",
    "MA",
    "XOM",
    "CVX",
    "ABBV",
    "BAC",
    "KO",
    "PEP",
    "COST",
    "AVGO",
    "TMO",
    "MRK",
    "WMT",
    "DIS",
    "ADBE",
    "CRM",
    "NFLX",
    "PFE",
    "CSCO",
    "ACN",
    "MCD",
    "ABT",
    "DHR",
    "LIN",
    "AMD",
    "ORCL",
    "INTC",
    "NKE",
    "WFC",
    "QCOM",
    "TXN",
    "UPS",
    "CAT",
    "GS",
    "HON",
    "LOW",
    "AMGN",
]


@dataclass(frozen=True)
class EvalConfig:
    lookback_days: int = 63
    trend_threshold: float = 0.05
    rolling_window: int = 20
    test_ratio: float = 0.30
    top_quantile: float = 0.90
    bottom_quantile: float = 0.10
    min_asset_samples: int = 40


def _rank_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    frame = pd.DataFrame({"y": y, "s": s}).dropna()
    if frame.empty:
        return float("nan")
    if frame["y"].nunique() < 2:
        return float("nan")
    pos = frame["y"] == 1
    neg = frame["y"] == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = frame["s"].rank(method="average")
    sum_pos = float(ranks[pos].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _t_stat(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) < 2:
        return float("nan")
    std = float(x.std(ddof=1))
    if std == 0:
        return float("nan")
    return float(x.mean() / std * math.sqrt(len(x)))


def _prepare_panel(bars: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "asset", "close"]).sort_values(["asset", "timestamp"])
    frame["asset"] = frame["asset"].astype(str)
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["high"] = pd.to_numeric(frame["high"], errors="coerce")
    frame["low"] = pd.to_numeric(frame["low"], errors="coerce")
    frame = frame.dropna(subset=["close", "high", "low"])
    frame = frame[frame["close"] > 0].copy()

    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    frame["typical_price"] = typical

    # Anchored VWAP: reset anchor at each calendar-year start per asset.
    frame["year"] = frame["timestamp"].dt.year
    pv = frame["typical_price"] * frame["volume"]
    grouped = frame.groupby(["asset", "year"], sort=False)
    cum_pv = pv.groupby([frame["asset"], frame["year"]], sort=False).cumsum()
    cum_vol = frame["volume"].groupby([frame["asset"], frame["year"]], sort=False).cumsum()
    frame["anchored_vwap"] = cum_pv / cum_vol.replace(0.0, np.nan)

    # Rolling VWAP and rolling average price.
    roll = frame.groupby("asset", sort=False)
    pv_roll = (frame["typical_price"] * frame["volume"]).groupby(frame["asset"], sort=False).rolling(
        cfg.rolling_window, min_periods=cfg.rolling_window
    )
    v_roll = frame["volume"].groupby(frame["asset"], sort=False).rolling(
        cfg.rolling_window, min_periods=cfg.rolling_window
    )
    frame["rolling_vwap"] = pv_roll.sum().reset_index(level=0, drop=True) / v_roll.sum().reset_index(
        level=0, drop=True
    ).replace(0.0, np.nan)
    frame["average_price"] = roll["typical_price"].rolling(cfg.rolling_window, min_periods=cfg.rolling_window).mean().reset_index(
        level=0, drop=True
    )

    # Long-term trend and reversal target.
    look = cfg.lookback_days
    frame["past_return"] = roll["close"].pct_change(look)
    frame["future_return"] = roll["close"].shift(-look) / frame["close"] - 1.0

    trend_ok = frame["past_return"].abs() >= cfg.trend_threshold
    opposite_sign = (frame["past_return"] * frame["future_return"]) < 0
    frame["reversal_event"] = (trend_ok & opposite_sign).astype(int)

    # Signal scores: positive -> stronger expected reversal probability.
    frame["score_anchored_vwap"] = np.sign(frame["past_return"]) * (frame["close"] / frame["anchored_vwap"] - 1.0)
    frame["score_rolling_vwap"] = np.sign(frame["past_return"]) * (frame["close"] / frame["rolling_vwap"] - 1.0)
    frame["score_average_price"] = np.sign(frame["past_return"]) * (frame["close"] / frame["average_price"] - 1.0)

    keep_cols = [
        "timestamp",
        "asset",
        "close",
        "past_return",
        "future_return",
        "reversal_event",
        "score_anchored_vwap",
        "score_rolling_vwap",
        "score_average_price",
    ]
    out = frame[keep_cols].dropna()
    out = out[out["past_return"].abs() >= cfg.trend_threshold].copy()
    return out.sort_values(["timestamp", "asset"]).reset_index(drop=True)


def _split_frame(panel: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = pd.Index(sorted(panel["timestamp"].dropna().unique()))
    if len(dates) < 10:
        raise ValueError("Insufficient timestamps for train/test split.")
    split_idx = int((1.0 - test_ratio) * len(dates))
    split_idx = min(max(split_idx, 1), len(dates) - 1)
    split_ts = pd.Timestamp(dates[split_idx])
    train = panel[panel["timestamp"] < split_ts].copy()
    test = panel[panel["timestamp"] >= split_ts].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty partition.")
    return train, test, split_ts


def _evaluate_metric(
    train: pd.DataFrame,
    test: pd.DataFrame,
    score_col: str,
    cfg: EvalConfig,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    train_stats = (
        train.groupby("asset", as_index=False)[score_col]
        .agg(train_mean="mean", train_std="std")
        .replace({"train_std": {0.0: np.nan}})
    )

    train_norm = train[["asset", score_col]].merge(train_stats, on="asset", how="left")
    train_norm["score_norm"] = (pd.to_numeric(train_norm[score_col], errors="coerce") - train_norm["train_mean"]) / train_norm[
        "train_std"
    ]
    train_norm = train_norm.dropna(subset=["score_norm"])
    q_hi = float(train_norm["score_norm"].quantile(cfg.top_quantile))
    q_lo = float(train_norm["score_norm"].quantile(cfg.bottom_quantile))

    scored = test[["timestamp", "asset", "past_return", "future_return", "reversal_event", score_col]].copy()
    scored = scored.rename(columns={score_col: "score"})
    scored = scored.merge(train_stats, on="asset", how="left")
    scored["score_norm"] = (pd.to_numeric(scored["score"], errors="coerce") - scored["train_mean"]) / scored["train_std"]
    scored = scored.dropna(subset=["score_norm"])
    scored["is_top"] = scored["score_norm"] >= q_hi
    scored["is_bottom"] = scored["score_norm"] <= q_lo
    scored["contrarian_position"] = -np.sign(scored["past_return"]).replace(0.0, 0.0)
    scored["contrarian_forward_return"] = scored["contrarian_position"] * scored["future_return"]

    base_rate = float(scored["reversal_event"].mean())
    top_mask = scored["is_top"]
    bot_mask = scored["is_bottom"]
    top_rate = float(scored.loc[top_mask, "reversal_event"].mean()) if top_mask.any() else float("nan")
    bot_rate = float(scored.loc[bot_mask, "reversal_event"].mean()) if bot_mask.any() else float("nan")

    top_returns = scored.loc[top_mask, "contrarian_forward_return"]
    auc = _rank_auc(scored["reversal_event"], scored["score_norm"])
    effective_auc = max(auc, 1.0 - auc) if not math.isnan(auc) else float("nan")
    signal_side = "top_decile" if (math.isnan(top_rate) or math.isnan(bot_rate) or top_rate >= bot_rate) else "bottom_decile"
    selected_rate = top_rate if signal_side == "top_decile" else bot_rate
    rows = {
        "metric": score_col,
        "n_test": float(len(scored)),
        "reversal_rate_base": base_rate,
        "reversal_rate_top_decile": top_rate,
        "reversal_rate_bottom_decile": bot_rate,
        "top_minus_bottom_reversal_spread": float(top_rate - bot_rate) if not (math.isnan(top_rate) or math.isnan(bot_rate)) else float("nan"),
        "top_vs_base_lift_pct": float((top_rate / base_rate - 1.0) * 100.0) if base_rate > 0 and not math.isnan(top_rate) else float("nan"),
        "auc_reversal": auc,
        "auc_effective_abs": effective_auc,
        "preferred_signal_side": signal_side,
        "preferred_signal_reversal_rate": selected_rate,
        "n_top_decile": float(int(top_mask.sum())),
        "contrarian_mean_return_top_decile": float(top_returns.mean()) if len(top_returns) else float("nan"),
        "contrarian_win_rate_top_decile": float((top_returns > 0).mean()) if len(top_returns) else float("nan"),
        "contrarian_tstat_top_decile": _t_stat(top_returns),
    }

    # Asset-level AUC on OOS for robustness.
    by_asset: list[dict[str, float | str]] = []
    for asset, g in scored.groupby("asset"):
        if len(g) < cfg.min_asset_samples or g["reversal_event"].nunique() < 2:
            continue
        by_asset.append(
            {
                "metric": score_col,
                "asset": asset,
                "n_obs": float(len(g)),
                "auc_reversal": _rank_auc(g["reversal_event"], g["score_norm"]),
                "reversal_rate": float(g["reversal_event"].mean()),
            }
        )
    return rows, pd.DataFrame(by_asset)


def main() -> None:
    out_root = Path("trading_research/examples/_output/16_vwap_reversal_universe_study")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = EvalConfig()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(UNIVERSE_50, period="3y", interval="1d")
    if bars.empty:
        raise ValueError("No Yahoo bars returned for the requested universe.")

    panel = _prepare_panel(bars, cfg)
    train, test, split_ts = _split_frame(panel, cfg.test_ratio)

    metric_rows: list[dict[str, float | str]] = []
    asset_rows: list[pd.DataFrame] = []
    for score_col in ["score_anchored_vwap", "score_rolling_vwap", "score_average_price"]:
        row, by_asset = _evaluate_metric(train, test, score_col, cfg)
        metric_rows.append(row)
        asset_rows.append(by_asset)

    metric_df = pd.DataFrame(metric_rows).sort_values("auc_reversal", ascending=False).reset_index(drop=True)
    asset_df = pd.concat(asset_rows, ignore_index=True) if asset_rows else pd.DataFrame()
    asset_summary = (
        asset_df.groupby("metric", as_index=False)
        .agg(
            mean_asset_auc=("auc_reversal", "mean"),
            median_asset_auc=("auc_reversal", "median"),
            pct_assets_auc_gt_0_5=("auc_reversal", lambda s: float((s > 0.5).mean())),
            n_assets=("asset", "nunique"),
        )
        .sort_values("mean_asset_auc", ascending=False)
        if not asset_df.empty
        else pd.DataFrame(columns=["metric", "mean_asset_auc", "median_asset_auc", "pct_assets_auc_gt_0_5", "n_assets"])
    )

    summary = {
        "universe_size_requested": len(UNIVERSE_50),
        "universe_size_fetched": int(bars["asset"].nunique()),
        "start_timestamp": str(panel["timestamp"].min()),
        "end_timestamp": str(panel["timestamp"].max()),
        "split_timestamp": str(split_ts),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "trend_lookback_days": cfg.lookback_days,
        "future_horizon_days": cfg.lookback_days,
        "trend_threshold_abs_return": cfg.trend_threshold,
        "best_metric_by_auc": str(metric_df.iloc[0]["metric"]) if not metric_df.empty else "n/a",
        "best_metric_auc": float(metric_df.iloc[0]["auc_reversal"]) if not metric_df.empty else float("nan"),
    }

    panel.to_csv(reports_dir / "panel_features.csv", index=False)
    metric_df.to_csv(reports_dir / "metric_oos_performance.csv", index=False)
    asset_df.to_csv(reports_dir / "asset_level_auc.csv", index=False)
    asset_summary.to_csv(reports_dir / "asset_level_auc_summary.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("universe fetched:", summary["universe_size_fetched"])
    print("test rows:", summary["test_rows"], "split timestamp:", summary["split_timestamp"])
    print(metric_df.to_string(index=False))
    if not asset_summary.empty:
        print("\nasset-level auc summary:")
        print(asset_summary.to_string(index=False))


if __name__ == "__main__":
    main()
