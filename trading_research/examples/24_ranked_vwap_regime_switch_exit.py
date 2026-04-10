"""Regime-switch exit policy test for ranked VWAP momentum entries.

Policy requested:
- Keep same entry (top-decile ranked VWAP momentum).
- Use simple regime-aware exits (low-overfit):
  * Trend regime -> fixed 63-day hold.
  * High-vol regime -> trailing 12% with 42-day cap.
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

EXTRA_UNIVERSE_40 = [
    "BLK",
    "BKNG",
    "C",
    "CMCSA",
    "COP",
    "DE",
    "ELV",
    "GE",
    "GILD",
    "IBM",
    "ISRG",
    "LMT",
    "MDT",
    "MMM",
    "MO",
    "MS",
    "MU",
    "NOW",
    "PANW",
    "PLD",
    "PM",
    "PYPL",
    "RTX",
    "SBUX",
    "SCHW",
    "SPGI",
    "SYK",
    "T",
    "TJX",
    "TMUS",
    "UBER",
    "UNP",
    "VRTX",
    "AXP",
    "CB",
    "ETN",
    "INTU",
    "MDLZ",
    "PGR",
    "SO",
]

UNIVERSE_90 = tuple(dict.fromkeys([*UNIVERSE_50, *EXTRA_UNIVERSE_40]))


@dataclass(frozen=True)
class RegimeSwitchConfig:
    signal_lookback_days: int = 63
    rolling_vwap_window: int = 20
    top_quantile: float = 0.90
    test_ratio: float = 0.30
    gross_target: float = 1.0
    max_abs_weight_per_asset: float = 0.05
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    # Regime definitions
    trend_window_days: int = 63
    vol_window_days: int = 21
    # Exit policy params
    trend_max_hold_days: int = 63
    high_vol_max_hold_days: int = 42
    high_vol_trailing_stop: float = 0.12
    min_hold_days: int = 5


def _split_time(panel: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = sorted(panel["timestamp"].dropna().unique())
    split_idx = int((1.0 - test_ratio) * len(dates))
    split_idx = min(max(split_idx, 1), len(dates) - 1)
    split_ts = pd.Timestamp(dates[split_idx])
    train = panel[panel["timestamp"] < split_ts].copy()
    test = panel[panel["timestamp"] >= split_ts].copy()
    return train, test, split_ts


def _one_way_cost_return(cfg: RegimeSwitchConfig) -> float:
    return (cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side) / 10_000.0


def _build_panel(bars: pd.DataFrame, cfg: RegimeSwitchConfig) -> pd.DataFrame:
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "asset", "close"]).sort_values(["asset", "timestamp"])
    frame["asset"] = frame["asset"].astype(str)
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["close", "high", "low"])
    frame = frame[frame["close"] > 0].copy()

    typ = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    pv = typ * frame["volume"].fillna(0.0)
    g = frame.groupby("asset", sort=False)
    frame["rolling_vwap"] = (
        pv.groupby(frame["asset"], sort=False)
        .rolling(cfg.rolling_vwap_window, min_periods=cfg.rolling_vwap_window)
        .sum()
        .reset_index(level=0, drop=True)
        / frame["volume"]
        .fillna(0.0)
        .groupby(frame["asset"], sort=False)
        .rolling(cfg.rolling_vwap_window, min_periods=cfg.rolling_vwap_window)
        .sum()
        .reset_index(level=0, drop=True)
        .replace(0.0, np.nan)
    )

    look = cfg.signal_lookback_days
    frame["past_return"] = g["close"].pct_change(look)
    frame["score"] = np.sign(frame["past_return"]) * (frame["close"] / frame["rolling_vwap"] - 1.0)
    frame["rank_pct"] = frame.groupby("timestamp")["score"].rank(pct=True, method="average")
    frame["signal_sign"] = np.sign(frame["past_return"]).replace(0.0, np.nan)
    frame = frame.dropna(subset=["close", "rolling_vwap", "score", "rank_pct", "signal_sign"]).reset_index(drop=True)
    return frame


def _build_market_regimes(test_panel: pd.DataFrame, spy_df: pd.DataFrame, cfg: RegimeSwitchConfig) -> pd.DataFrame:
    spy = spy_df.copy()
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True, errors="coerce")
    spy = spy.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    spy["close"] = pd.to_numeric(spy["close"], errors="coerce")
    spy = spy.dropna(subset=["close"]).copy()
    spy["ret"] = spy["close"].pct_change()
    spy["trend"] = (1.0 + spy["ret"]).rolling(cfg.trend_window_days, min_periods=cfg.trend_window_days).apply(np.prod, raw=True) - 1.0
    spy["vol"] = spy["ret"].rolling(cfg.vol_window_days, min_periods=cfg.vol_window_days).std()
    vol_med = float(spy["vol"].median(skipna=True))
    spy["trend_regime"] = np.where(spy["trend"] >= 0.0, "trend", "non_trend")
    spy["vol_regime"] = np.where(spy["vol"] >= vol_med, "high_vol", "low_vol")
    out = spy[["timestamp", "trend_regime", "vol_regime"]].dropna().copy()

    # Align strictly to test timestamps.
    test_ts = pd.Series(sorted(test_panel["timestamp"].unique()), name="timestamp")
    aligned = pd.DataFrame({"timestamp": test_ts}).merge(out, on="timestamp", how="left").ffill()
    return aligned.dropna(subset=["trend_regime", "vol_regime"]).reset_index(drop=True)


def _build_regime_switch_trades(test_panel: pd.DataFrame, regimes: pd.DataFrame, cfg: RegimeSwitchConfig) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    test_panel = test_panel.sort_values(["asset", "timestamp"]).copy()
    all_dates = sorted(test_panel["timestamp"].unique())
    date_to_idx = {ts: i for i, ts in enumerate(all_dates)}
    n_dates = len(all_dates)

    reg = regimes.copy()
    reg["timestamp"] = pd.to_datetime(reg["timestamp"], utc=True, errors="coerce")
    regime_map = dict(zip(reg["timestamp"], zip(reg["trend_regime"], reg["vol_regime"])))

    for asset, g in test_panel.groupby("asset", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        g["entry_flag"] = (g["rank_pct"] >= cfg.top_quantile) & (g["signal_sign"].notna())
        last_end_idx = -1

        for _, r in g[g["entry_flag"]].iterrows():
            sig_ts = pd.Timestamp(r["timestamp"])
            sig_idx = date_to_idx.get(sig_ts)
            if sig_idx is None or sig_idx <= last_end_idx:
                continue
            start_idx = sig_idx + 1
            if start_idx >= n_dates:
                continue

            sign = float(r["signal_sign"])
            entry_price = float(r["close"])
            if not np.isfinite(sign) or not np.isfinite(entry_price) or entry_price <= 0:
                continue

            trend_regime, vol_regime = regime_map.get(sig_ts, ("trend", "low_vol"))
            if vol_regime == "high_vol":
                max_hold = cfg.high_vol_max_hold_days
                use_trailing = True
            else:
                max_hold = cfg.trend_max_hold_days
                use_trailing = False

            local_end_cap = min(sig_idx + max_hold, n_dates - 1)
            if local_end_cap <= start_idx:
                continue

            exit_idx = local_end_cap
            exit_reason = "max_hold"
            peak_signed_ret = 0.0
            for j in range(start_idx, local_end_cap + 1):
                ts_j = all_dates[j]
                row_j = g[g["timestamp"] == ts_j]
                if row_j.empty:
                    continue
                close_j = float(row_j.iloc[0]["close"])
                signed_ret_now = float(sign * (close_j / entry_price - 1.0))
                peak_signed_ret = max(peak_signed_ret, signed_ret_now)
                if (j - sig_idx) < cfg.min_hold_days:
                    continue
                if use_trailing and (peak_signed_ret - signed_ret_now) >= cfg.high_vol_trailing_stop:
                    exit_idx = j
                    exit_reason = "high_vol_trailing"
                    break

            strength = float(max(1e-6, r["rank_pct"] - cfg.top_quantile))
            rows.append(
                {
                    "strategy": "regime_switch_63_vs_hv_trail12_42",
                    "asset": str(asset),
                    "signal_timestamp": str(sig_ts),
                    "start_idx": float(start_idx),
                    "end_idx": float(exit_idx),
                    "signal_sign": sign,
                    "signal_strength": strength,
                    "exit_reason": exit_reason,
                    "hold_days": float(max(1, exit_idx - sig_idx)),
                    "trend_regime": str(trend_regime),
                    "vol_regime": str(vol_regime),
                }
            )
            last_end_idx = exit_idx

    return pd.DataFrame(rows)


def _weights_from_active(active: pd.DataFrame, cfg: RegimeSwitchConfig, asset_cols: list[str]) -> pd.Series:
    w = pd.Series(0.0, index=asset_cols, dtype=float)
    if active.empty:
        return w
    by_asset = active.groupby("asset", as_index=False).agg(
        raw=("signal_strength", lambda s: float(np.sum(s))),
        sign=("signal_sign", lambda s: float(np.sign(np.sum(s)))),
    )
    by_asset["raw_signed"] = by_asset["raw"] * by_asset["sign"]
    denom = float(np.sum(np.abs(by_asset["raw_signed"])))
    if denom <= 0.0:
        return w
    by_asset["w"] = (by_asset["raw_signed"] / denom) * cfg.gross_target
    by_asset["w"] = by_asset["w"].clip(-cfg.max_abs_weight_per_asset, cfg.max_abs_weight_per_asset)
    for _, r in by_asset.iterrows():
        a = str(r["asset"])
        if a in w.index:
            w.loc[a] = float(r["w"])
    return w


def _simulate_portfolio(test_panel: pd.DataFrame, trades: pd.DataFrame, spy_df: pd.DataFrame, cfg: RegimeSwitchConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    close = test_panel.pivot(index="timestamp", columns="asset", values="close").sort_index().ffill()
    rets = close.pct_change().fillna(0.0)
    dates = rets.index.to_list()
    assets = rets.columns.to_list()

    spy = spy_df.copy()
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True, errors="coerce")
    spy = spy.dropna(subset=["timestamp"]).sort_values("timestamp")
    spy = spy.set_index("timestamp")["close"].reindex(rets.index).ffill()
    spy_ret = spy.pct_change().fillna(0.0)

    if trades.empty:
        out = pd.DataFrame({"timestamp": rets.index, "net_return": 0.0, "spy_return": spy_ret.values, "alpha_return": -spy_ret.values})
        out["equity"] = 1.0
        out["running_max"] = 1.0
        out["drawdown"] = 0.0
        return out, {"n_days": float(len(out)), "n_trades": 0.0, "annual_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0}

    trades = trades.copy()
    trades["start_idx"] = pd.to_numeric(trades["start_idx"], errors="coerce").astype("Int64")
    trades["end_idx"] = pd.to_numeric(trades["end_idx"], errors="coerce").astype("Int64")
    trades = trades.dropna(subset=["start_idx", "end_idx", "asset", "signal_sign", "signal_strength"]).copy()

    one_way_cost = _one_way_cost_return(cfg)
    prev_w = pd.Series(0.0, index=assets, dtype=float)
    rows: list[dict[str, float | str]] = []

    for i in range(1, len(dates)):
        active = trades[(trades["start_idx"] <= i) & (trades["end_idx"] >= i)]
        w = _weights_from_active(active, cfg, assets)
        gross = float(np.dot(w.values, rets.iloc[i].reindex(assets).fillna(0.0).values))
        turnover = float(np.abs(w - prev_w).sum())
        cost = turnover * one_way_cost
        net = gross - cost
        spy_r = float(spy_ret.iloc[i]) if np.isfinite(spy_ret.iloc[i]) else 0.0
        rows.append(
            {
                "timestamp": str(dates[i]),
                "net_return": net,
                "spy_return": spy_r,
                "alpha_return": net - spy_r,
                "turnover": turnover,
                "gross_exposure": float(np.abs(w).sum()),
            }
        )
        prev_w = w

    out = pd.DataFrame(rows)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    out["running_max"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["running_max"] - 1.0

    if out.empty:
        ann_return = cagr = max_dd = sharpe_n = sharpe_a = float("nan")
    else:
        ann_return = float(np.exp(np.log1p(out["net_return"]).mean() * 252.0) - 1.0)
        start_ts = out["timestamp"].iloc[0]
        end_ts = out["timestamp"].iloc[-1]
        years = max(1e-9, (end_ts - start_ts).total_seconds() / (365.25 * 24 * 3600))
        final_equity = float(out["equity"].iloc[-1])
        cagr = float(final_equity ** (1.0 / years) - 1.0) if final_equity > 0 else float("nan")
        max_dd = float(out["drawdown"].min())
        sharpe_n = float((out["net_return"].mean() / out["net_return"].std(ddof=1)) * math.sqrt(252.0)) if len(out) > 1 and float(out["net_return"].std(ddof=1)) > 0 else float("nan")
        sharpe_a = float((out["alpha_return"].mean() / out["alpha_return"].std(ddof=1)) * math.sqrt(252.0)) if len(out) > 1 and float(out["alpha_return"].std(ddof=1)) > 0 else float("nan")

    summary = {
        "n_days": float(len(out)),
        "n_trades": float(len(trades)),
        "annual_return": ann_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "annualized_sharpe_net": sharpe_n,
        "annualized_sharpe_alpha": sharpe_a,
        "mean_hold_days": float(pd.to_numeric(trades["hold_days"], errors="coerce").mean()) if not trades.empty else float("nan"),
        "avg_gross_exposure": float(out["gross_exposure"].mean()) if not out.empty else float("nan"),
        "avg_turnover": float(out["turnover"].mean()) if not out.empty else float("nan"),
    }
    return out, summary


def main() -> None:
    out_root = Path("trading_research/examples/_output/24_ranked_vwap_regime_switch_exit")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = RegimeSwitchConfig()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(list(UNIVERSE_90), period="3y", interval="1d")
    spy = vendor.fetch_bars(["SPY"], period="3y", interval="1d")[["timestamp", "close"]]
    if bars.empty or spy.empty:
        raise ValueError("Missing data for regime-switch exit study.")

    panel = _build_panel(bars, cfg)
    _, test, split_ts = _split_time(panel, cfg.test_ratio)
    regimes = _build_market_regimes(test, spy, cfg)

    trades = _build_regime_switch_trades(test, regimes, cfg)
    daily, perf = _simulate_portfolio(test, trades, spy, cfg)

    baseline_path = Path("trading_research/examples/_output/23_ranked_vwap_exit_study/reports/exit_study_summary.csv")
    baseline = pd.read_csv(baseline_path)
    fixed = baseline[baseline["strategy"] == "fixed_63d"].copy()
    fixed_row = fixed.iloc[0].to_dict() if not fixed.empty else {}

    comparison = pd.DataFrame(
        [
            {"strategy": "fixed_63d", **fixed_row},
            {"strategy": "regime_switch_63_vs_hv_trail12_42", **perf},
        ]
    )
    comparison["delta_vs_fixed_cagr"] = comparison["cagr"] - float(fixed_row.get("cagr", np.nan))
    comparison["delta_vs_fixed_max_dd"] = comparison["max_drawdown"] - float(fixed_row.get("max_drawdown", np.nan))
    comparison["delta_vs_fixed_sharpe_net"] = comparison["annualized_sharpe_net"] - float(
        fixed_row.get("annualized_sharpe_net", np.nan)
    )

    summary = {
        "split_timestamp": str(split_ts),
        "universe_size_requested": len(UNIVERSE_90),
        "universe_size_fetched": int(pd.Series(bars["asset"]).nunique()),
        "entry_policy": "top_decile_ranked_vwap_momentum",
        "regime_switch_policy": {
            "trend_or_low_vol_exit": f"fixed_{cfg.trend_max_hold_days}d",
            "high_vol_exit": f"trailing_{int(cfg.high_vol_trailing_stop*100)}pct_or_{cfg.high_vol_max_hold_days}d",
            "trend_window_days": cfg.trend_window_days,
            "vol_window_days": cfg.vol_window_days,
            "min_hold_days": cfg.min_hold_days,
        },
        "comparison_vs_fixed63": comparison.to_dict(orient="records"),
    }

    trades.to_csv(reports_dir / "trades_regime_switch.csv", index=False)
    daily.to_csv(reports_dir / "portfolio_daily_regime_switch.csv", index=False)
    comparison.to_csv(reports_dir / "comparison_vs_fixed63.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("split_timestamp:", summary["split_timestamp"])
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

