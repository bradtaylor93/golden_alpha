"""Robustness checks for the selected pegged-high long strategy.

Chosen strategy under test:
- Entry: long when close crosses above 1-month pegged high (21 trading days).
- Exit: fixed 63-trading-day horizon.

Requested robustness checks:
1) Purged walk-forward (year-by-year OOS),
2) Transaction cost + slippage haircut,
3) Bootstrap / permutation significance tests,
4) Regime stress tests (bull/bear, high-vol/low-vol),
5) "real signal vs luck" scorecard.
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
class RobustnessConfig:
    test_ratio: float = 0.30
    peg_window_days: int = 21
    fixed_horizon_days: int = 63
    purge_days: int = 63
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    bootstrap_iterations: int = 3000
    signflip_iterations: int = 5000
    random_entry_iterations: int = 1200
    regime_vol_window_days: int = 21
    regime_trend_window_days: int = 63
    random_state: int = 42


def _t_stat(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) < 2:
        return float("nan")
    std = float(x.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(x.mean() / std * math.sqrt(len(x)))


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


def _build_panel(bars: pd.DataFrame, cfg: RobustnessConfig) -> pd.DataFrame:
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "asset", "close"]).sort_values(["asset", "timestamp"])
    frame["asset"] = frame["asset"].astype(str)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["high"] = pd.to_numeric(frame["high"], errors="coerce")
    frame["low"] = pd.to_numeric(frame["low"], errors="coerce")
    frame = frame.dropna(subset=["close", "high", "low"])
    frame = frame[frame["close"] > 0].copy()

    g = frame.groupby("asset", sort=False)
    lb = cfg.peg_window_days
    frame["pegged_high_1m"] = g["high"].rolling(lb, min_periods=lb).max().shift(1).reset_index(level=0, drop=True)
    prev_close = g["close"].shift(1)
    frame["entry_long"] = ((prev_close <= frame["pegged_high_1m"]) & (frame["close"] > frame["pegged_high_1m"])).astype(int)

    keep = ["timestamp", "asset", "close", "entry_long"]
    return frame[keep].dropna().reset_index(drop=True)


def _roundtrip_cost_return(cfg: RobustnessConfig) -> float:
    total_bps = 2.0 * (cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side)
    return total_bps / 10_000.0


def _build_benchmark_close_series(vendor: YahooMarketDataVendor) -> pd.Series:
    bench = vendor.fetch_bars(["SPY"], period="3y", interval="1d")
    if bench.empty:
        raise ValueError("No benchmark bars returned for SPY.")
    b = bench.copy()
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True, errors="coerce")
    b["close"] = pd.to_numeric(b["close"], errors="coerce")
    b = b.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    if b.empty:
        raise ValueError("Benchmark SPY data is empty after preprocessing.")
    return b.drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp")["close"].astype(float)


def _map_benchmark_returns(
    benchmark_close: pd.Series,
    entry_ts: pd.Series,
    exit_ts: pd.Series,
) -> pd.Series:
    entry_idx = pd.DatetimeIndex(pd.to_datetime(entry_ts, utc=True, errors="coerce"))
    exit_idx = pd.DatetimeIndex(pd.to_datetime(exit_ts, utc=True, errors="coerce"))
    b_entry = benchmark_close.reindex(entry_idx, method="ffill")
    b_exit = benchmark_close.reindex(exit_idx, method="ffill")
    out = b_exit.to_numpy() / b_entry.to_numpy() - 1.0
    return pd.Series(out, index=entry_ts.index, dtype=float)


def _attach_benchmark_alpha(trades: pd.DataFrame, benchmark_close: pd.Series) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    out["benchmark_return"] = _map_benchmark_returns(
        benchmark_close=benchmark_close,
        entry_ts=out["entry_timestamp"],
        exit_ts=out["exit_timestamp"],
    )
    out["alpha_return"] = pd.to_numeric(out["net_return"], errors="coerce") - pd.to_numeric(
        out["benchmark_return"], errors="coerce"
    )
    hold = pd.to_numeric(out["hold_days"], errors="coerce")
    out["daily_alpha_return"] = out["alpha_return"] / hold.replace(0.0, np.nan)
    return out


def _simulate_non_overlapping_trades(asset_df: pd.DataFrame, cfg: RobustnessConfig) -> pd.DataFrame:
    df = asset_df.sort_values("timestamp").reset_index(drop=True)
    trades: list[dict[str, float | str]] = []
    rt_cost = _roundtrip_cost_return(cfg)

    i = 1
    n = len(df)
    while i < n - 1:
        if int(df.loc[i, "entry_long"]) != 1:
            i += 1
            continue

        entry_idx = i
        exit_idx = min(entry_idx + cfg.fixed_horizon_days, n - 1)
        if exit_idx <= entry_idx:
            i += 1
            continue

        entry_px = float(df.loc[entry_idx, "close"])
        exit_px = float(df.loc[exit_idx, "close"])
        gross_ret = exit_px / entry_px - 1.0
        net_ret = gross_ret - rt_cost
        hold_days = int(exit_idx - entry_idx)
        daily_log_net = np.log1p(net_ret) / hold_days if hold_days > 0 and net_ret > -1.0 else float("nan")

        trades.append(
            {
                "asset": str(df.loc[entry_idx, "asset"]),
                "entry_timestamp": str(pd.Timestamp(df.loc[entry_idx, "timestamp"])),
                "exit_timestamp": str(pd.Timestamp(df.loc[exit_idx, "timestamp"])),
                "entry_idx": float(entry_idx),
                "exit_idx": float(exit_idx),
                "hold_days": float(hold_days),
                "gross_return": float(gross_ret),
                "net_return": float(net_ret),
                "daily_log_net_return": float(daily_log_net),
            }
        )
        i = exit_idx + 1

    return pd.DataFrame(trades)


def _summarize_trades(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "n_trades": 0.0,
            "mean_net_return": float("nan"),
            "median_net_return": float("nan"),
            "win_rate_net": float("nan"),
            "t_stat_net": float("nan"),
            "sharpe_per_trade_net": float("nan"),
            "annualized_sharpe_daily_proxy_net": float("nan"),
            "mean_hold_days": float("nan"),
            "p05_net_return": float("nan"),
            "cvar10_net_return": float("nan"),
            "worst_net_return": float("nan"),
            "mean_alpha_return": float("nan"),
            "win_rate_alpha": float("nan"),
            "t_stat_alpha": float("nan"),
            "sharpe_per_trade_alpha": float("nan"),
            "annualized_sharpe_daily_proxy_alpha": float("nan"),
        }
    r = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    dlog = pd.to_numeric(trades["daily_log_net_return"], errors="coerce").dropna()
    h = pd.to_numeric(trades["hold_days"], errors="coerce").dropna()

    sharpe_trade = float(r.mean() / r.std(ddof=1)) if len(r) > 1 and float(r.std(ddof=1)) > 0 else float("nan")
    sharpe_daily = (
        float((dlog.mean() / dlog.std(ddof=1)) * math.sqrt(252.0))
        if len(dlog) > 1 and float(dlog.std(ddof=1)) > 0
        else float("nan")
    )
    alpha = pd.to_numeric(trades.get("alpha_return", pd.Series(index=trades.index, dtype=float)), errors="coerce").dropna()
    daily_alpha = pd.to_numeric(
        trades.get("daily_alpha_return", pd.Series(index=trades.index, dtype=float)),
        errors="coerce",
    ).dropna()
    sharpe_alpha = (
        float(alpha.mean() / alpha.std(ddof=1))
        if len(alpha) > 1 and float(alpha.std(ddof=1)) > 0
        else float("nan")
    )
    sharpe_alpha_daily = (
        float((daily_alpha.mean() / daily_alpha.std(ddof=1)) * math.sqrt(252.0))
        if len(daily_alpha) > 1 and float(daily_alpha.std(ddof=1)) > 0
        else float("nan")
    )
    tail_n = max(1, int(len(r) * 0.10))
    sorted_r = np.sort(r.to_numpy())
    return {
        "n_trades": float(len(r)),
        "mean_net_return": float(r.mean()),
        "median_net_return": float(r.median()),
        "win_rate_net": float((r > 0).mean()),
        "t_stat_net": _t_stat(r),
        "sharpe_per_trade_net": sharpe_trade,
        "annualized_sharpe_daily_proxy_net": sharpe_daily,
        "mean_hold_days": float(h.mean()) if len(h) else float("nan"),
        "p05_net_return": float(r.quantile(0.05)),
        "cvar10_net_return": float(sorted_r[:tail_n].mean()),
        "worst_net_return": float(r.min()),
        "mean_alpha_return": float(alpha.mean()) if len(alpha) else float("nan"),
        "win_rate_alpha": float((alpha > 0).mean()) if len(alpha) else float("nan"),
        "t_stat_alpha": _t_stat(alpha) if len(alpha) else float("nan"),
        "sharpe_per_trade_alpha": sharpe_alpha,
        "annualized_sharpe_daily_proxy_alpha": sharpe_alpha_daily,
    }


def _walkforward_yearly_purged(test_panel: pd.DataFrame, cfg: RobustnessConfig) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for asset, g in test_panel.groupby("asset", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        g["year"] = g["timestamp"].dt.year
        for year, gy in g.groupby("year", sort=True):
            if gy.empty:
                continue
            year_start = pd.Timestamp(gy["timestamp"].min())
            purge_cut = year_start + pd.Timedelta(days=cfg.purge_days)
            gy = gy[gy["timestamp"] >= purge_cut].copy()
            if gy.empty:
                continue
            trades = _simulate_non_overlapping_trades(gy.drop(columns=["year"]), cfg)
            if trades.empty:
                continue
            stats = _summarize_trades(trades)
            rows.append({"asset": asset, "year": int(year), **stats})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.groupby("year", as_index=False)
        .agg(
            n_assets=("asset", "nunique"),
            n_trades=("n_trades", "sum"),
            mean_net_return=("mean_net_return", "mean"),
            median_net_return=("median_net_return", "median"),
            win_rate_net=("win_rate_net", "mean"),
            t_stat_net=("t_stat_net", "mean"),
            sharpe_per_trade_net=("sharpe_per_trade_net", "mean"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )


def _build_market_regimes(test_panel: pd.DataFrame, cfg: RobustnessConfig) -> pd.DataFrame:
    pvt = test_panel.pivot(index="timestamp", columns="asset", values="close").sort_index()
    rets = pvt.pct_change()
    mkt_ret = rets.mean(axis=1, skipna=True)
    out = pd.DataFrame({"timestamp": mkt_ret.index, "mkt_ret": mkt_ret.values}).dropna()

    trend_w = cfg.regime_trend_window_days
    vol_w = cfg.regime_vol_window_days
    out["mkt_trend"] = (1.0 + out["mkt_ret"]).rolling(trend_w, min_periods=trend_w).apply(np.prod, raw=True) - 1.0
    out["mkt_vol"] = out["mkt_ret"].rolling(vol_w, min_periods=vol_w).std()
    vol_med = float(out["mkt_vol"].median(skipna=True))
    out["trend_regime"] = np.where(out["mkt_trend"] >= 0.0, "bull", "bear")
    out["vol_regime"] = np.where(out["mkt_vol"] >= vol_med, "high_vol", "low_vol")
    return out[["timestamp", "trend_regime", "vol_regime"]].dropna().reset_index(drop=True)


def _regime_trade_performance(trades: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or regimes.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["entry_timestamp"] = pd.to_datetime(t["entry_timestamp"], utc=True, errors="coerce")
    t = t.merge(regimes, left_on="entry_timestamp", right_on="timestamp", how="left")
    t = t.dropna(subset=["trend_regime", "vol_regime", "net_return"])
    if t.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | str]] = []
    for regime_col in ["trend_regime", "vol_regime"]:
        for regime, g in t.groupby(regime_col):
            r = pd.to_numeric(g["net_return"], errors="coerce").dropna()
            if r.empty:
                continue
            rows.append(
                {
                    "regime_type": regime_col,
                    "regime": str(regime),
                    "n_trades": float(len(r)),
                    "mean_net_return": float(r.mean()),
                    "win_rate_net": float((r > 0).mean()),
                    "t_stat_net": _t_stat(r),
                    "sharpe_per_trade_net": float(r.mean() / r.std(ddof=1)) if len(r) > 1 and float(r.std(ddof=1)) > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(["regime_type", "regime"]).reset_index(drop=True)


def _bootstrap_and_permutation_tests(trades: pd.DataFrame, cfg: RobustnessConfig) -> tuple[dict[str, float], pd.DataFrame]:
    r = pd.to_numeric(trades["net_return"], errors="coerce").dropna().to_numpy()
    if len(r) < 5:
        return {
            "obs_mean_net_return": float("nan"),
            "bootstrap_ci_low_95": float("nan"),
            "bootstrap_ci_high_95": float("nan"),
            "bootstrap_p_mean_le_zero": float("nan"),
            "signflip_p_two_sided": float("nan"),
        }, pd.DataFrame()

    rng = np.random.default_rng(cfg.random_state)
    obs_mean = float(np.mean(r))

    boot_means = np.empty(cfg.bootstrap_iterations, dtype=float)
    n = len(r)
    for i in range(cfg.bootstrap_iterations):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(np.mean(r[idx]))

    ci_low = float(np.quantile(boot_means, 0.025))
    ci_high = float(np.quantile(boot_means, 0.975))
    p_boot = float(np.mean(boot_means <= 0.0))

    signflip_null = np.empty(cfg.signflip_iterations, dtype=float)
    for i in range(cfg.signflip_iterations):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        signflip_null[i] = float(np.mean(r * signs))
    p_signflip = float(np.mean(np.abs(signflip_null) >= abs(obs_mean)))

    max_len = max(len(boot_means), len(signflip_null))
    boot_series = pd.Series(boot_means, dtype=float).reindex(range(max_len))
    sign_series = pd.Series(signflip_null, dtype=float).reindex(range(max_len))
    null_df = pd.DataFrame({"bootstrap_mean": boot_series, "signflip_mean": sign_series})
    tests = {
        "obs_mean_net_return": obs_mean,
        "bootstrap_ci_low_95": ci_low,
        "bootstrap_ci_high_95": ci_high,
        "bootstrap_p_mean_le_zero": p_boot,
        "signflip_p_two_sided": p_signflip,
    }
    return tests, null_df


def _alpha_bootstrap_and_permutation_tests(
    trades: pd.DataFrame, cfg: RobustnessConfig
) -> tuple[dict[str, float], pd.DataFrame]:
    r = pd.to_numeric(trades.get("alpha_return"), errors="coerce").dropna().to_numpy()
    if len(r) < 5:
        return {
            "obs_mean_alpha_return": float("nan"),
            "alpha_bootstrap_ci_low_95": float("nan"),
            "alpha_bootstrap_ci_high_95": float("nan"),
            "alpha_bootstrap_p_mean_le_zero": float("nan"),
            "alpha_signflip_p_two_sided": float("nan"),
        }, pd.DataFrame()

    rng = np.random.default_rng(cfg.random_state + 7)
    obs_mean = float(np.mean(r))
    n = len(r)
    boot_means = np.empty(cfg.bootstrap_iterations, dtype=float)
    for i in range(cfg.bootstrap_iterations):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(np.mean(r[idx]))
    ci_low = float(np.quantile(boot_means, 0.025))
    ci_high = float(np.quantile(boot_means, 0.975))
    p_boot = float(np.mean(boot_means <= 0.0))

    signflip_null = np.empty(cfg.signflip_iterations, dtype=float)
    for i in range(cfg.signflip_iterations):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        signflip_null[i] = float(np.mean(r * signs))
    p_signflip = float(np.mean(np.abs(signflip_null) >= abs(obs_mean)))

    max_len = max(len(boot_means), len(signflip_null))
    boot_series = pd.Series(boot_means, dtype=float).reindex(range(max_len))
    sign_series = pd.Series(signflip_null, dtype=float).reindex(range(max_len))
    null_df = pd.DataFrame({"alpha_bootstrap_mean": boot_series, "alpha_signflip_mean": sign_series})
    out = {
        "obs_mean_alpha_return": obs_mean,
        "alpha_bootstrap_ci_low_95": ci_low,
        "alpha_bootstrap_ci_high_95": ci_high,
        "alpha_bootstrap_p_mean_le_zero": p_boot,
        "alpha_signflip_p_two_sided": p_signflip,
    }
    return out, null_df


def _market_neutral_random_entry_permutation_test(
    test_panel: pd.DataFrame,
    benchmark_close: pd.Series,
    observed_mean_alpha: float,
    cfg: RobustnessConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_state + 101)
    rt_cost = _roundtrip_cost_return(cfg)
    horizon = cfg.fixed_horizon_days

    counts_by_asset = (
        test_panel.groupby("asset", as_index=False)["entry_long"].sum().rename(columns={"entry_long": "n_entries"})
    )
    counts_by_asset["n_entries"] = counts_by_asset["n_entries"].astype(int)
    counts_map = dict(zip(counts_by_asset["asset"], counts_by_asset["n_entries"]))

    null_means = np.empty(cfg.random_entry_iterations, dtype=float)
    for i in range(cfg.random_entry_iterations):
        sim_alpha_returns: list[float] = []
        for asset, g in test_panel.groupby("asset", sort=False):
            n_entries = int(counts_map.get(asset, 0))
            if n_entries <= 0:
                continue
            df = g.sort_values("timestamp").reset_index(drop=True)
            max_entry_idx = len(df) - horizon - 1
            if max_entry_idx <= 1:
                continue
            candidates = np.arange(1, max_entry_idx + 1)
            replace = len(candidates) < n_entries
            picks = rng.choice(candidates, size=n_entries, replace=replace)
            exit_idx = picks + horizon
            entry_px = pd.to_numeric(df.loc[picks, "close"], errors="coerce").to_numpy()
            exit_px = pd.to_numeric(df.loc[exit_idx, "close"], errors="coerce").to_numpy()
            gross = exit_px / entry_px - 1.0
            net = gross - rt_cost
            entry_ts = pd.to_datetime(df.loc[picks, "timestamp"], utc=True, errors="coerce")
            exit_ts = pd.to_datetime(df.loc[exit_idx, "timestamp"], utc=True, errors="coerce")
            bench = _map_benchmark_returns(
                benchmark_close=benchmark_close,
                entry_ts=entry_ts,
                exit_ts=exit_ts,
            ).to_numpy()
            alpha = net - bench
            sim_alpha_returns.extend(alpha[np.isfinite(alpha)].tolist())
        null_means[i] = float(np.mean(sim_alpha_returns)) if sim_alpha_returns else float("nan")

    null = pd.Series(null_means).dropna()
    p_right = float(np.mean(null >= observed_mean_alpha)) if len(null) else float("nan")
    p_left = float(np.mean(null <= observed_mean_alpha)) if len(null) else float("nan")
    out = {
        "market_neutral_random_entry_null_mean_alpha": float(null.mean()) if len(null) else float("nan"),
        "market_neutral_random_entry_null_std_alpha": float(null.std(ddof=1)) if len(null) > 1 else float("nan"),
        "market_neutral_random_entry_p_right": p_right,
        "market_neutral_random_entry_p_left": p_left,
        "market_neutral_random_entry_effect_zscore": float((observed_mean_alpha - null.mean()) / null.std(ddof=1))
        if len(null) > 1 and float(null.std(ddof=1)) > 0
        else float("nan"),
    }
    return out, pd.DataFrame({"market_neutral_random_entry_null_mean_alpha": null.to_numpy()})


def _build_signal_vs_luck_scorecard(
    overall: dict[str, float],
    walkforward: pd.DataFrame,
    regime_perf: pd.DataFrame,
    significance: dict[str, float],
    alpha_significance: dict[str, float],
    market_neutral_random_test: dict[str, float],
) -> dict[str, object]:
    checks: list[tuple[str, bool, str]] = []

    mret = float(overall.get("mean_net_return", float("nan")))
    checks.append(("Net mean return > 0", bool(np.isfinite(mret) and mret > 0), f"{mret:.6f}"))

    sharpe = float(overall.get("annualized_sharpe_daily_proxy_net", float("nan")))
    checks.append(("Annualized Sharpe proxy > 0.5", bool(np.isfinite(sharpe) and sharpe > 0.5), f"{sharpe:.4f}"))

    if not walkforward.empty:
        pos_ratio = float((walkforward["mean_net_return"] > 0).mean())
    else:
        pos_ratio = float("nan")
    checks.append(("Walk-forward positive years >= 60%", bool(np.isfinite(pos_ratio) and pos_ratio >= 0.6), f"{pos_ratio:.3f}"))

    p_boot = float(significance.get("bootstrap_p_mean_le_zero", float("nan")))
    checks.append(("Bootstrap p(mean<=0) < 0.05", bool(np.isfinite(p_boot) and p_boot < 0.05), f"{p_boot:.4f}"))

    p_sign = float(significance.get("signflip_p_two_sided", float("nan")))
    checks.append(("Sign-flip p(two-sided) < 0.05", bool(np.isfinite(p_sign) and p_sign < 0.05), f"{p_sign:.4f}"))

    mean_alpha = float(overall.get("mean_alpha_return", float("nan")))
    checks.append(("Mean alpha return > 0", bool(np.isfinite(mean_alpha) and mean_alpha > 0), f"{mean_alpha:.6f}"))

    alpha_p_boot = float(alpha_significance.get("alpha_bootstrap_p_mean_le_zero", float("nan")))
    checks.append(
        (
            "Alpha bootstrap p(mean<=0) < 0.05",
            bool(np.isfinite(alpha_p_boot) and alpha_p_boot < 0.05),
            f"{alpha_p_boot:.4f}",
        )
    )

    p_rand = float(market_neutral_random_test.get("market_neutral_random_entry_p_right", float("nan")))
    checks.append(
        (
            "Better than beta-matched random-entry null (p_right < 0.05)",
            bool(np.isfinite(p_rand) and p_rand < 0.05),
            f"{p_rand:.4f}",
        )
    )
    rand_mean = float(market_neutral_random_test.get("market_neutral_random_entry_null_mean_alpha", float("nan")))
    checks.append(
        (
            "Mean alpha return > beta-matched random-entry null mean",
            bool(np.isfinite(mean_alpha) and np.isfinite(rand_mean) and mean_alpha > rand_mean),
            f"obs={mean_alpha:.6f}, null={rand_mean:.6f}",
        )
    )

    bull_ok = bear_ok = False
    hv_ok = lv_ok = False
    if not regime_perf.empty:
        trend = regime_perf[regime_perf["regime_type"] == "trend_regime"].set_index("regime")
        vol = regime_perf[regime_perf["regime_type"] == "vol_regime"].set_index("regime")
        if "bull" in trend.index:
            bull_ok = float(trend.loc["bull", "mean_net_return"]) > 0
        if "bear" in trend.index:
            bear_ok = float(trend.loc["bear", "mean_net_return"]) > 0
        if "high_vol" in vol.index:
            hv_ok = float(vol.loc["high_vol", "mean_net_return"]) > 0
        if "low_vol" in vol.index:
            lv_ok = float(vol.loc["low_vol", "mean_net_return"]) > 0
    checks.append(("Positive in both bull and bear regimes", bool(bull_ok and bear_ok), f"bull={bull_ok}, bear={bear_ok}"))
    checks.append(("Positive in both high-vol and low-vol regimes", bool(hv_ok and lv_ok), f"high_vol={hv_ok}, low_vol={lv_ok}"))

    passed = int(sum(1 for _, ok, _ in checks if ok))
    total = int(len(checks))
    score = passed / total if total > 0 else float("nan")
    critical_random_check = bool(np.isfinite(p_rand) and p_rand < 0.05)
    critical_alpha_check = bool(np.isfinite(mean_alpha) and np.isfinite(rand_mean) and mean_alpha > rand_mean)

    if score >= 0.75 and critical_random_check and critical_alpha_check:
        verdict = "likely_real_signal"
    elif score >= 0.50:
        verdict = "mixed_signal_some_luck"
    else:
        verdict = "likely_luck_or_regime_specific"

    return {
        "score": score,
        "passed_checks": passed,
        "total_checks": total,
        "verdict": verdict,
        "checks": [{"name": name, "passed": ok, "detail": detail} for name, ok, detail in checks],
    }


def main() -> None:
    out_root = Path("trading_research/examples/_output/19_best_strategy_robustness_scorecard")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = RobustnessConfig()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(list(UNIVERSE_90), period="3y", interval="1d")
    benchmark_close = _build_benchmark_close_series(vendor)
    if bars.empty:
        raise ValueError("No Yahoo bars returned for requested universe.")

    panel = _build_panel(bars, cfg)
    train, test, split_ts = _split_frame(panel, cfg.test_ratio)

    all_trades: list[pd.DataFrame] = []
    for _, g in test.groupby("asset", sort=False):
        all_trades.append(_simulate_non_overlapping_trades(g, cfg))
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    trades = _attach_benchmark_alpha(trades, benchmark_close=benchmark_close)
    overall = _summarize_trades(trades)

    walkforward = _walkforward_yearly_purged(test, cfg)
    regimes = _build_market_regimes(test, cfg)
    regime_perf = _regime_trade_performance(trades, regimes)

    significance, null_boot_sign = _bootstrap_and_permutation_tests(trades, cfg)
    alpha_significance, null_alpha_boot_sign = _alpha_bootstrap_and_permutation_tests(trades, cfg)
    market_neutral_random_test, null_random = _market_neutral_random_entry_permutation_test(
        test,
        benchmark_close=benchmark_close,
        observed_mean_alpha=float(overall.get("mean_alpha_return", float("nan"))),
        cfg=cfg,
    )

    scorecard = _build_signal_vs_luck_scorecard(
        overall,
        walkforward,
        regime_perf,
        significance,
        alpha_significance,
        market_neutral_random_test,
    )

    summary = {
        "universe_size_requested": len(UNIVERSE_90),
        "universe_size_fetched": int(bars["asset"].nunique()),
        "start_timestamp": str(panel["timestamp"].min()),
        "end_timestamp": str(panel["timestamp"].max()),
        "split_timestamp": str(split_ts),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "strategy": "long_cross_above_1m_pegged_high_exit_63d",
        "cost_model": {
            "transaction_cost_bps_per_side": cfg.transaction_cost_bps_per_side,
            "slippage_bps_per_side": cfg.slippage_bps_per_side,
            "roundtrip_cost_return": _roundtrip_cost_return(cfg),
        },
        "overall": overall,
        "significance": {**significance, **alpha_significance, **market_neutral_random_test},
        "scorecard": scorecard,
    }

    trades.to_csv(reports_dir / "trades_oos.csv", index=False)
    walkforward.to_csv(reports_dir / "walkforward_yearly_purged.csv", index=False)
    regime_perf.to_csv(reports_dir / "regime_performance.csv", index=False)
    null_boot_sign.to_csv(reports_dir / "null_bootstrap_signflip.csv", index=False)
    null_alpha_boot_sign.to_csv(reports_dir / "null_alpha_bootstrap_signflip.csv", index=False)
    null_random.to_csv(reports_dir / "null_market_neutral_random_entry.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (reports_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("universe fetched:", summary["universe_size_fetched"])
    print("test rows:", summary["test_rows"], "split timestamp:", summary["split_timestamp"])
    print("overall:", json.dumps(overall, indent=2))
    print("significance:", json.dumps({**significance, **alpha_significance, **market_neutral_random_test}, indent=2))
    print("scorecard:", json.dumps(scorecard, indent=2))


if __name__ == "__main__":
    main()

