"""Alpha-seeking upgrade for pegged-high breakout strategy.

Goal:
Improve the prior breakout strategy by adding train-calibrated filters:
1) Relative strength filter vs SPY (63d excess momentum threshold),
2) Optional market regime gate (SPY above 200d moving average),
3) Exit-horizon selection from a small candidate set.

Then evaluate strict OOS alpha quality on a 90-asset universe:
- net return and alpha (vs SPY over matched holding windows),
- bootstrap/sign-flip significance on alpha,
- beta-matched random-entry null on alpha,
- walk-forward and regime diagnostics,
- final "real alpha vs luck" scorecard.
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
    "PM",
    "UNP",
    "RTX",
    "IBM",
    "INTU",
    "SBUX",
    "BLK",
    "MDT",
    "AMAT",
    "GE",
    "ISRG",
    "BKNG",
    "GILD",
    "DE",
    "MMM",
    "AXP",
    "SPGI",
    "PLD",
    "LRCX",
    "CB",
    "T",
    "ADP",
    "MO",
    "CI",
    "ETN",
    "SYK",
    "C",
    "SCHW",
    "TMUS",
    "PGR",
    "MU",
    "ELV",
    "ZTS",
    "USB",
    "SO",
    "DUK",
    "AON",
    "PANW",
    "KLAC",
    "SNPS",
]

UNIVERSE_90 = tuple(dict.fromkeys([*UNIVERSE_50, *EXTRA_UNIVERSE_40]))


@dataclass(frozen=True)
class UpgradeConfig:
    test_ratio: float = 0.30
    peg_window_days: int = 21
    momentum_lookback_days: int = 63
    spy_trend_ma_days: int = 200
    default_exit_horizon_days: int = 63
    candidate_exit_horizons: tuple[int, ...] = (42, 63, 84)
    candidate_rel_mom_thresholds: tuple[float, ...] = (-0.02, 0.0, 0.03, 0.05, 0.08, 0.10, 0.12)
    candidate_use_spy_gate: tuple[bool, ...] = (False, True)
    min_train_trades: int = 80
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    bootstrap_iterations: int = 2500
    signflip_iterations: int = 4000
    random_entry_iterations: int = 1000
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


def _roundtrip_cost_return(cfg: UpgradeConfig) -> float:
    total_bps = 2.0 * (cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side)
    return total_bps / 10_000.0


def _build_spy_series(vendor: YahooMarketDataVendor) -> pd.DataFrame:
    spy = vendor.fetch_bars(["SPY"], period="3y", interval="1d")
    if spy.empty:
        raise ValueError("Failed to load SPY benchmark bars.")
    out = spy[["timestamp", "close"]].rename(columns={"close": "spy_close"}).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["spy_close"] = pd.to_numeric(out["spy_close"], errors="coerce")
    out = out.dropna(subset=["timestamp", "spy_close"]).sort_values("timestamp").drop_duplicates("timestamp")
    return out


def _build_panel(bars: pd.DataFrame, spy_series: pd.DataFrame, cfg: UpgradeConfig) -> pd.DataFrame:
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "asset", "close", "high"]).sort_values(["asset", "timestamp"])
    frame["asset"] = frame["asset"].astype(str)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["high"] = pd.to_numeric(frame["high"], errors="coerce")
    frame = frame.dropna(subset=["close", "high"])
    frame = frame[frame["close"] > 0].copy()

    look = cfg.peg_window_days
    mom_look = cfg.momentum_lookback_days
    g = frame.groupby("asset", sort=False)
    frame["pegged_high_1m"] = g["high"].rolling(look, min_periods=look).max().shift(1).reset_index(level=0, drop=True)
    prev_close = g["close"].shift(1)
    frame["entry_base"] = ((prev_close <= frame["pegged_high_1m"]) & (frame["close"] > frame["pegged_high_1m"])).astype(int)
    frame["asset_ret_mom"] = g["close"].pct_change(mom_look)

    spy = spy_series.copy()
    spy = spy.sort_values("timestamp")
    spy["spy_ret_mom"] = spy["spy_close"].pct_change(mom_look)
    spy["spy_ma"] = spy["spy_close"].rolling(cfg.spy_trend_ma_days, min_periods=cfg.spy_trend_ma_days).mean()
    spy["spy_up"] = (spy["spy_close"] > spy["spy_ma"]).astype(int)
    frame = frame.merge(spy[["timestamp", "spy_close", "spy_ret_mom", "spy_up"]], on="timestamp", how="left")
    frame["rel_mom_63"] = frame["asset_ret_mom"] - frame["spy_ret_mom"]

    keep = ["timestamp", "asset", "close", "entry_base", "rel_mom_63", "spy_up", "spy_close"]
    out = frame[keep].dropna().sort_values(["timestamp", "asset"]).reset_index(drop=True)
    return out


def _simulate_non_overlapping_trades(
    asset_df: pd.DataFrame,
    *,
    signal_col: str,
    horizon_days: int,
    cfg: UpgradeConfig,
) -> pd.DataFrame:
    df = asset_df.sort_values("timestamp").reset_index(drop=True)
    rt_cost = _roundtrip_cost_return(cfg)
    trades: list[dict[str, float | str]] = []

    i = 1
    n = len(df)
    while i < n - 1:
        if int(df.loc[i, signal_col]) != 1:
            i += 1
            continue

        entry_idx = i
        exit_idx = min(entry_idx + horizon_days, n - 1)
        if exit_idx <= entry_idx:
            i += 1
            continue

        entry_px = float(df.loc[entry_idx, "close"])
        exit_px = float(df.loc[exit_idx, "close"])
        if not (np.isfinite(entry_px) and np.isfinite(exit_px) and entry_px > 0):
            i = exit_idx + 1
            continue

        gross_ret = exit_px / entry_px - 1.0
        net_ret = gross_ret - rt_cost

        spy_entry = float(df.loc[entry_idx, "spy_close"])
        spy_exit = float(df.loc[exit_idx, "spy_close"])
        bench_net = float("nan")
        alpha_ret = float("nan")
        if np.isfinite(spy_entry) and np.isfinite(spy_exit) and spy_entry > 0:
            bench_net = spy_exit / spy_entry - 1.0 - rt_cost
            alpha_ret = net_ret - bench_net

        hold_days = int(exit_idx - entry_idx)
        daily_log_net = np.log1p(net_ret) / hold_days if hold_days > 0 and net_ret > -1.0 else float("nan")
        daily_log_alpha = (
            np.log1p(alpha_ret) / hold_days if hold_days > 0 and np.isfinite(alpha_ret) and alpha_ret > -1.0 else float("nan")
        )

        trades.append(
            {
                "asset": str(df.loc[entry_idx, "asset"]),
                "entry_timestamp": str(pd.Timestamp(df.loc[entry_idx, "timestamp"])),
                "exit_timestamp": str(pd.Timestamp(df.loc[exit_idx, "timestamp"])),
                "entry_idx": float(entry_idx),
                "exit_idx": float(exit_idx),
                "hold_days": float(hold_days),
                "net_return": float(net_ret),
                "benchmark_net_return": float(bench_net),
                "alpha_return": float(alpha_ret),
                "daily_log_net_return": float(daily_log_net),
                "daily_log_alpha_return": float(daily_log_alpha),
            }
        )
        i = exit_idx + 1

    return pd.DataFrame(trades)


def _summarize_trades(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "n_trades": 0.0,
            "mean_net_return": float("nan"),
            "win_rate_net": float("nan"),
            "t_stat_net": float("nan"),
            "sharpe_trade_net": float("nan"),
            "annualized_sharpe_daily_net": float("nan"),
            "mean_alpha_return": float("nan"),
            "win_rate_alpha": float("nan"),
            "t_stat_alpha": float("nan"),
            "sharpe_trade_alpha": float("nan"),
            "annualized_sharpe_daily_alpha": float("nan"),
            "mean_hold_days": float("nan"),
            "worst_net_return": float("nan"),
            "cvar10_net_return": float("nan"),
        }

    r = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    a = pd.to_numeric(trades["alpha_return"], errors="coerce").dropna()
    dlog = pd.to_numeric(trades["daily_log_net_return"], errors="coerce").dropna()
    dlog_a = pd.to_numeric(trades["daily_log_alpha_return"], errors="coerce").dropna()
    h = pd.to_numeric(trades["hold_days"], errors="coerce").dropna()

    sharpe_trade_net = float(r.mean() / r.std(ddof=1)) if len(r) > 1 and float(r.std(ddof=1)) > 0 else float("nan")
    sharpe_trade_alpha = float(a.mean() / a.std(ddof=1)) if len(a) > 1 and float(a.std(ddof=1)) > 0 else float("nan")
    sharpe_daily_net = (
        float((dlog.mean() / dlog.std(ddof=1)) * math.sqrt(252.0)) if len(dlog) > 1 and float(dlog.std(ddof=1)) > 0 else float("nan")
    )
    sharpe_daily_alpha = (
        float((dlog_a.mean() / dlog_a.std(ddof=1)) * math.sqrt(252.0))
        if len(dlog_a) > 1 and float(dlog_a.std(ddof=1)) > 0
        else float("nan")
    )

    tail_n = max(1, int(len(r) * 0.10))
    sorted_r = np.sort(r.to_numpy())
    return {
        "n_trades": float(len(r)),
        "mean_net_return": float(r.mean()),
        "win_rate_net": float((r > 0).mean()),
        "t_stat_net": _t_stat(r),
        "sharpe_trade_net": sharpe_trade_net,
        "annualized_sharpe_daily_net": sharpe_daily_net,
        "mean_alpha_return": float(a.mean()) if len(a) else float("nan"),
        "win_rate_alpha": float((a > 0).mean()) if len(a) else float("nan"),
        "t_stat_alpha": _t_stat(a),
        "sharpe_trade_alpha": sharpe_trade_alpha,
        "annualized_sharpe_daily_alpha": sharpe_daily_alpha,
        "mean_hold_days": float(h.mean()) if len(h) else float("nan"),
        "worst_net_return": float(r.min()) if len(r) else float("nan"),
        "cvar10_net_return": float(sorted_r[:tail_n].mean()) if len(r) else float("nan"),
    }


def _build_signal(df: pd.DataFrame, *, rel_mom_threshold: float, use_spy_gate: bool) -> pd.Series:
    sig = (df["entry_base"] == 1) & (pd.to_numeric(df["rel_mom_63"], errors="coerce") >= rel_mom_threshold)
    if use_spy_gate:
        sig = sig & (pd.to_numeric(df["spy_up"], errors="coerce").fillna(0) == 1)
    return sig.astype(int)


def _simulate_strategy(
    panel: pd.DataFrame,
    *,
    rel_mom_threshold: float,
    use_spy_gate: bool,
    horizon_days: int,
    cfg: UpgradeConfig,
) -> pd.DataFrame:
    out_frames: list[pd.DataFrame] = []
    for _, g in panel.groupby("asset", sort=False):
        g = g.sort_values("timestamp").copy()
        g["signal"] = _build_signal(g, rel_mom_threshold=rel_mom_threshold, use_spy_gate=use_spy_gate)
        out_frames.append(
            _simulate_non_overlapping_trades(
                g,
                signal_col="signal",
                horizon_days=horizon_days,
                cfg=cfg,
            )
        )
    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()


def _tune_on_train(train_panel: pd.DataFrame, cfg: UpgradeConfig) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, float | int | bool]] = []
    for thr in cfg.candidate_rel_mom_thresholds:
        for gate in cfg.candidate_use_spy_gate:
            for horizon in cfg.candidate_exit_horizons:
                tr = _simulate_strategy(
                    train_panel,
                    rel_mom_threshold=float(thr),
                    use_spy_gate=bool(gate),
                    horizon_days=int(horizon),
                    cfg=cfg,
                )
                s = _summarize_trades(tr)
                n = int(s.get("n_trades", 0.0))
                mean_alpha = float(s.get("mean_alpha_return", float("nan")))
                sharpe_alpha = float(s.get("sharpe_trade_alpha", float("nan")))
                # Favor alpha, lightly favor sharper alpha; enforce minimum trade count.
                score = float("nan")
                if n >= cfg.min_train_trades and np.isfinite(mean_alpha):
                    score = mean_alpha + 0.0025 * (sharpe_alpha if np.isfinite(sharpe_alpha) else 0.0)
                rows.append(
                    {
                        "rel_mom_threshold": float(thr),
                        "use_spy_gate": bool(gate),
                        "horizon_days": int(horizon),
                        "n_trades_train": n,
                        "mean_alpha_train": mean_alpha,
                        "mean_net_train": float(s.get("mean_net_return", float("nan"))),
                        "sharpe_alpha_train": sharpe_alpha,
                        "score": score,
                    }
                )
    grid = pd.DataFrame(rows)
    viable = grid[np.isfinite(grid["score"])].copy()
    if viable.empty:
        # Fallback to baseline params.
        best = {"rel_mom_threshold": 0.0, "use_spy_gate": False, "horizon_days": cfg.default_exit_horizon_days}
        return best, grid.sort_values(["n_trades_train"], ascending=False).reset_index(drop=True)
    viable = viable.sort_values(["score", "mean_alpha_train", "n_trades_train"], ascending=[False, False, False]).reset_index(
        drop=True
    )
    top = viable.iloc[0]
    best = {
        "rel_mom_threshold": float(top["rel_mom_threshold"]),
        "use_spy_gate": bool(top["use_spy_gate"]),
        "horizon_days": int(top["horizon_days"]),
    }
    return best, grid.sort_values(["score", "mean_alpha_train"], ascending=[False, False]).reset_index(drop=True)


def _bootstrap_signflip(series: pd.Series, *, iterations_boot: int, iterations_sign: int, seed: int, prefix: str) -> tuple[dict[str, float], pd.DataFrame]:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(x) < 5:
        return {
            f"obs_mean_{prefix}": float("nan"),
            f"bootstrap_ci_low_95_{prefix}": float("nan"),
            f"bootstrap_ci_high_95_{prefix}": float("nan"),
            f"bootstrap_p_mean_le_zero_{prefix}": float("nan"),
            f"signflip_p_two_sided_{prefix}": float("nan"),
        }, pd.DataFrame()

    rng = np.random.default_rng(seed)
    n = len(x)
    obs = float(np.mean(x))
    boot = np.empty(iterations_boot, dtype=float)
    for i in range(iterations_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(x[idx]))
    sign = np.empty(iterations_sign, dtype=float)
    for i in range(iterations_sign):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        sign[i] = float(np.mean(x * signs))

    stats = {
        f"obs_mean_{prefix}": obs,
        f"bootstrap_ci_low_95_{prefix}": float(np.quantile(boot, 0.025)),
        f"bootstrap_ci_high_95_{prefix}": float(np.quantile(boot, 0.975)),
        f"bootstrap_p_mean_le_zero_{prefix}": float(np.mean(boot <= 0.0)),
        f"signflip_p_two_sided_{prefix}": float(np.mean(np.abs(sign) >= abs(obs))),
    }
    max_len = max(len(boot), len(sign))
    out = pd.DataFrame(
        {
            f"bootstrap_mean_{prefix}": pd.Series(boot, dtype=float).reindex(range(max_len)),
            f"signflip_mean_{prefix}": pd.Series(sign, dtype=float).reindex(range(max_len)),
        }
    )
    return stats, out


def _beta_matched_random_entry_null(
    test_panel: pd.DataFrame,
    *,
    observed_mean_alpha: float,
    n_entries_by_asset: dict[str, int],
    horizon_days: int,
    cfg: UpgradeConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_state + 1007)
    rt_cost = _roundtrip_cost_return(cfg)
    null_means = np.empty(cfg.random_entry_iterations, dtype=float)
    for i in range(cfg.random_entry_iterations):
        sim_alpha: list[float] = []
        for asset, g in test_panel.groupby("asset", sort=False):
            n_entries = int(n_entries_by_asset.get(asset, 0))
            if n_entries <= 0:
                continue
            df = g.sort_values("timestamp").reset_index(drop=True)
            max_entry_idx = len(df) - horizon_days - 1
            if max_entry_idx <= 1:
                continue
            candidates = np.arange(1, max_entry_idx + 1)
            replace = len(candidates) < n_entries
            picks = rng.choice(candidates, size=n_entries, replace=replace)
            exits = picks + horizon_days
            entry_px = pd.to_numeric(df.loc[picks, "close"], errors="coerce").to_numpy()
            exit_px = pd.to_numeric(df.loc[exits, "close"], errors="coerce").to_numpy()
            spy_entry = pd.to_numeric(df.loc[picks, "spy_close"], errors="coerce").to_numpy()
            spy_exit = pd.to_numeric(df.loc[exits, "spy_close"], errors="coerce").to_numpy()

            net = exit_px / entry_px - 1.0 - rt_cost
            bench = spy_exit / spy_entry - 1.0 - rt_cost
            alpha = net - bench
            good = np.isfinite(alpha)
            if good.any():
                sim_alpha.extend(alpha[good].tolist())
        null_means[i] = float(np.mean(sim_alpha)) if sim_alpha else float("nan")

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


def _walkforward_yearly_purged(test_panel: pd.DataFrame, cfg: UpgradeConfig, params: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    horizon = int(params["horizon_days"])
    for asset, g in test_panel.groupby("asset", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        g["year"] = g["timestamp"].dt.year
        for year, gy in g.groupby("year", sort=True):
            if gy.empty:
                continue
            purge_cut = pd.Timestamp(gy["timestamp"].min()) + pd.Timedelta(days=horizon)
            gy = gy[gy["timestamp"] >= purge_cut].copy()
            if gy.empty:
                continue
            tr = _simulate_strategy(
                gy.drop(columns=["year"]),
                rel_mom_threshold=float(params["rel_mom_threshold"]),
                use_spy_gate=bool(params["use_spy_gate"]),
                horizon_days=horizon,
                cfg=cfg,
            )
            if tr.empty:
                continue
            s = _summarize_trades(tr)
            rows.append(
                {
                    "asset": asset,
                    "year": int(year),
                    "n_trades": float(s["n_trades"]),
                    "mean_net_return": float(s["mean_net_return"]),
                    "mean_alpha_return": float(s["mean_alpha_return"]),
                    "win_rate_net": float(s["win_rate_net"]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.groupby("year", as_index=False)
        .agg(
            n_assets=("asset", "nunique"),
            n_trades=("n_trades", "sum"),
            mean_net_return=("mean_net_return", "mean"),
            mean_alpha_return=("mean_alpha_return", "mean"),
            win_rate_net=("win_rate_net", "mean"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )


def _regime_performance(test_panel: pd.DataFrame, trades: pd.DataFrame, cfg: UpgradeConfig) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    pvt = test_panel.pivot(index="timestamp", columns="asset", values="close").sort_index()
    rets = pvt.pct_change()
    mkt_ret = rets.mean(axis=1, skipna=True)
    reg = pd.DataFrame({"timestamp": mkt_ret.index, "mkt_ret": mkt_ret.values}).dropna()
    reg["mkt_trend"] = (1.0 + reg["mkt_ret"]).rolling(cfg.momentum_lookback_days, min_periods=cfg.momentum_lookback_days).apply(
        np.prod, raw=True
    ) - 1.0
    reg["mkt_vol"] = reg["mkt_ret"].rolling(cfg.peg_window_days, min_periods=cfg.peg_window_days).std()
    vol_med = float(reg["mkt_vol"].median(skipna=True))
    reg["trend_regime"] = np.where(reg["mkt_trend"] >= 0.0, "bull", "bear")
    reg["vol_regime"] = np.where(reg["mkt_vol"] >= vol_med, "high_vol", "low_vol")
    reg = reg[["timestamp", "trend_regime", "vol_regime"]].dropna()

    t = trades.copy()
    t["entry_timestamp"] = pd.to_datetime(t["entry_timestamp"], utc=True, errors="coerce")
    t = t.merge(reg, left_on="entry_timestamp", right_on="timestamp", how="left").dropna(
        subset=["trend_regime", "vol_regime", "net_return", "alpha_return"]
    )
    if t.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | str]] = []
    for rcol in ("trend_regime", "vol_regime"):
        for regime, g in t.groupby(rcol):
            net = pd.to_numeric(g["net_return"], errors="coerce").dropna()
            alpha = pd.to_numeric(g["alpha_return"], errors="coerce").dropna()
            if net.empty:
                continue
            rows.append(
                {
                    "regime_type": rcol,
                    "regime": str(regime),
                    "n_trades": float(len(net)),
                    "mean_net_return": float(net.mean()),
                    "mean_alpha_return": float(alpha.mean()) if len(alpha) else float("nan"),
                    "win_rate_net": float((net > 0).mean()),
                    "win_rate_alpha": float((alpha > 0).mean()) if len(alpha) else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(["regime_type", "regime"]).reset_index(drop=True)


def _build_scorecard(
    tuned_summary: dict[str, float],
    walkforward: pd.DataFrame,
    regime: pd.DataFrame,
    alpha_sig: dict[str, float],
    mn_random: dict[str, float],
) -> dict[str, object]:
    checks: list[tuple[str, bool, str]] = []
    mnet = float(tuned_summary.get("mean_net_return", float("nan")))
    malpha = float(tuned_summary.get("mean_alpha_return", float("nan")))
    checks.append(("Net mean return > 0", bool(np.isfinite(mnet) and mnet > 0), f"{mnet:.6f}"))
    checks.append(("Alpha mean return > 0", bool(np.isfinite(malpha) and malpha > 0), f"{malpha:.6f}"))
    asharpe = float(tuned_summary.get("annualized_sharpe_daily_alpha", float("nan")))
    checks.append(("Annualized alpha Sharpe proxy > 0.5", bool(np.isfinite(asharpe) and asharpe > 0.5), f"{asharpe:.4f}"))

    if not walkforward.empty:
        pos_year_ratio = float((walkforward["mean_alpha_return"] > 0).mean())
    else:
        pos_year_ratio = float("nan")
    checks.append(("Walk-forward years with alpha > 0 >= 60%", bool(np.isfinite(pos_year_ratio) and pos_year_ratio >= 0.6), f"{pos_year_ratio:.3f}"))

    p_boot = float(alpha_sig.get("bootstrap_p_mean_le_zero_alpha", float("nan")))
    p_sign = float(alpha_sig.get("signflip_p_two_sided_alpha", float("nan")))
    checks.append(("Alpha bootstrap p(mean<=0) < 0.05", bool(np.isfinite(p_boot) and p_boot < 0.05), f"{p_boot:.4f}"))
    checks.append(("Alpha sign-flip p(two-sided) < 0.05", bool(np.isfinite(p_sign) and p_sign < 0.05), f"{p_sign:.4f}"))

    p_rand = float(mn_random.get("market_neutral_random_entry_p_right", float("nan")))
    null_alpha = float(mn_random.get("market_neutral_random_entry_null_mean_alpha", float("nan")))
    checks.append(("Better than beta-matched random-entry null (p_right < 0.05)", bool(np.isfinite(p_rand) and p_rand < 0.05), f"{p_rand:.4f}"))
    checks.append(
        (
            "Mean alpha > beta-matched random-entry null mean",
            bool(np.isfinite(malpha) and np.isfinite(null_alpha) and malpha > null_alpha),
            f"obs={malpha:.6f}, null={null_alpha:.6f}",
        )
    )

    bull_ok = bear_ok = hv_ok = lv_ok = False
    if not regime.empty:
        tr = regime[regime["regime_type"] == "trend_regime"].set_index("regime")
        vr = regime[regime["regime_type"] == "vol_regime"].set_index("regime")
        if "bull" in tr.index:
            bull_ok = float(tr.loc["bull", "mean_alpha_return"]) > 0
        if "bear" in tr.index:
            bear_ok = float(tr.loc["bear", "mean_alpha_return"]) > 0
        if "high_vol" in vr.index:
            hv_ok = float(vr.loc["high_vol", "mean_alpha_return"]) > 0
        if "low_vol" in vr.index:
            lv_ok = float(vr.loc["low_vol", "mean_alpha_return"]) > 0
    checks.append(("Positive alpha in both bull and bear regimes", bool(bull_ok and bear_ok), f"bull={bull_ok}, bear={bear_ok}"))
    checks.append(("Positive alpha in both high-vol and low-vol regimes", bool(hv_ok and lv_ok), f"high_vol={hv_ok}, low_vol={lv_ok}"))

    passed = int(sum(1 for _, ok, _ in checks if ok))
    total = int(len(checks))
    score = passed / total if total else float("nan")
    critical = bool(np.isfinite(p_rand) and p_rand < 0.05 and np.isfinite(malpha) and np.isfinite(null_alpha) and malpha > null_alpha)
    if score >= 0.75 and critical:
        verdict = "likely_real_alpha"
    elif score >= 0.50:
        verdict = "mixed_signal_some_luck"
    else:
        verdict = "likely_luck_or_beta_exposure"
    return {
        "score": score,
        "passed_checks": passed,
        "total_checks": total,
        "verdict": verdict,
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
    }


def main() -> None:
    out_root = Path("trading_research/examples/_output/20_alpha_upgrade_scorecard")
    reports = out_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    cfg = UpgradeConfig()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(list(UNIVERSE_90), period="3y", interval="1d")
    if bars.empty:
        raise ValueError("No Yahoo bars returned for requested universe.")
    spy = _build_spy_series(vendor)
    panel = _build_panel(bars, spy, cfg)
    train, test, split_ts = _split_frame(panel, cfg.test_ratio)

    best_params, train_grid = _tune_on_train(train, cfg)
    tuned_trades = _simulate_strategy(
        test,
        rel_mom_threshold=float(best_params["rel_mom_threshold"]),
        use_spy_gate=bool(best_params["use_spy_gate"]),
        horizon_days=int(best_params["horizon_days"]),
        cfg=cfg,
    )
    baseline_trades = _simulate_strategy(
        test,
        rel_mom_threshold=-1e9,
        use_spy_gate=False,
        horizon_days=cfg.default_exit_horizon_days,
        cfg=cfg,
    )

    tuned_summary = _summarize_trades(tuned_trades)
    baseline_summary = _summarize_trades(baseline_trades)

    alpha_sig, null_alpha = _bootstrap_signflip(
        tuned_trades.get("alpha_return", pd.Series(dtype=float)),
        iterations_boot=cfg.bootstrap_iterations,
        iterations_sign=cfg.signflip_iterations,
        seed=cfg.random_state + 17,
        prefix="alpha",
    )

    # Match random entry counts to the tuned strategy's observed entry distribution.
    tuned_counts = tuned_trades["asset"].value_counts().to_dict() if not tuned_trades.empty else {}
    mn_random, null_random = _beta_matched_random_entry_null(
        test,
        observed_mean_alpha=float(tuned_summary.get("mean_alpha_return", float("nan"))),
        n_entries_by_asset=tuned_counts,
        horizon_days=int(best_params["horizon_days"]),
        cfg=cfg,
    )

    walkforward = _walkforward_yearly_purged(test, cfg, best_params)
    regime = _regime_performance(test, tuned_trades, cfg)
    scorecard = _build_scorecard(tuned_summary, walkforward, regime, alpha_sig, mn_random)

    comparison = pd.DataFrame(
        [
            {"strategy": "baseline_breakout_63d", **baseline_summary},
            {
                "strategy": "tuned_breakout_relmom_spygate",
                **tuned_summary,
                "rel_mom_threshold": float(best_params["rel_mom_threshold"]),
                "use_spy_gate": bool(best_params["use_spy_gate"]),
                "horizon_days": int(best_params["horizon_days"]),
            },
        ]
    )

    summary = {
        "universe_size_requested": len(UNIVERSE_90),
        "universe_size_fetched": int(bars["asset"].nunique()),
        "split_timestamp": str(split_ts),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "cost_model": {
            "transaction_cost_bps_per_side": cfg.transaction_cost_bps_per_side,
            "slippage_bps_per_side": cfg.slippage_bps_per_side,
            "roundtrip_cost_return": _roundtrip_cost_return(cfg),
        },
        "best_params": best_params,
        "baseline_summary": baseline_summary,
        "tuned_summary": tuned_summary,
        "alpha_significance": {**alpha_sig, **mn_random},
        "scorecard": scorecard,
    }

    train_grid.to_csv(reports / "tuning_grid_train.csv", index=False)
    comparison.to_csv(reports / "strategy_comparison_test.csv", index=False)
    baseline_trades.to_csv(reports / "trades_test_baseline.csv", index=False)
    tuned_trades.to_csv(reports / "trades_test_tuned.csv", index=False)
    walkforward.to_csv(reports / "walkforward_tuned.csv", index=False)
    regime.to_csv(reports / "regime_tuned.csv", index=False)
    null_alpha.to_csv(reports / "null_alpha_bootstrap_signflip_tuned.csv", index=False)
    null_random.to_csv(reports / "null_beta_matched_random_tuned.csv", index=False)
    (reports / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (reports / "scorecard.json").write_text(json.dumps(scorecard, indent=2), encoding="utf-8")

    print("reports:", reports.resolve())
    print("best_params:", best_params)
    print("baseline_summary:", json.dumps(baseline_summary, indent=2))
    print("tuned_summary:", json.dumps(tuned_summary, indent=2))
    print("alpha_significance:", json.dumps({**alpha_sig, **mn_random}, indent=2))
    print("scorecard:", json.dumps(scorecard, indent=2))


if __name__ == "__main__":
    main()
