"""Rigorous validation of baseline_rel2 with larger universe and long history.

baseline_rel2 definition (kept fixed):
- Cross-sectional ranked VWAP opportunity score.
- Enter top-decile names with relative-strength >= +2% vs SPY.
- Hold 63 trading days, non-overlapping per asset.
- Portfolio-level simulation with costs and exposure caps.

Validation protocol:
1) Anchored walk-forward yearly OOS folds (strict chronology),
2) Pooled OOS portfolio metrics (net + alpha vs SPY),
3) Bootstrap/sign-flip significance on OOS alpha,
4) Beta-1 market-neutral random-entry null (same trade counts/holds).
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

EXTRA_UNIVERSE_40_MORE = [
    "AON",
    "APH",
    "CCI",
    "CL",
    "COF",
    "CSX",
    "DD",
    "DOW",
    "DUK",
    "ECL",
    "EMR",
    "EQIX",
    "EW",
    "FDX",
    "FIS",
    "FITB",
    "GD",
    "HCA",
    "ICE",
    "ILMN",
    "KMB",
    "KMI",
    "KLAC",
    "LHX",
    "MAR",
    "MMC",
    "MNST",
    "MSI",
    "NSC",
    "OXY",
    "PSA",
    "REGN",
    "ROP",
    "SHW",
    "SNPS",
    "STZ",
    "TFC",
    "TT",
    "VLO",
    "WMB",
]

EXTRA_UNIVERSE_40_NEW = [
    "ADP",
    "AEP",
    "AFL",
    "AJG",
    "ALL",
    "AMP",
    "APD",
    "AZO",
    "BDX",
    "BIIB",
    "BRO",
    "CDNS",
    "CHD",
    "CPRT",
    "CTAS",
    "D",
    "DG",
    "DLR",
    "EXC",
    "FAST",
    "FANG",
    "GIS",
    "HAL",
    "KR",
    "LEN",
    "LH",
    "MCHP",
    "MCK",
    "MET",
    "NEM",
    "ORLY",
    "PAYX",
    "PEG",
    "PPG",
    "PRU",
    "ROST",
    "SRE",
    "TRV",
    "VRSK",
    "YUM",
]

UNIVERSE_170 = tuple(
    dict.fromkeys([*UNIVERSE_50, *EXTRA_UNIVERSE_40, *EXTRA_UNIVERSE_40_MORE, *EXTRA_UNIVERSE_40_NEW])
)


@dataclass(frozen=True)
class Config:
    period: str = "10y"
    interval: str = "1d"
    signal_lookback_days: int = 63
    rolling_vwap_window: int = 20
    rel_strength_threshold: float = 0.02
    top_quantile: float = 0.90
    hold_days: int = 63
    min_train_years: int = 3
    gross_target: float = 1.0
    max_abs_weight_per_asset: float = 0.04
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    bootstrap_iterations: int = 3000
    signflip_iterations: int = 5000
    random_null_iterations: int = 1200
    random_state: int = 42


def _one_way_cost_return(cfg: Config) -> float:
    return (cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side) / 10_000.0


def _roundtrip_cost_return(cfg: Config) -> float:
    return 2.0 * _one_way_cost_return(cfg)


def _build_panel(bars: pd.DataFrame, spy: pd.DataFrame, cfg: Config) -> pd.DataFrame:
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

    spy = spy.copy()
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True, errors="coerce")
    spy = spy.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    spy["close"] = pd.to_numeric(spy["close"], errors="coerce")
    spy = spy.dropna(subset=["close"]).copy()
    spy["spy_ret_63"] = spy["close"] / spy["close"].shift(look) - 1.0
    spy = spy.rename(columns={"close": "spy_close"})
    frame = frame.merge(spy[["timestamp", "spy_close", "spy_ret_63"]], on="timestamp", how="left")
    frame["rel_strength_63"] = frame["past_return"] - frame["spy_ret_63"]

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["timestamp", "asset", "close", "past_return", "score", "rank_pct", "spy_close", "rel_strength_63"]
    )
    frame["year"] = frame["timestamp"].dt.year
    return frame.reset_index(drop=True)


def _weight_from_active(active: pd.DataFrame, cfg: Config, assets: list[str]) -> pd.Series:
    w = pd.Series(0.0, index=assets, dtype=float)
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


def _build_trades(test_panel: pd.DataFrame, cfg: Config, fold_id: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    all_dates = sorted(test_panel["timestamp"].unique())
    date_to_idx = {ts: i for i, ts in enumerate(all_dates)}
    n_dates = len(all_dates)
    if n_dates < 3:
        return pd.DataFrame()

    for asset, g in test_panel.groupby("asset", sort=False):
        g = g.sort_values("timestamp").copy()
        last_end = -1
        for _, r in g.iterrows():
            rank = float(r["rank_pct"])
            trend_sign = float(np.sign(float(r["past_return"])))
            rel = float(r["rel_strength_63"])
            if rank < cfg.top_quantile or rel < cfg.rel_strength_threshold:
                continue
            if not np.isfinite(trend_sign) or trend_sign == 0.0:
                continue
            sig_idx = date_to_idx.get(pd.Timestamp(r["timestamp"]))
            if sig_idx is None or sig_idx <= last_end:
                continue
            start_idx = sig_idx + 1
            end_idx = min(sig_idx + cfg.hold_days, n_dates - 1)
            if start_idx >= n_dates or end_idx <= start_idx:
                continue
            rows.append(
                {
                    "fold_id": fold_id,
                    "asset": str(asset),
                    "signal_timestamp": str(r["timestamp"]),
                    "start_idx": float(start_idx),
                    "end_idx": float(end_idx),
                    "signal_sign": float(trend_sign),
                    "signal_strength": float(max(1e-6, rank - cfg.top_quantile)),
                    "hold_days": float(end_idx - sig_idx),
                }
            )
            last_end = end_idx
    return pd.DataFrame(rows)


def _simulate_fold(
    test_panel: pd.DataFrame,
    trades: pd.DataFrame,
    spy_df: pd.DataFrame,
    cfg: Config,
    fold_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = test_panel.pivot(index="timestamp", columns="asset", values="close").sort_index().ffill()
    rets = close.pct_change().fillna(0.0)
    dates = rets.index.to_list()
    assets = rets.columns.to_list()
    if len(dates) < 2:
        return pd.DataFrame(), pd.DataFrame()

    spy = spy_df.copy()
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True, errors="coerce")
    spy = spy.dropna(subset=["timestamp"]).sort_values("timestamp")
    spy = spy.set_index("timestamp")["close"].reindex(rets.index).ffill()
    spy_ret = spy.pct_change().fillna(0.0)

    t = trades.copy()
    if t.empty:
        out = pd.DataFrame({"timestamp": rets.index, "net_return": 0.0, "spy_return": spy_ret.values, "alpha_return": -spy_ret.values})
        out["turnover"] = 0.0
        out["gross_exposure"] = 0.0
    else:
        t["start_idx"] = pd.to_numeric(t["start_idx"], errors="coerce").astype("Int64")
        t["end_idx"] = pd.to_numeric(t["end_idx"], errors="coerce").astype("Int64")
        t = t.dropna(subset=["start_idx", "end_idx", "asset", "signal_sign", "signal_strength"])

        prev_w = pd.Series(0.0, index=assets, dtype=float)
        one_way = _one_way_cost_return(cfg)
        rows: list[dict[str, float | str]] = []
        for i in range(1, len(dates)):
            active = t[(t["start_idx"] <= i) & (t["end_idx"] >= i)]
            w = _weight_from_active(active, cfg, assets)
            gross = float(np.dot(w.values, rets.iloc[i].reindex(assets).fillna(0.0).values))
            turnover = float(np.abs(w - prev_w).sum())
            cost = turnover * one_way
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
    out["fold_id"] = fold_id

    trade_rows: list[dict[str, float | str]] = []
    if not t.empty:
        for _, r in t.iterrows():
            sidx = int(r["start_idx"])
            eidx = int(r["end_idx"])
            asset = str(r["asset"])
            if sidx < 0 or eidx >= len(dates) or sidx >= eidx or asset not in close.columns:
                continue
            entry_ts = dates[sidx]
            exit_ts = dates[eidx]
            entry_px = float(close.loc[entry_ts, asset])
            exit_px = float(close.loc[exit_ts, asset])
            sign = float(r["signal_sign"])
            gross_ret = sign * (exit_px / entry_px - 1.0)
            net_ret = gross_ret - _roundtrip_cost_return(cfg)
            spy_entry = float(spy.loc[entry_ts]) if entry_ts in spy.index else float("nan")
            spy_exit = float(spy.loc[exit_ts]) if exit_ts in spy.index else float("nan")
            bench = spy_exit / spy_entry - 1.0 if np.isfinite(spy_entry) and np.isfinite(spy_exit) and spy_entry > 0 else float("nan")
            trade_rows.append(
                {
                    "fold_id": fold_id,
                    "asset": asset,
                    "signal_timestamp": str(r["signal_timestamp"]),
                    "entry_timestamp": str(entry_ts),
                    "exit_timestamp": str(exit_ts),
                    "hold_days": float(eidx - sidx),
                    "signal_sign": sign,
                    "signal_strength": float(r["signal_strength"]),
                    "gross_return": gross_ret,
                    "net_return": net_ret,
                    "benchmark_return": bench,
                    "alpha_return": net_ret - bench if np.isfinite(bench) else float("nan"),
                }
            )
    trades_out = pd.DataFrame(trade_rows)
    return out, trades_out


def _portfolio_metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {
            "n_days": 0.0,
            "n_trades": float(len(trades)),
            "annual_return": float("nan"),
            "cagr": float("nan"),
            "max_drawdown": float("nan"),
            "annualized_sharpe_net": float("nan"),
            "annualized_sharpe_alpha": float("nan"),
            "avg_turnover": float("nan"),
            "avg_gross_exposure": float("nan"),
            "mean_trade_alpha": float(pd.to_numeric(trades.get("alpha_return"), errors="coerce").mean())
            if not trades.empty
            else float("nan"),
        }
    ann = float(np.exp(np.log1p(daily["net_return"]).mean() * 252.0) - 1.0)
    start, end = daily["timestamp"].iloc[0], daily["timestamp"].iloc[-1]
    years = max(1e-9, (end - start).total_seconds() / (365.25 * 24 * 3600))
    eq_end = float(daily["equity"].iloc[-1])
    cagr = float(eq_end ** (1.0 / years) - 1.0) if eq_end > 0 else float("nan")
    max_dd = float(daily["drawdown"].min())
    sr_n = (
        float((daily["net_return"].mean() / daily["net_return"].std(ddof=1)) * math.sqrt(252.0))
        if len(daily) > 1 and float(daily["net_return"].std(ddof=1)) > 0
        else float("nan")
    )
    sr_a = (
        float((daily["alpha_return"].mean() / daily["alpha_return"].std(ddof=1)) * math.sqrt(252.0))
        if len(daily) > 1 and float(daily["alpha_return"].std(ddof=1)) > 0
        else float("nan")
    )
    return {
        "n_days": float(len(daily)),
        "n_trades": float(len(trades)),
        "annual_return": ann,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "annualized_sharpe_net": sr_n,
        "annualized_sharpe_alpha": sr_a,
        "avg_turnover": float(daily["turnover"].mean()),
        "avg_gross_exposure": float(daily["gross_exposure"].mean()),
        "mean_trade_alpha": float(pd.to_numeric(trades.get("alpha_return"), errors="coerce").mean())
        if not trades.empty
        else float("nan"),
    }


def _bootstrap_significance(x: pd.Series, iterations: int, seed: int) -> dict[str, float]:
    s = pd.to_numeric(x, errors="coerce").dropna()
    if len(s) < 10:
        return {
            "obs_mean": float("nan"),
            "bootstrap_ci_low_95": float("nan"),
            "bootstrap_ci_high_95": float("nan"),
            "bootstrap_p_mean_le_zero": float("nan"),
            "signflip_p_two_sided": float("nan"),
        }
    arr = s.to_numpy()
    n = len(arr)
    rng = np.random.default_rng(seed)
    obs = float(np.mean(arr))
    boot = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(arr[idx]))
    signflip = np.empty(iterations, dtype=float)
    for i in range(iterations):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        signflip[i] = float(np.mean(arr * signs))
    return {
        "obs_mean": obs,
        "bootstrap_ci_low_95": float(np.quantile(boot, 0.025)),
        "bootstrap_ci_high_95": float(np.quantile(boot, 0.975)),
        "bootstrap_p_mean_le_zero": float(np.mean(boot <= 0.0)),
        "signflip_p_two_sided": float(np.mean(np.abs(signflip) >= abs(obs))),
    }


def _random_entry_alpha_null(
    panel: pd.DataFrame,
    observed_trades: pd.DataFrame,
    spy_close: pd.Series,
    cfg: Config,
) -> dict[str, float]:
    t = observed_trades.copy()
    if t.empty:
        return {
            "null_mean_alpha": float("nan"),
            "null_std_alpha": float("nan"),
            "p_right": float("nan"),
            "effect_zscore": float("nan"),
        }
    rng = np.random.default_rng(cfg.random_state + 123)
    panel_by_fold_asset = {(f, a): g.sort_values("timestamp").reset_index(drop=True) for (f, a), g in panel.groupby(["fold_id", "asset"])}
    grouped = t.groupby(["fold_id", "asset"], sort=False)
    obs_mean_alpha = float(pd.to_numeric(t["alpha_return"], errors="coerce").mean())
    null_means = np.empty(cfg.random_null_iterations, dtype=float)
    rt_cost = _roundtrip_cost_return(cfg)

    for i in range(cfg.random_null_iterations):
        alpha_values: list[float] = []
        for (fold_id, asset), gt in grouped:
            df = panel_by_fold_asset.get((fold_id, asset))
            if df is None or len(df) < 5:
                continue
            holds = pd.to_numeric(gt["hold_days"], errors="coerce").dropna().astype(int).clip(lower=1)
            for h in holds:
                max_entry = len(df) - h - 1
                if max_entry < 1:
                    continue
                entry_idx = int(rng.integers(1, max_entry + 1))
                exit_idx = entry_idx + h
                entry_px = float(df.loc[entry_idx, "close"])
                exit_px = float(df.loc[exit_idx, "close"])
                gross = exit_px / entry_px - 1.0
                net = gross - rt_cost
                ets = pd.Timestamp(df.loc[entry_idx, "timestamp"])
                xts = pd.Timestamp(df.loc[exit_idx, "timestamp"])
                spy_entry = float(spy_close.reindex([ets], method="ffill").iloc[0]) if len(spy_close) else float("nan")
                spy_exit = float(spy_close.reindex([xts], method="ffill").iloc[0]) if len(spy_close) else float("nan")
                if not np.isfinite(spy_entry) or not np.isfinite(spy_exit) or spy_entry <= 0:
                    continue
                bench = spy_exit / spy_entry - 1.0
                alpha_values.append(float(net - bench))
        null_means[i] = float(np.mean(alpha_values)) if alpha_values else float("nan")

    null = pd.Series(null_means, dtype=float).dropna()
    if null.empty:
        return {
            "null_mean_alpha": float("nan"),
            "null_std_alpha": float("nan"),
            "p_right": float("nan"),
            "effect_zscore": float("nan"),
        }
    return {
        "null_mean_alpha": float(null.mean()),
        "null_std_alpha": float(null.std(ddof=1)) if len(null) > 1 else float("nan"),
        "p_right": float(np.mean(null >= obs_mean_alpha)),
        "effect_zscore": float((obs_mean_alpha - null.mean()) / null.std(ddof=1))
        if len(null) > 1 and float(null.std(ddof=1)) > 0
        else float("nan"),
    }


def main() -> None:
    out_root = Path("trading_research/examples/_output/29_baseline_rel2_rigorous_validation")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(list(UNIVERSE_170), period=cfg.period, interval=cfg.interval)
    spy_df = vendor.fetch_bars(["SPY"], period=cfg.period, interval=cfg.interval)[["timestamp", "close"]]
    if bars.empty or spy_df.empty:
        raise ValueError("Missing Yahoo data for requested universe or SPY benchmark.")

    panel = _build_panel(bars, spy_df, cfg)
    years = sorted(panel["year"].dropna().unique().tolist())
    if len(years) <= cfg.min_train_years:
        raise ValueError("Insufficient years for anchored walk-forward with requested min_train_years.")

    all_daily: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    fold_rows: list[dict[str, float | str]] = []

    for test_year in years[cfg.min_train_years :]:
        fold_id = f"fold_{int(test_year)}"
        train = panel[panel["year"] < test_year].copy()
        test = panel[panel["year"] == test_year].copy()
        if train.empty or test.empty:
            continue
        trades = _build_trades(test, cfg, fold_id=fold_id)
        daily, trades_eval = _simulate_fold(test, trades, spy_df=spy_df, cfg=cfg, fold_id=fold_id)
        if daily.empty:
            continue
        fold_metrics = _portfolio_metrics(daily, trades_eval)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "test_year": int(test_year),
                "train_years": int(train["year"].nunique()),
                "assets_in_test": int(test["asset"].nunique()),
                **fold_metrics,
            }
        )
        test_with_fold = test.copy()
        test_with_fold["fold_id"] = fold_id
        all_daily.append(daily)
        all_trades.append(trades_eval)

    daily_all = pd.concat([d for d in all_daily if not d.empty], ignore_index=True) if all_daily else pd.DataFrame()
    trades_all = pd.concat([t for t in all_trades if not t.empty], ignore_index=True) if all_trades else pd.DataFrame()
    fold_df = pd.DataFrame(fold_rows).sort_values("test_year").reset_index(drop=True) if fold_rows else pd.DataFrame()

    if not daily_all.empty:
        daily_all = daily_all.sort_values("timestamp").reset_index(drop=True)
        daily_all["year"] = daily_all["timestamp"].dt.year
    yearly_df = (
        daily_all.groupby("year", as_index=False)
        .agg(
            n_days=("net_return", "count"),
            mean_net_return=("net_return", "mean"),
            mean_alpha_return=("alpha_return", "mean"),
            annualized_sharpe_net=("net_return", lambda s: float((s.mean() / s.std(ddof=1)) * math.sqrt(252.0)) if len(s) > 1 and float(s.std(ddof=1)) > 0 else float("nan")),
            annualized_sharpe_alpha=("alpha_return", lambda s: float((s.mean() / s.std(ddof=1)) * math.sqrt(252.0)) if len(s) > 1 and float(s.std(ddof=1)) > 0 else float("nan")),
        )
        .sort_values("year")
        .reset_index(drop=True)
        if not daily_all.empty
        else pd.DataFrame()
    )

    overall = _portfolio_metrics(daily_all, trades_all)
    sig_daily_alpha = _bootstrap_significance(
        daily_all["alpha_return"] if not daily_all.empty else pd.Series(dtype=float),
        iterations=cfg.bootstrap_iterations,
        seed=cfg.random_state,
    )
    sig_trade_alpha = _bootstrap_significance(
        trades_all["alpha_return"] if not trades_all.empty else pd.Series(dtype=float),
        iterations=cfg.bootstrap_iterations,
        seed=cfg.random_state + 11,
    )
    spy_close = (
        spy_df.assign(timestamp=lambda d: pd.to_datetime(d["timestamp"], utc=True, errors="coerce"))
        .dropna(subset=["timestamp", "close"])
        .sort_values("timestamp")
        .set_index("timestamp")["close"]
        .astype(float)
    )
    panel_for_null = panel.copy()
    if not fold_df.empty:
        panel_fold_parts: list[pd.DataFrame] = []
        for _, r in fold_df.iterrows():
            y = int(r["test_year"])
            fid = str(r["fold_id"])
            g = panel_for_null[panel_for_null["year"] == y].copy()
            g["fold_id"] = fid
            panel_fold_parts.append(g)
        panel_for_null = pd.concat(panel_fold_parts, ignore_index=True) if panel_fold_parts else pd.DataFrame()
    null_test = _random_entry_alpha_null(panel_for_null, trades_all, spy_close=spy_close, cfg=cfg)

    scorecard = {
        "daily_alpha_mean_positive": bool(np.isfinite(sig_daily_alpha["obs_mean"]) and sig_daily_alpha["obs_mean"] > 0),
        "daily_alpha_bootstrap_p_lt_0_05": bool(
            np.isfinite(sig_daily_alpha["bootstrap_p_mean_le_zero"]) and sig_daily_alpha["bootstrap_p_mean_le_zero"] < 0.05
        ),
        "trade_alpha_bootstrap_p_lt_0_05": bool(
            np.isfinite(sig_trade_alpha["bootstrap_p_mean_le_zero"]) and sig_trade_alpha["bootstrap_p_mean_le_zero"] < 0.05
        ),
        "random_null_p_right_lt_0_05": bool(np.isfinite(null_test["p_right"]) and null_test["p_right"] < 0.05),
        "oos_years_positive_alpha_ratio": float((yearly_df["mean_alpha_return"] > 0).mean()) if not yearly_df.empty else float("nan"),
    }

    summary = {
        "strategy": "baseline_rel2",
        "universe_size_requested": len(UNIVERSE_170),
        "universe_size_fetched": int(panel["asset"].nunique()) if not panel.empty else 0,
        "history_start": str(panel["timestamp"].min()) if not panel.empty else None,
        "history_end": str(panel["timestamp"].max()) if not panel.empty else None,
        "n_walkforward_folds": int(len(fold_df)),
        "config": {
            "period": cfg.period,
            "interval": cfg.interval,
            "signal_lookback_days": cfg.signal_lookback_days,
            "rolling_vwap_window": cfg.rolling_vwap_window,
            "rel_strength_threshold": cfg.rel_strength_threshold,
            "top_quantile": cfg.top_quantile,
            "hold_days": cfg.hold_days,
            "gross_target": cfg.gross_target,
            "max_abs_weight_per_asset": cfg.max_abs_weight_per_asset,
            "one_way_cost_return": _one_way_cost_return(cfg),
            "roundtrip_cost_return": _roundtrip_cost_return(cfg),
            "min_train_years": cfg.min_train_years,
        },
        "overall_oos_portfolio": overall,
        "significance_daily_alpha": sig_daily_alpha,
        "significance_trade_alpha": sig_trade_alpha,
        "beta1_random_entry_null": null_test,
        "scorecard": scorecard,
    }

    fold_df.to_csv(reports_dir / "walkforward_fold_metrics.csv", index=False)
    yearly_df.to_csv(reports_dir / "walkforward_yearly_metrics.csv", index=False)
    daily_all.to_csv(reports_dir / "daily_oos_all_folds.csv", index=False)
    trades_all.to_csv(reports_dir / "trades_oos_all_folds.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (reports_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("folds:", len(fold_df), "years:", sorted(fold_df["test_year"].tolist()) if not fold_df.empty else [])
    print("overall:", json.dumps(overall, indent=2))
    print("daily alpha significance:", json.dumps(sig_daily_alpha, indent=2))
    print("trade alpha significance:", json.dumps(sig_trade_alpha, indent=2))
    print("random-entry null:", json.dumps(null_test, indent=2))
    print("scorecard:", json.dumps(scorecard, indent=2))


if __name__ == "__main__":
    main()
