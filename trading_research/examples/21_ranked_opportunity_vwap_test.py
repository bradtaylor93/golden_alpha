"""Ranked-opportunity VWAP test with strict market-neutral validation.

This experiment tests the user's ranking idea directly:
- Build daily cross-sectional ranks of a VWAP-derived signal.
- Trade ranked opportunities rather than hard absolute thresholds.

Signals tested:
1) top_decile_momentum   (long in top score decile, momentum direction)
2) bottom_decile_momentum (long/short via momentum sign in bottom decile)
3) top_decile_contrarian
4) bottom_decile_contrarian

For each signal:
- OOS mean net return, alpha vs SPY matched horizon, win rate, t-stats.
- Beta-matched random-entry null test on alpha.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data.vendors.yahoo import YahooMarketDataVendor

UNIVERSE_90 = [
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
]


@dataclass(frozen=True)
class RankedConfig:
    test_ratio: float = 0.30
    signal_lookback_days: int = 63
    vwap_window_days: int = 20
    hold_days: int = 63
    top_quantile: float = 0.90
    bottom_quantile: float = 0.10
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    null_iterations: int = 1000
    random_state: int = 42


def _roundtrip_cost_return(cfg: RankedConfig) -> float:
    return 2.0 * (cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side) / 10_000.0


def _t_stat(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < 2:
        return float("nan")
    std = float(x.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(x.mean() / std * math.sqrt(len(x)))


def _split_time(panel: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = sorted(panel["timestamp"].dropna().unique())
    split_idx = int((1.0 - test_ratio) * len(dates))
    split_idx = min(max(split_idx, 1), len(dates) - 1)
    split_ts = pd.Timestamp(dates[split_idx])
    train = panel[panel["timestamp"] < split_ts].copy()
    test = panel[panel["timestamp"] >= split_ts].copy()
    return train, test, split_ts


def _ensure_spy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "spy_close" not in out.columns:
        if "spy_close_x" in out.columns:
            out["spy_close"] = pd.to_numeric(out["spy_close_x"], errors="coerce")
        elif "spy_close_y" in out.columns:
            out["spy_close"] = pd.to_numeric(out["spy_close_y"], errors="coerce")
    if "spy_exit_close" not in out.columns:
        if "spy_exit_close_x" in out.columns:
            out["spy_exit_close"] = pd.to_numeric(out["spy_exit_close_x"], errors="coerce")
        elif "spy_exit_close_y" in out.columns:
            out["spy_exit_close"] = pd.to_numeric(out["spy_exit_close_y"], errors="coerce")
    return out


def _build_panel(bars: pd.DataFrame, spy_close: pd.DataFrame, cfg: RankedConfig) -> pd.DataFrame:
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

    vwap_w = cfg.vwap_window_days
    frame["rolling_vwap"] = (
        pv.groupby(frame["asset"], sort=False).rolling(vwap_w, min_periods=vwap_w).sum().reset_index(level=0, drop=True)
        / frame["volume"]
        .fillna(0.0)
        .groupby(frame["asset"], sort=False)
        .rolling(vwap_w, min_periods=vwap_w)
        .sum()
        .reset_index(level=0, drop=True)
        .replace(0.0, np.nan)
    )
    look = cfg.signal_lookback_days
    frame["past_return"] = g["close"].pct_change(look)
    frame["score"] = np.sign(frame["past_return"]) * (frame["close"] / frame["rolling_vwap"] - 1.0)

    frame = frame.merge(spy_close, on="timestamp", how="left")
    frame = frame.dropna(subset=["score", "past_return", "spy_close"]).copy()
    frame["rank_pct"] = frame.groupby("timestamp")["score"].rank(pct=True, method="average")

    hold = cfg.hold_days
    frame["exit_close"] = g["close"].shift(-hold)
    frame["exit_timestamp"] = g["timestamp"].shift(-hold)

    spy = spy_close.sort_values("timestamp").copy()
    spy["spy_exit_close"] = spy["spy_close"].shift(-hold)
    frame = frame.merge(
        spy.rename(columns={"timestamp": "entry_timestamp"})[["entry_timestamp", "spy_close", "spy_exit_close"]],
        left_on="timestamp",
        right_on="entry_timestamp",
        how="left",
    )
    frame = frame.drop(columns=["entry_timestamp"])
    return frame.reset_index(drop=True)


def _signal_position(panel: pd.DataFrame, signal_name: str, cfg: RankedConfig) -> pd.Series:
    if signal_name == "top_decile_momentum":
        mask = panel["rank_pct"] >= cfg.top_quantile
        pos = np.sign(panel["past_return"])
    elif signal_name == "bottom_decile_momentum":
        mask = panel["rank_pct"] <= cfg.bottom_quantile
        pos = np.sign(panel["past_return"])
    elif signal_name == "top_decile_contrarian":
        mask = panel["rank_pct"] >= cfg.top_quantile
        pos = -np.sign(panel["past_return"])
    elif signal_name == "bottom_decile_contrarian":
        mask = panel["rank_pct"] <= cfg.bottom_quantile
        pos = -np.sign(panel["past_return"])
    else:
        raise ValueError(f"Unknown signal: {signal_name}")
    out = pd.Series(np.where(mask, pos, np.nan), index=panel.index, dtype=float)
    return out.replace(0.0, np.nan)


def _evaluate_signal(test_panel: pd.DataFrame, signal_name: str, cfg: RankedConfig) -> tuple[pd.DataFrame, dict[str, float | str]]:
    t = _ensure_spy_columns(test_panel.copy())
    t["position"] = _signal_position(t, signal_name, cfg)
    # If merges introduced suffixed SPY columns, normalize back to canonical names.
    if "spy_close" not in t.columns and "spy_close_x" in t.columns:
        t["spy_close"] = pd.to_numeric(t["spy_close_x"], errors="coerce")
    if "spy_exit_close" not in t.columns and "spy_exit_close_x" in t.columns:
        t["spy_exit_close"] = pd.to_numeric(t["spy_exit_close_x"], errors="coerce")
    t = t.dropna(subset=["position", "close", "exit_close", "spy_close", "spy_exit_close"]).copy()
    rt_cost = _roundtrip_cost_return(cfg)
    t["net_return"] = t["position"] * (t["exit_close"] / t["close"] - 1.0) - rt_cost
    t["spy_net"] = t["spy_exit_close"] / t["spy_close"] - 1.0 - rt_cost
    t["alpha_return"] = t["net_return"] - t["spy_net"]
    t["hold_days"] = cfg.hold_days
    t["daily_log_net"] = np.log1p(t["net_return"]) / cfg.hold_days
    t["daily_log_alpha"] = np.log1p(t["alpha_return"].clip(lower=-0.999999)) / cfg.hold_days

    r = t["net_return"]
    a = t["alpha_return"]
    summary = {
        "signal": signal_name,
        "n_trades": float(len(t)),
        "mean_net_return": float(r.mean()) if len(r) else float("nan"),
        "win_rate_net": float((r > 0).mean()) if len(r) else float("nan"),
        "t_stat_net": _t_stat(r),
        "mean_alpha_return": float(a.mean()) if len(a) else float("nan"),
        "win_rate_alpha": float((a > 0).mean()) if len(a) else float("nan"),
        "t_stat_alpha": _t_stat(a),
        "sharpe_trade_alpha": float(a.mean() / a.std(ddof=1)) if len(a) > 1 and float(a.std(ddof=1)) > 0 else float("nan"),
        "annualized_sharpe_daily_alpha": float((t["daily_log_alpha"].mean() / t["daily_log_alpha"].std(ddof=1)) * math.sqrt(252.0))
        if len(t["daily_log_alpha"]) > 1 and float(t["daily_log_alpha"].std(ddof=1)) > 0
        else float("nan"),
    }
    return t, summary


def _market_neutral_random_null(
    test_panel: pd.DataFrame,
    observed_mean_alpha: float,
    n_trades_target: int,
    cfg: RankedConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_state + 501)
    rt_cost = _roundtrip_cost_return(cfg)
    hold = cfg.hold_days

    pool = _ensure_spy_columns(test_panel.copy())
    pool = pool.dropna(subset=["close", "exit_close", "spy_close", "spy_exit_close"]).copy()
    pool = pool.reset_index(drop=True)
    if pool.empty or n_trades_target <= 0:
        out = {
            "null_mean_alpha": float("nan"),
            "null_std_alpha": float("nan"),
            "p_right": float("nan"),
            "p_left": float("nan"),
            "effect_zscore": float("nan"),
        }
        return out, pd.DataFrame()

    null_means = np.empty(cfg.null_iterations, dtype=float)
    for i in range(cfg.null_iterations):
        take = min(n_trades_target, len(pool))
        sample = pool.sample(n=take, replace=False, random_state=int(rng.integers(0, 1_000_000_000)))
        rand_pos = rng.choice([-1.0, 1.0], size=len(sample), replace=True)
        net = rand_pos * (sample["exit_close"].to_numpy() / sample["close"].to_numpy() - 1.0) - rt_cost
        spy_net = sample["spy_exit_close"].to_numpy() / sample["spy_close"].to_numpy() - 1.0 - rt_cost
        alpha = net - spy_net
        null_means[i] = float(np.mean(alpha)) if len(alpha) else float("nan")

    null = pd.Series(null_means).dropna()
    p_right = float((null >= observed_mean_alpha).mean()) if len(null) else float("nan")
    p_left = float((null <= observed_mean_alpha).mean()) if len(null) else float("nan")
    out = {
        "null_mean_alpha": float(null.mean()) if len(null) else float("nan"),
        "null_std_alpha": float(null.std(ddof=1)) if len(null) > 1 else float("nan"),
        "p_right": p_right,
        "p_left": p_left,
        "effect_zscore": float((observed_mean_alpha - null.mean()) / null.std(ddof=1))
        if len(null) > 1 and float(null.std(ddof=1)) > 0
        else float("nan"),
    }
    return out, pd.DataFrame({"null_mean_alpha": null.to_numpy()})


def main() -> None:
    out_root = Path("trading_research/examples/_output/21_ranked_opportunity_vwap_test")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = RankedConfig()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(UNIVERSE_90, period="3y", interval="1d")
    spy = vendor.fetch_bars(["SPY"], period="3y", interval="1d")[["timestamp", "close"]].rename(columns={"close": "spy_close"})
    if bars.empty or spy.empty:
        raise ValueError("Missing market data for ranked-opportunity test.")
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True, errors="coerce")
    spy = spy.dropna(subset=["timestamp", "spy_close"]).sort_values("timestamp")

    panel = _build_panel(bars, spy, cfg)
    _, test, split_ts = _split_time(panel, cfg.test_ratio)

    signal_names = [
        "top_decile_momentum",
        "bottom_decile_momentum",
        "top_decile_contrarian",
        "bottom_decile_contrarian",
    ]
    summaries: list[dict[str, float | str]] = []
    null_rows: list[pd.DataFrame] = []
    trade_tables: dict[str, pd.DataFrame] = {}
    for s in signal_names:
        trades, summary = _evaluate_signal(test, s, cfg)
        trade_tables[s] = trades
        null_stats, null_df = _market_neutral_random_null(
            test,
            observed_mean_alpha=float(summary["mean_alpha_return"]),
            n_trades_target=int(summary["n_trades"]),
            cfg=cfg,
        )
        summary = {**summary, **{f"null_{k}": v for k, v in null_stats.items()}}
        summaries.append(summary)
        if not null_df.empty:
            null_df = null_df.rename(columns={"null_mean_alpha": f"null_mean_alpha_{s}"})
            null_rows.append(null_df.reset_index(drop=True))

    summary_df = pd.DataFrame(summaries).sort_values("mean_alpha_return", ascending=False).reset_index(drop=True)
    best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}

    summary = {
        "universe_size_requested": len(UNIVERSE_90),
        "universe_size_fetched": int(pd.Series(bars["asset"]).nunique()),
        "split_timestamp": str(split_ts),
        "test_rows": int(len(test)),
        "hold_days": cfg.hold_days,
        "best_signal_by_alpha": best.get("signal", "n/a"),
        "best_mean_alpha_return": float(best.get("mean_alpha_return", float("nan"))),
        "best_signal_null_p_right": float(best.get("null_p_right", float("nan"))),
    }

    for name, df in trade_tables.items():
        df.to_csv(reports_dir / f"trades_{name}.csv", index=False)
    summary_df.to_csv(reports_dir / "ranked_signal_comparison.csv", index=False)
    if null_rows:
        null_wide = pd.concat(null_rows, axis=1)
        null_wide.to_csv(reports_dir / "null_market_neutral_random_ranked.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("split_timestamp:", summary["split_timestamp"], "test_rows:", summary["test_rows"])
    print(summary_df.to_string(index=False))
    print("\nsummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
