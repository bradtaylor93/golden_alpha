"""Evaluate long-short extension of ranked VWAP opportunity strategy.

Compares:
1) Baseline: top-decile ranked momentum signal (existing approach).
2) Extension: long top-decile AND short inverse bottom-decile signal.

Both are run with identical portfolio simulation settings so incremental
alpha impact is attributable to the signal extension, not implementation
differences.
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
class Config:
    signal_lookback_days: int = 63
    rolling_vwap_window: int = 20
    hold_days: int = 63
    top_quantile: float = 0.90
    bottom_quantile: float = 0.10
    test_ratio: float = 0.30
    gross_target: float = 1.0
    max_abs_weight_per_asset: float = 0.05
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0


def _split_time(panel: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = sorted(panel["timestamp"].dropna().unique())
    split_idx = int((1.0 - test_ratio) * len(dates))
    split_idx = min(max(split_idx, 1), len(dates) - 1)
    split_ts = pd.Timestamp(dates[split_idx])
    train = panel[panel["timestamp"] < split_ts].copy()
    test = panel[panel["timestamp"] >= split_ts].copy()
    return train, test, split_ts


def _one_way_cost_return(cfg: Config) -> float:
    bps = cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side
    return bps / 10_000.0


def _build_panel(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
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
    frame = frame.dropna(subset=["close", "score", "past_return", "rank_pct"]).reset_index(drop=True)
    return frame


def _signal_sign_and_strength(row: pd.Series, strategy: str, cfg: Config) -> tuple[float, float] | None:
    rank = float(row["rank_pct"])
    past_ret = float(row["past_return"])
    trend_sign = float(np.sign(past_ret))
    if not np.isfinite(trend_sign) or trend_sign == 0.0:
        return None

    if strategy == "baseline_top_decile_momentum":
        if rank < cfg.top_quantile:
            return None
        sign = trend_sign
        strength = float(max(1e-6, rank - cfg.top_quantile))
        return sign, strength

    if strategy == "long_short_rank_inverse":
        if rank >= cfg.top_quantile:
            # Keep same long-side as baseline at top ranks.
            sign = trend_sign
            strength = float(max(1e-6, rank - cfg.top_quantile))
            return sign, strength
        if rank <= cfg.bottom_quantile:
            # Inverse side: explicit short at opposite tail.
            sign = -trend_sign
            strength = float(max(1e-6, cfg.bottom_quantile - rank))
            return sign, strength
        return None

    raise ValueError(f"Unknown strategy: {strategy}")


def _build_trades(test_panel: pd.DataFrame, strategy: str, cfg: Config) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    test_panel = test_panel.sort_values(["asset", "timestamp"]).copy()
    all_dates = sorted(test_panel["timestamp"].unique())
    date_to_idx = {ts: i for i, ts in enumerate(all_dates)}
    n_dates = len(all_dates)

    for asset, g in test_panel.groupby("asset", sort=False):
        g = g.sort_values("timestamp").copy()
        last_end = -1
        for _, r in g.iterrows():
            out = _signal_sign_and_strength(r, strategy, cfg)
            if out is None:
                continue
            sign, strength = out
            sig_idx = date_to_idx.get(pd.Timestamp(r["timestamp"]))
            if sig_idx is None or sig_idx <= last_end:
                continue
            start_idx = sig_idx + 1
            end_idx = min(sig_idx + cfg.hold_days, n_dates - 1)
            if start_idx >= n_dates or end_idx <= start_idx:
                continue
            rows.append(
                {
                    "strategy": strategy,
                    "asset": str(asset),
                    "signal_timestamp": str(r["timestamp"]),
                    "start_idx": float(start_idx),
                    "end_idx": float(end_idx),
                    "signal_sign": float(sign),
                    "signal_strength": float(strength),
                    "hold_days": float(end_idx - sig_idx),
                }
            )
            last_end = end_idx
    return pd.DataFrame(rows)


def _weights_from_active(active: pd.DataFrame, cfg: Config, asset_cols: list[str]) -> pd.Series:
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


def _simulate_portfolio(
    test_panel: pd.DataFrame,
    trades: pd.DataFrame,
    spy_df: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, float]]:
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
        out = pd.DataFrame(
            {
                "timestamp": rets.index,
                "net_return": 0.0,
                "spy_return": spy_ret.values,
                "alpha_return": -spy_ret.values,
                "turnover": 0.0,
                "gross_exposure": 0.0,
            }
        )
        out["equity"] = 1.0
        out["running_max"] = 1.0
        out["drawdown"] = 0.0
        return out, {
            "n_days": float(len(out)),
            "n_trades": 0.0,
            "annual_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "annualized_sharpe_net": float("nan"),
            "annualized_sharpe_alpha": float("nan"),
            "mean_hold_days": float("nan"),
            "avg_gross_exposure": 0.0,
            "avg_turnover": 0.0,
        }

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
        sharpe_n = (
            float((out["net_return"].mean() / out["net_return"].std(ddof=1)) * math.sqrt(252.0))
            if len(out) > 1 and float(out["net_return"].std(ddof=1)) > 0
            else float("nan")
        )
        sharpe_a = (
            float((out["alpha_return"].mean() / out["alpha_return"].std(ddof=1)) * math.sqrt(252.0))
            if len(out) > 1 and float(out["alpha_return"].std(ddof=1)) > 0
            else float("nan")
        )

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
    out_root = Path("trading_research/examples/_output/25_ranked_vwap_long_short_extension")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(list(UNIVERSE_90), period="3y", interval="1d")
    spy = vendor.fetch_bars(["SPY"], period="3y", interval="1d")[["timestamp", "close"]]
    if bars.empty or spy.empty:
        raise ValueError("Missing data for long-short extension test.")

    panel = _build_panel(bars, cfg)
    _, test, split_ts = _split_time(panel, cfg.test_ratio)

    strategies = ["baseline_top_decile_momentum", "long_short_rank_inverse"]
    rows: list[dict[str, float | str]] = []
    for s in strategies:
        trades = _build_trades(test, s, cfg)
        daily, perf = _simulate_portfolio(test, trades, spy, cfg)
        rows.append({"strategy": s, **perf})
        trades.to_csv(reports_dir / f"trades_{s}.csv", index=False)
        daily.to_csv(reports_dir / f"portfolio_daily_{s}.csv", index=False)

    summary_df = pd.DataFrame(rows).sort_values("annualized_sharpe_alpha", ascending=False).reset_index(drop=True)
    baseline = summary_df[summary_df["strategy"] == "baseline_top_decile_momentum"]
    extended = summary_df[summary_df["strategy"] == "long_short_rank_inverse"]
    if not baseline.empty and not extended.empty:
        b = baseline.iloc[0]
        e = extended.iloc[0]
        uplift = {
            "delta_cagr_long_short_minus_baseline": float(e["cagr"] - b["cagr"]),
            "delta_alpha_sharpe_long_short_minus_baseline": float(
                e["annualized_sharpe_alpha"] - b["annualized_sharpe_alpha"]
            ),
            "delta_max_dd_long_short_minus_baseline": float(e["max_drawdown"] - b["max_drawdown"]),
            "delta_net_sharpe_long_short_minus_baseline": float(
                e["annualized_sharpe_net"] - b["annualized_sharpe_net"]
            ),
        }
    else:
        uplift = {}

    summary = {
        "split_timestamp": str(split_ts),
        "universe_size_requested": len(UNIVERSE_90),
        "universe_size_fetched": int(pd.Series(bars["asset"]).nunique()),
        "config": {
            "signal_lookback_days": cfg.signal_lookback_days,
            "rolling_vwap_window": cfg.rolling_vwap_window,
            "hold_days": cfg.hold_days,
            "top_quantile": cfg.top_quantile,
            "bottom_quantile": cfg.bottom_quantile,
            "gross_target": cfg.gross_target,
            "max_abs_weight_per_asset": cfg.max_abs_weight_per_asset,
            "one_way_cost_return": _one_way_cost_return(cfg),
        },
        "uplift": uplift,
    }

    summary_df.to_csv(reports_dir / "strategy_comparison.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("split_timestamp:", summary["split_timestamp"])
    print(summary_df.to_string(index=False))
    print("\nuplift:", json.dumps(uplift, indent=2))


if __name__ == "__main__":
    main()

