"""Portfolio-level backtest for ranked VWAP top-decile momentum.

This script converts the ranked-opportunity signal into a realistic portfolio
simulation with:
- cross-sectional ranking and top-decile signal selection,
- fixed holding period and non-overlapping entries per asset,
- gross exposure targeting and per-asset weight cap,
- turnover-based transaction+slippage costs,
- daily equity curve, CAGR, annual return table, and max drawdown.
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
class BacktestConfig:
    signal_lookback_days: int = 63
    rolling_vwap_window: int = 20
    hold_days: int = 63
    top_quantile: float = 0.90
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


def _roundtrip_cost_return(cfg: BacktestConfig) -> float:
    bps_side = cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side
    return 2.0 * bps_side / 10_000.0


def _one_way_cost_return(cfg: BacktestConfig) -> float:
    bps_side = cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side
    return bps_side / 10_000.0


def _build_panel(bars: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
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


def _build_non_overlapping_trades(test_panel: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    test_panel = test_panel.sort_values(["asset", "timestamp"]).copy()
    all_dates = sorted(test_panel["timestamp"].unique())
    date_to_idx = {ts: i for i, ts in enumerate(all_dates)}
    n_dates = len(all_dates)

    for asset, g in test_panel.groupby("asset", sort=False):
        g = g.sort_values("timestamp").copy()
        # top-decile momentum signal
        g["entry_flag"] = (g["rank_pct"] >= cfg.top_quantile) & (np.sign(g["past_return"]) != 0)
        g["signal_side"] = np.sign(g["past_return"]).replace(0.0, np.nan)
        last_end = -1
        for _, r in g[g["entry_flag"]].iterrows():
            sig_idx = date_to_idx.get(pd.Timestamp(r["timestamp"]))
            if sig_idx is None:
                continue
            if sig_idx <= last_end:
                continue
            start_idx = sig_idx + 1
            end_idx = min(sig_idx + cfg.hold_days, n_dates - 1)
            if start_idx >= n_dates or end_idx <= start_idx:
                continue
            sign = float(r["signal_side"])
            if not np.isfinite(sign) or sign == 0.0:
                continue
            strength = float(max(1e-6, r["rank_pct"] - cfg.top_quantile))
            rows.append(
                {
                    "asset": str(asset),
                    "signal_timestamp": str(r["timestamp"]),
                    "start_idx": float(start_idx),
                    "end_idx": float(end_idx),
                    "signal_sign": sign,
                    "signal_strength": strength,
                }
            )
            last_end = end_idx

    return pd.DataFrame(rows)


def _weights_from_active(active: pd.DataFrame, cfg: BacktestConfig, asset_cols: list[str]) -> pd.Series:
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
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    close = (
        test_panel.pivot(index="timestamp", columns="asset", values="close")
        .sort_index()
        .ffill()
    )
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
                "gross_return": 0.0,
                "turnover": 0.0,
                "cost_return": 0.0,
                "net_return": 0.0,
                "spy_return": spy_ret.values,
                "alpha_return": -spy_ret.values,
                "gross_exposure": 0.0,
            }
        )
        summary = {
            "n_days": float(len(out)),
            "n_trades": 0.0,
            "annual_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
        }
        return out, summary

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
                "gross_return": gross,
                "turnover": turnover,
                "cost_return": cost,
                "net_return": net,
                "spy_return": spy_r,
                "alpha_return": net - spy_r,
                "gross_exposure": float(np.abs(w).sum()),
            }
        )
        prev_w = w

    out = pd.DataFrame(rows)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    out["spy_equity"] = (1.0 + out["spy_return"]).cumprod()
    out["alpha_equity"] = (1.0 + out["alpha_return"]).cumprod()
    out["running_max"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["running_max"] - 1.0

    if out.empty:
        ann_return = cagr = max_dd = float("nan")
    else:
        ann_return = float(np.exp(np.log1p(out["net_return"]).mean() * 252.0) - 1.0)
        start_ts = out["timestamp"].iloc[0]
        end_ts = out["timestamp"].iloc[-1]
        years = max(1e-9, (end_ts - start_ts).total_seconds() / (365.25 * 24 * 3600))
        final_equity = float(out["equity"].iloc[-1])
        cagr = float(final_equity ** (1.0 / years) - 1.0) if final_equity > 0 else float("nan")
        max_dd = float(out["drawdown"].min())

    summary = {
        "n_days": float(len(out)),
        "n_trades": float(len(trades)),
        "avg_gross_exposure": float(out["gross_exposure"].mean()) if not out.empty else float("nan"),
        "avg_turnover": float(out["turnover"].mean()) if not out.empty else float("nan"),
        "annual_return": ann_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "final_equity": float(out["equity"].iloc[-1]) if not out.empty else float("nan"),
        "final_spy_equity": float(out["spy_equity"].iloc[-1]) if not out.empty else float("nan"),
        "final_alpha_equity": float(out["alpha_equity"].iloc[-1]) if not out.empty else float("nan"),
    }
    return out, summary


def main() -> None:
    out_root = Path("trading_research/examples/_output/22_portfolio_backtest_ranked_vwap")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = BacktestConfig()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(list(UNIVERSE_90), period="3y", interval="1d")
    spy = vendor.fetch_bars(["SPY"], period="3y", interval="1d")[["timestamp", "close"]]
    if bars.empty or spy.empty:
        raise ValueError("Missing data for portfolio backtest.")

    panel = _build_panel(bars, cfg)
    _, test, split_ts = _split_time(panel, cfg.test_ratio)
    trades = _build_non_overlapping_trades(test, cfg)
    daily, summary = _simulate_portfolio(test, trades, spy, cfg)

    if not daily.empty:
        yearly = (
            daily.assign(year=daily["timestamp"].dt.year)
            .groupby("year", as_index=False)["net_return"]
            .apply(lambda s: float(np.prod(1.0 + s) - 1.0))
            .rename(columns={"net_return": "year_return"})
        )
    else:
        yearly = pd.DataFrame(columns=["year", "year_return"])

    metadata = {
        "universe_size_requested": len(UNIVERSE_90),
        "universe_size_fetched": int(pd.Series(bars["asset"]).nunique()),
        "split_timestamp": str(split_ts),
        "config": {
            "signal_lookback_days": cfg.signal_lookback_days,
            "rolling_vwap_window": cfg.rolling_vwap_window,
            "hold_days": cfg.hold_days,
            "top_quantile": cfg.top_quantile,
            "gross_target": cfg.gross_target,
            "max_abs_weight_per_asset": cfg.max_abs_weight_per_asset,
            "roundtrip_cost_return": _roundtrip_cost_return(cfg),
        },
        "performance": summary,
    }

    trades.to_csv(reports_dir / "trades_non_overlapping.csv", index=False)
    daily.to_csv(reports_dir / "portfolio_daily.csv", index=False)
    yearly.to_csv(reports_dir / "yearly_returns.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("split_timestamp:", metadata["split_timestamp"])
    print("annual_return:", summary["annual_return"])
    print("cagr:", summary["cagr"])
    print("max_drawdown:", summary["max_drawdown"])
    if not yearly.empty:
        print("yearly_returns:")
        print(yearly.to_string(index=False))


if __name__ == "__main__":
    main()
