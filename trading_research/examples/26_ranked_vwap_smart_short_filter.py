"""Smart-short extension test for ranked VWAP strategy.

Objective:
- Keep ranked VWAP momentum entry structure,
- Add a more selective short sleeve (bottom-tail + weak relative strength
  + bearish market confirmation),
- Evaluate whether the short sleeve adds alpha without heavy parameter search.
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

UNIVERSE_130 = tuple(dict.fromkeys([*UNIVERSE_50, *EXTRA_UNIVERSE_40, *EXTRA_UNIVERSE_40_MORE]))


@dataclass(frozen=True)
class Config:
    signal_lookback_days: int = 63
    rolling_vwap_window: int = 20
    top_quantile: float = 0.90
    bottom_quantile: float = 0.10
    test_ratio: float = 0.30
    gross_target: float = 1.0
    max_abs_weight_per_asset: float = 0.05
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    hold_days: int = 63
    spy_ma_window_days: int = 200
    short_rel_strength_thresholds: tuple[float, ...] = (-0.01, -0.03)


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


def _build_panel(bars: pd.DataFrame, spy_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
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

    spy = spy_df.copy()
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True, errors="coerce")
    spy["spy_close"] = pd.to_numeric(spy["close"], errors="coerce")
    spy = spy.dropna(subset=["timestamp", "spy_close"]).sort_values("timestamp")
    spy["spy_past_return"] = spy["spy_close"].pct_change(look)
    spy["spy_ma"] = spy["spy_close"].rolling(cfg.spy_ma_window_days, min_periods=cfg.spy_ma_window_days).mean()
    spy["bear_confirm"] = ((spy["spy_close"] < spy["spy_ma"]) & (spy["spy_past_return"] < 0.0)).astype(int)

    frame = frame.merge(
        spy[["timestamp", "spy_close", "spy_past_return", "bear_confirm"]],
        on="timestamp",
        how="left",
    )
    frame["rel_strength"] = frame["past_return"] - frame["spy_past_return"]
    frame = frame.dropna(
        subset=["close", "score", "past_return", "rank_pct", "spy_close", "spy_past_return", "rel_strength"]
    ).reset_index(drop=True)
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
        return trend_sign, float(max(1e-6, rank - cfg.top_quantile))

    if strategy.startswith("smart_short_rel_"):
        # Keep baseline long sleeve.
        if rank >= cfg.top_quantile:
            return trend_sign, float(max(1e-6, rank - cfg.top_quantile))
        # Add selective short sleeve only on weak names in bearish regime.
        threshold = float(strategy.split("_")[-1]) / 100.0  # rel_1 -> 0.01 etc.
        rel_threshold = -threshold
        bear = int(row["bear_confirm"]) == 1
        rel = float(row["rel_strength"])
        if rank <= cfg.bottom_quantile and trend_sign < 0.0 and rel <= rel_threshold and bear:
            return -1.0, float(max(1e-6, cfg.bottom_quantile - rank))
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


def _simulate_portfolio(test_panel: pd.DataFrame, trades: pd.DataFrame, spy_df: pd.DataFrame, cfg: Config) -> dict[str, float]:
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
        return {
            "n_days": float(len(rets)),
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
    net_list: list[float] = []
    alpha_list: list[float] = []
    turnover_list: list[float] = []
    gross_exposure_list: list[float] = []

    equity = 1.0
    runmax = 1.0
    max_dd = 0.0

    for i in range(1, len(dates)):
        active = trades[(trades["start_idx"] <= i) & (trades["end_idx"] >= i)]
        w = _weights_from_active(active, cfg, assets)
        gross = float(np.dot(w.values, rets.iloc[i].reindex(assets).fillna(0.0).values))
        turnover = float(np.abs(w - prev_w).sum())
        cost = turnover * one_way_cost
        net = gross - cost
        spy_r = float(spy_ret.iloc[i]) if np.isfinite(spy_ret.iloc[i]) else 0.0
        alpha = net - spy_r

        net_list.append(net)
        alpha_list.append(alpha)
        turnover_list.append(turnover)
        gross_exposure_list.append(float(np.abs(w).sum()))

        equity *= (1.0 + net)
        runmax = max(runmax, equity)
        dd = equity / runmax - 1.0
        max_dd = min(max_dd, dd)

        prev_w = w

    out = pd.DataFrame({"net": net_list, "alpha": alpha_list})
    if out.empty:
        ann_return = cagr = sharpe_n = sharpe_a = float("nan")
    else:
        ann_return = float(np.exp(np.log1p(out["net"]).mean() * 252.0) - 1.0)
        years = max(1e-9, (len(out) / 252.0))
        cagr = float(equity ** (1.0 / years) - 1.0) if equity > 0 else float("nan")
        sharpe_n = (
            float((out["net"].mean() / out["net"].std(ddof=1)) * math.sqrt(252.0))
            if len(out) > 1 and float(out["net"].std(ddof=1)) > 0
            else float("nan")
        )
        sharpe_a = (
            float((out["alpha"].mean() / out["alpha"].std(ddof=1)) * math.sqrt(252.0))
            if len(out) > 1 and float(out["alpha"].std(ddof=1)) > 0
            else float("nan")
        )

    return {
        "n_days": float(len(out)),
        "n_trades": float(len(trades)),
        "annual_return": ann_return,
        "cagr": cagr,
        "max_drawdown": float(max_dd),
        "annualized_sharpe_net": sharpe_n,
        "annualized_sharpe_alpha": sharpe_a,
        "mean_hold_days": float(pd.to_numeric(trades["hold_days"], errors="coerce").mean()) if not trades.empty else float("nan"),
        "avg_gross_exposure": float(np.mean(gross_exposure_list)) if gross_exposure_list else float("nan"),
        "avg_turnover": float(np.mean(turnover_list)) if turnover_list else float("nan"),
    }


def main() -> None:
    out_root = Path("trading_research/examples/_output/26_ranked_vwap_smart_short_filter")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(list(UNIVERSE_130), period="3y", interval="1d")
    spy = vendor.fetch_bars(["SPY"], period="3y", interval="1d")[["timestamp", "close"]]
    if bars.empty or spy.empty:
        raise ValueError("Missing data for smart-short filter test.")

    panel = _build_panel(bars, spy, cfg)
    _, test, split_ts = _split_time(panel, cfg.test_ratio)

    strategies = ["baseline_top_decile_momentum"] + [
        f"smart_short_rel_{int(abs(th) * 100):02d}" for th in cfg.short_rel_strength_thresholds
    ]

    rows: list[dict[str, float | str]] = []
    for strategy in strategies:
        trades = _build_trades(test, strategy, cfg)
        perf = _simulate_portfolio(test, trades, spy, cfg)
        rows.append({"strategy": strategy, **perf})
        trades.to_csv(reports_dir / f"trades_{strategy}.csv", index=False)

    summary_df = pd.DataFrame(rows).sort_values(["annualized_sharpe_alpha", "cagr"], ascending=[False, False]).reset_index(drop=True)

    baseline = summary_df[summary_df["strategy"] == "baseline_top_decile_momentum"]
    baseline_row = baseline.iloc[0].to_dict() if not baseline.empty else {}
    best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}

    uplift = {}
    if baseline_row and best:
        uplift = {
            "best_strategy": str(best["strategy"]),
            "delta_cagr_best_minus_baseline": float(best["cagr"] - baseline_row["cagr"]),
            "delta_alpha_sharpe_best_minus_baseline": float(
                best["annualized_sharpe_alpha"] - baseline_row["annualized_sharpe_alpha"]
            ),
            "delta_max_dd_best_minus_baseline": float(best["max_drawdown"] - baseline_row["max_drawdown"]),
            "delta_net_sharpe_best_minus_baseline": float(
                best["annualized_sharpe_net"] - baseline_row["annualized_sharpe_net"]
            ),
        }

    metadata = {
        "split_timestamp": str(split_ts),
        "universe_size_requested": len(UNIVERSE_130),
        "universe_size_fetched": int(pd.Series(bars["asset"]).nunique()),
        "config": {
            "signal_lookback_days": cfg.signal_lookback_days,
            "rolling_vwap_window": cfg.rolling_vwap_window,
            "hold_days": cfg.hold_days,
            "top_quantile": cfg.top_quantile,
            "bottom_quantile": cfg.bottom_quantile,
            "short_rel_strength_thresholds": list(cfg.short_rel_strength_thresholds),
            "one_way_cost_return": _one_way_cost_return(cfg),
        },
        "uplift_vs_baseline": uplift,
    }

    summary_df.to_csv(reports_dir / "strategy_comparison.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("split_timestamp:", metadata["split_timestamp"])
    print(summary_df.to_string(index=False))
    print("\nuplift:", json.dumps(uplift, indent=2))


if __name__ == "__main__":
    main()

