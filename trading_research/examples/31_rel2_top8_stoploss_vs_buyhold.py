"""Evaluate rel2_tight_top8 stop-loss variants vs buy-and-hold-all.

Focus:
- Keep the best recent entry logic (rel2_tight_top8).
- Compare multiple simple stop-loss configurations.
- Benchmark all variants against a buy-and-hold-all universe portfolio.
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
    top_quantile: float = 0.92
    hold_days: int = 63
    min_train_years: int = 3
    gross_target: float = 1.0
    max_abs_weight_per_asset: float = 0.04
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0


@dataclass(frozen=True)
class StopSpec:
    name: str
    stop_loss_pct: float | None
    trailing_stop_pct: float | None


STOP_SPECS: tuple[StopSpec, ...] = (
    StopSpec(name="rel2_top8_no_stop", stop_loss_pct=None, trailing_stop_pct=None),
    StopSpec(name="rel2_top8_stop_08", stop_loss_pct=0.08, trailing_stop_pct=None),
    StopSpec(name="rel2_top8_stop_10", stop_loss_pct=0.10, trailing_stop_pct=None),
    StopSpec(name="rel2_top8_trailing_10", stop_loss_pct=None, trailing_stop_pct=0.10),
    StopSpec(name="rel2_top8_trailing_12", stop_loss_pct=None, trailing_stop_pct=0.12),
    StopSpec(name="rel2_top8_stop08_trailing12", stop_loss_pct=0.08, trailing_stop_pct=0.12),
)


def _one_way_cost_return(cfg: Config) -> float:
    return (cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side) / 10_000.0


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
        subset=[
            "timestamp",
            "asset",
            "close",
            "past_return",
            "score",
            "rank_pct",
            "spy_close",
            "rel_strength_63",
        ]
    )
    frame["year"] = frame["timestamp"].dt.year
    return frame.reset_index(drop=True)


def _weights_from_active(active: pd.DataFrame, cfg: Config, assets: list[str]) -> pd.Series:
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


def _build_trades_for_stop(
    test_panel: pd.DataFrame,
    cfg: Config,
    stop_spec: StopSpec,
    fold_id: str,
) -> pd.DataFrame:
    close_wide = test_panel.pivot(index="timestamp", columns="asset", values="close").sort_index().ffill()
    all_dates = close_wide.index.to_list()
    date_to_idx = {ts: i for i, ts in enumerate(all_dates)}
    n_dates = len(all_dates)
    if n_dates < 3:
        return pd.DataFrame()

    rows: list[dict[str, float | str]] = []
    for asset, g in test_panel.groupby("asset", sort=False):
        if asset not in close_wide.columns:
            continue
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
            hard_end = min(sig_idx + cfg.hold_days, n_dates - 1)
            if start_idx >= n_dates or hard_end <= start_idx:
                continue

            entry_ts = all_dates[start_idx]
            entry_px = float(close_wide.loc[entry_ts, asset])
            if not np.isfinite(entry_px) or entry_px <= 0.0:
                continue

            exit_idx = hard_end
            exit_reason = "fixed_hold"
            peak_signed = 0.0
            for j in range(start_idx, hard_end + 1):
                ts_j = all_dates[j]
                px = float(close_wide.loc[ts_j, asset])
                if not np.isfinite(px) or px <= 0.0:
                    continue
                signed_ret = trend_sign * (px / entry_px - 1.0)
                peak_signed = max(peak_signed, signed_ret)

                if stop_spec.stop_loss_pct is not None and signed_ret <= -float(stop_spec.stop_loss_pct):
                    exit_idx = j
                    exit_reason = f"stop_{int(round(stop_spec.stop_loss_pct * 100)):02d}"
                    break
                if stop_spec.trailing_stop_pct is not None and (peak_signed - signed_ret) >= float(stop_spec.trailing_stop_pct):
                    exit_idx = j
                    exit_reason = f"trailing_{int(round(stop_spec.trailing_stop_pct * 100)):02d}"
                    break

            rows.append(
                {
                    "fold_id": fold_id,
                    "strategy": stop_spec.name,
                    "asset": str(asset),
                    "signal_timestamp": str(r["timestamp"]),
                    "start_idx": float(start_idx),
                    "end_idx": float(exit_idx),
                    "signal_sign": float(trend_sign),
                    "signal_strength": float(max(1e-6, rank - cfg.top_quantile)),
                    "hold_days": float(exit_idx - sig_idx),
                    "exit_reason": exit_reason,
                }
            )
            last_end = exit_idx

    return pd.DataFrame(rows)


def _simulate_fold(
    test_panel: pd.DataFrame,
    trades: pd.DataFrame,
    spy_df: pd.DataFrame,
    cfg: Config,
    fold_id: str,
    strategy_name: str,
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
        out = pd.DataFrame(
            {"timestamp": rets.index, "net_return": 0.0, "spy_return": spy_ret.values, "alpha_return": -spy_ret.values}
        )
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
            w = _weights_from_active(active, cfg, assets)
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
    out["strategy"] = strategy_name

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
            net_ret = gross_ret - 2.0 * _one_way_cost_return(cfg)
            spy_entry = float(spy.loc[entry_ts]) if entry_ts in spy.index else float("nan")
            spy_exit = float(spy.loc[exit_ts]) if exit_ts in spy.index else float("nan")
            bench = spy_exit / spy_entry - 1.0 if np.isfinite(spy_entry) and np.isfinite(spy_exit) and spy_entry > 0 else float("nan")
            trade_rows.append(
                {
                    "fold_id": fold_id,
                    "strategy": strategy_name,
                    "asset": asset,
                    "entry_timestamp": str(entry_ts),
                    "exit_timestamp": str(exit_ts),
                    "hold_days": float(eidx - sidx),
                    "signal_strength": float(r.get("signal_strength", np.nan)),
                    "net_return": net_ret,
                    "benchmark_return": bench,
                    "alpha_return": net_ret - bench if np.isfinite(bench) else float("nan"),
                    "exit_reason": str(r.get("exit_reason", "na")),
                }
            )
    trades_out = pd.DataFrame(trade_rows)
    return out, trades_out


def _simulate_buy_hold_all_fold(
    test_panel: pd.DataFrame,
    spy_df: pd.DataFrame,
    cfg: Config,
    fold_id: str,
) -> pd.DataFrame:
    close = test_panel.pivot(index="timestamp", columns="asset", values="close").sort_index().ffill()
    rets = close.pct_change().fillna(0.0)
    if rets.empty:
        return pd.DataFrame()
    bh = rets.mean(axis=1, skipna=True).fillna(0.0).to_frame(name="net_return").reset_index().rename(columns={"index": "timestamp"})

    # Simple roundtrip approximation: one-way in at start, one-way out at end.
    one_way = _one_way_cost_return(cfg)
    if len(bh) >= 2:
        bh.loc[0, "net_return"] = float(bh.loc[0, "net_return"]) - one_way
        bh.loc[len(bh) - 1, "net_return"] = float(bh.loc[len(bh) - 1, "net_return"]) - one_way

    spy = spy_df.copy()
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True, errors="coerce")
    spy = spy.dropna(subset=["timestamp"]).sort_values("timestamp")
    spy = spy.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(bh["timestamp"]), method="ffill")
    spy_ret = spy.pct_change().fillna(0.0).to_numpy()

    bh["spy_return"] = spy_ret
    bh["alpha_return"] = pd.to_numeric(bh["net_return"], errors="coerce") - pd.Series(spy_ret, index=bh.index)
    bh["turnover"] = 0.0
    bh["gross_exposure"] = 1.0
    bh["equity"] = (1.0 + pd.to_numeric(bh["net_return"], errors="coerce")).cumprod()
    bh["running_max"] = bh["equity"].cummax()
    bh["drawdown"] = bh["equity"] / bh["running_max"] - 1.0
    bh["fold_id"] = fold_id
    bh["strategy"] = "buy_hold_all"
    bh["timestamp"] = pd.to_datetime(bh["timestamp"], utc=True, errors="coerce")
    return bh


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
            "mean_daily_alpha": float("nan"),
            "annualized_mean_alpha_daily": float("nan"),
            "daily_alpha_hit_rate": float("nan"),
            "avg_turnover": float("nan"),
            "avg_gross_exposure": float("nan"),
            "mean_trade_alpha": float("nan"),
        }

    ann = float(np.exp(np.log1p(pd.to_numeric(daily["net_return"], errors="coerce")).mean() * 252.0) - 1.0)
    start, end = daily["timestamp"].iloc[0], daily["timestamp"].iloc[-1]
    years = max(1e-9, (end - start).total_seconds() / (365.25 * 24 * 3600))
    eq_end = float(pd.to_numeric(daily["equity"], errors="coerce").iloc[-1])
    cagr = float(eq_end ** (1.0 / years) - 1.0) if eq_end > 0 else float("nan")
    max_dd = float(pd.to_numeric(daily["drawdown"], errors="coerce").min())
    net = pd.to_numeric(daily["net_return"], errors="coerce")
    alpha = pd.to_numeric(daily["alpha_return"], errors="coerce")
    sr_n = float((net.mean() / net.std(ddof=1)) * math.sqrt(252.0)) if len(net) > 1 and float(net.std(ddof=1)) > 0 else float("nan")
    sr_a = float((alpha.mean() / alpha.std(ddof=1)) * math.sqrt(252.0)) if len(alpha) > 1 and float(alpha.std(ddof=1)) > 0 else float("nan")
    alpha_mean = float(alpha.mean()) if len(alpha) else float("nan")

    return {
        "n_days": float(len(daily)),
        "n_trades": float(len(trades)),
        "annual_return": ann,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "annualized_sharpe_net": sr_n,
        "annualized_sharpe_alpha": sr_a,
        "mean_daily_alpha": alpha_mean,
        "annualized_mean_alpha_daily": float(alpha_mean * 252.0) if np.isfinite(alpha_mean) else float("nan"),
        "daily_alpha_hit_rate": float((alpha > 0).mean()) if len(alpha) else float("nan"),
        "avg_turnover": float(pd.to_numeric(daily["turnover"], errors="coerce").mean()),
        "avg_gross_exposure": float(pd.to_numeric(daily["gross_exposure"], errors="coerce").mean()),
        "mean_trade_alpha": float(pd.to_numeric(trades.get("alpha_return"), errors="coerce").mean()) if not trades.empty else float("nan"),
    }


def main() -> None:
    out_root = Path("trading_research/examples/_output/31_rel2_top8_stoploss_vs_buyhold")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    universe = _load_universe_170()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(list(universe), period=cfg.period, interval=cfg.interval)
    spy_df = vendor.fetch_bars(["SPY"], period=cfg.period, interval=cfg.interval)[["timestamp", "close"]]
    if bars.empty or spy_df.empty:
        raise ValueError("Missing Yahoo data for universe or SPY benchmark.")

    panel = _build_panel(bars, spy_df, cfg)
    years = sorted(panel["year"].dropna().unique().tolist())
    if len(years) <= cfg.min_train_years:
        raise ValueError("Insufficient years for anchored walk-forward.")

    strategy_names = [s.name for s in STOP_SPECS] + ["buy_hold_all"]
    all_daily_by_strategy: dict[str, list[pd.DataFrame]] = {s: [] for s in strategy_names}
    all_trades_by_strategy: dict[str, list[pd.DataFrame]] = {s: [] for s in strategy_names}
    fold_rows: list[dict[str, float | str]] = []

    for test_year in years[cfg.min_train_years :]:
        fold_id = f"fold_{int(test_year)}"
        train = panel[panel["year"] < test_year].copy()
        test = panel[panel["year"] == test_year].copy()
        if train.empty or test.empty:
            continue

        for stop_spec in STOP_SPECS:
            trades = _build_trades_for_stop(test_panel=test, cfg=cfg, stop_spec=stop_spec, fold_id=fold_id)
            daily, trades_eval = _simulate_fold(
                test_panel=test,
                trades=trades,
                spy_df=spy_df,
                cfg=cfg,
                fold_id=fold_id,
                strategy_name=stop_spec.name,
            )
            if daily.empty:
                continue
            all_daily_by_strategy[stop_spec.name].append(daily)
            all_trades_by_strategy[stop_spec.name].append(trades_eval)
            fold_metrics = _portfolio_metrics(daily, trades_eval)
            fold_rows.append(
                {
                    "strategy": stop_spec.name,
                    "fold_id": fold_id,
                    "test_year": int(test_year),
                    "train_years": int(train["year"].nunique()),
                    "assets_in_test": int(test["asset"].nunique()),
                    **fold_metrics,
                }
            )

        bh_daily = _simulate_buy_hold_all_fold(test_panel=test, spy_df=spy_df, cfg=cfg, fold_id=fold_id)
        if not bh_daily.empty:
            all_daily_by_strategy["buy_hold_all"].append(bh_daily)
            all_trades_by_strategy["buy_hold_all"].append(pd.DataFrame())
            fold_metrics = _portfolio_metrics(bh_daily, pd.DataFrame())
            fold_rows.append(
                {
                    "strategy": "buy_hold_all",
                    "fold_id": fold_id,
                    "test_year": int(test_year),
                    "train_years": int(train["year"].nunique()),
                    "assets_in_test": int(test["asset"].nunique()),
                    **fold_metrics,
                }
            )

    fold_df = pd.DataFrame(fold_rows).sort_values(["strategy", "test_year"]).reset_index(drop=True)

    overall_rows: list[dict[str, float | str]] = []
    yearly_rows: list[pd.DataFrame] = []
    daily_all_list: list[pd.DataFrame] = []
    trades_all_list: list[pd.DataFrame] = []
    for name in strategy_names:
        daily_all = (
            pd.concat([d for d in all_daily_by_strategy[name] if not d.empty], ignore_index=True)
            if all_daily_by_strategy[name]
            else pd.DataFrame()
        )
        trades_all = (
            pd.concat([t for t in all_trades_by_strategy[name] if not t.empty], ignore_index=True)
            if all_trades_by_strategy[name]
            else pd.DataFrame()
        )
        if not daily_all.empty:
            daily_all = daily_all.sort_values("timestamp").reset_index(drop=True)
            daily_all["year"] = daily_all["timestamp"].dt.year
            y = (
                daily_all.groupby("year", as_index=False)
                .agg(
                    n_days=("net_return", "count"),
                    mean_net_return=("net_return", "mean"),
                    mean_alpha_return=("alpha_return", "mean"),
                    annualized_sharpe_net=(
                        "net_return",
                        lambda s: float((s.mean() / s.std(ddof=1)) * math.sqrt(252.0))
                        if len(s) > 1 and float(s.std(ddof=1)) > 0
                        else float("nan"),
                    ),
                    annualized_sharpe_alpha=(
                        "alpha_return",
                        lambda s: float((s.mean() / s.std(ddof=1)) * math.sqrt(252.0))
                        if len(s) > 1 and float(s.std(ddof=1)) > 0
                        else float("nan"),
                    ),
                )
                .sort_values("year")
                .reset_index(drop=True)
            )
            y["strategy"] = name
            yearly_rows.append(y)
            daily_all_list.append(daily_all)
        if not trades_all.empty:
            trades_all_list.append(trades_all)

        overall = _portfolio_metrics(daily_all, trades_all)
        overall_rows.append({"strategy": name, **overall})

    overall_df = pd.DataFrame(overall_rows).sort_values(["annualized_sharpe_net", "cagr"], ascending=[False, False]).reset_index(drop=True)
    yearly_df = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
    daily_oos = pd.concat(daily_all_list, ignore_index=True) if daily_all_list else pd.DataFrame()
    trades_oos = pd.concat(trades_all_list, ignore_index=True) if trades_all_list else pd.DataFrame()

    baseline = overall_df[overall_df["strategy"] == "rel2_top8_no_stop"]
    buyhold = overall_df[overall_df["strategy"] == "buy_hold_all"]
    best = overall_df.iloc[0].to_dict() if not overall_df.empty else {}
    uplift = {}
    if not baseline.empty and not buyhold.empty and best:
        b = baseline.iloc[0]
        bh = buyhold.iloc[0]
        uplift = {
            "best_strategy": str(best["strategy"]),
            "delta_cagr_vs_no_stop": float(best["cagr"] - b["cagr"]),
            "delta_sharpe_vs_no_stop": float(best["annualized_sharpe_net"] - b["annualized_sharpe_net"]),
            "delta_maxdd_vs_no_stop": float(best["max_drawdown"] - b["max_drawdown"]),
            "delta_cagr_no_stop_vs_buy_hold_all": float(b["cagr"] - bh["cagr"]),
            "delta_sharpe_no_stop_vs_buy_hold_all": float(b["annualized_sharpe_net"] - bh["annualized_sharpe_net"]),
            "delta_alpha_annmean_no_stop_vs_buy_hold_all": float(
                b["annualized_mean_alpha_daily"] - bh["annualized_mean_alpha_daily"]
            ),
        }

    summary = {
        "universe_size_requested": len(universe),
        "universe_size_fetched": int(panel["asset"].nunique()) if not panel.empty else 0,
        "history_start": str(panel["timestamp"].min()) if not panel.empty else None,
        "history_end": str(panel["timestamp"].max()) if not panel.empty else None,
        "n_walkforward_folds": int(fold_df["fold_id"].nunique()) if not fold_df.empty else 0,
        "strategies": strategy_names,
        "config": {
            "period": cfg.period,
            "interval": cfg.interval,
            "signal_lookback_days": cfg.signal_lookback_days,
            "rolling_vwap_window": cfg.rolling_vwap_window,
            "rel_strength_threshold": cfg.rel_strength_threshold,
            "top_quantile": cfg.top_quantile,
            "hold_days": cfg.hold_days,
            "one_way_cost_return": _one_way_cost_return(cfg),
        },
        "stop_specs": [s.__dict__ for s in STOP_SPECS],
        "uplift": uplift,
    }

    fold_df.to_csv(reports_dir / "walkforward_fold_metrics.csv", index=False)
    yearly_df.to_csv(reports_dir / "walkforward_yearly_metrics.csv", index=False)
    overall_df.to_csv(reports_dir / "strategy_overall_comparison.csv", index=False)
    daily_oos.to_csv(reports_dir / "daily_oos_all_strategies.csv", index=False)
    trades_oos.to_csv(reports_dir / "trades_oos_all_strategies.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("overall comparison:")
    print(overall_df.to_string(index=False))
    print("uplift:", json.dumps(uplift, indent=2))


if __name__ == "__main__":
    main()

