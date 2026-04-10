"""Test pooled historical percentile entries vs daily cross-sectional ranking.

Idea under test:
- Instead of selecting by daily cross-sectional rank only, use a pooled
  historical percentile of normalized VWAP distortion across all assets.
- This should reduce overtrading by requiring historically extreme signals.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data.vendors.yahoo import YahooMarketDataVendor

UNIVERSE_170 = (
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
)


@dataclass(frozen=True)
class Config:
    period: str = "10y"
    interval: str = "1d"
    signal_lookback_days: int = 63
    rolling_vwap_window: int = 20
    vol_window_days: int = 21
    rel_strength_threshold: float = 0.02
    hold_days: int = 63
    min_train_years: int = 3
    pooled_min_history_obs: int = 3000
    gross_target: float = 1.0
    max_abs_weight_per_asset: float = 0.04
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0


@dataclass(frozen=True)
class StrategySpec:
    name: str
    mode: str  # "baseline_top8", "pooled"
    pooled_q: float | None = None
    require_xs_top_quantile: float | None = None


STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(name="rel2_tight_top8", mode="baseline_top8"),
    StrategySpec(name="pooled_norm_q97_5", mode="pooled", pooled_q=0.975),
    StrategySpec(name="pooled_norm_q99", mode="pooled", pooled_q=0.99),
    StrategySpec(
        name="pooled_norm_q97_5_xs_top20",
        mode="pooled",
        pooled_q=0.975,
        require_xs_top_quantile=0.80,
    ),
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
    frame["ret_1d"] = g["close"].pct_change()
    frame["vol_21"] = (
        g["ret_1d"]
        .rolling(cfg.vol_window_days, min_periods=cfg.vol_window_days)
        .std()
        .reset_index(level=0, drop=True)
    )
    frame["score"] = np.sign(frame["past_return"]) * (frame["close"] / frame["rolling_vwap"] - 1.0)
    frame["score_norm"] = frame["score"] / frame["vol_21"].clip(lower=1e-6)
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
            "score_norm",
            "rank_pct",
            "spy_close",
            "rel_strength_63",
        ]
    )
    frame["year"] = frame["timestamp"].dt.year
    return frame.reset_index(drop=True)


def _quantile_col(q: float) -> str:
    return f"pooled_q_{str(q).replace('.', '_')}"


def _build_pooled_threshold_frame(panel: pd.DataFrame, quantiles: list[float], min_history_obs: int) -> pd.DataFrame:
    dates = sorted(panel["timestamp"].dropna().unique())
    values_by_date = panel.groupby("timestamp")["score_norm"].apply(
        lambda s: pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    )
    history: list[float] = []
    rows: list[dict[str, object]] = []
    for ts in dates:
        row: dict[str, object] = {"timestamp": ts}
        if len(history) >= min_history_obs:
            arr = np.asarray(history, dtype=float)
            for q in quantiles:
                row[_quantile_col(q)] = float(np.quantile(arr, q))
        else:
            for q in quantiles:
                row[_quantile_col(q)] = float("nan")
        rows.append(row)
        vals = values_by_date.get(ts, np.array([], dtype=float))
        if len(vals):
            history.extend(vals.tolist())
    return pd.DataFrame(rows)


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


def _build_trades(test_panel: pd.DataFrame, cfg: Config, spec: StrategySpec, fold_id: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    all_dates = sorted(test_panel["timestamp"].unique())
    date_to_idx = {ts: i for i, ts in enumerate(all_dates)}
    n_dates = len(all_dates)
    if n_dates < 3:
        return pd.DataFrame()

    q_col = _quantile_col(float(spec.pooled_q)) if spec.pooled_q is not None else None
    for asset, g in test_panel.groupby("asset", sort=False):
        g = g.sort_values("timestamp").copy()
        last_end = -1
        for _, r in g.iterrows():
            rank = float(r["rank_pct"])
            trend_sign = float(np.sign(float(r["past_return"])))
            rel = float(r["rel_strength_63"])
            if rel < cfg.rel_strength_threshold:
                continue
            if not np.isfinite(trend_sign) or trend_sign == 0.0:
                continue

            passed = False
            if spec.mode == "baseline_top8":
                passed = rank >= 0.92
                raw_strength = rank - 0.92
            elif spec.mode == "pooled":
                if q_col is None or q_col not in g.columns:
                    continue
                qv = float(r[q_col])
                sn = float(r["score_norm"])
                passed = np.isfinite(qv) and np.isfinite(sn) and (sn >= qv)
                if spec.require_xs_top_quantile is not None:
                    passed = bool(passed and rank >= float(spec.require_xs_top_quantile))
                raw_strength = sn - qv if np.isfinite(sn) and np.isfinite(qv) else 0.0
            else:
                raise ValueError(f"Unknown mode: {spec.mode}")
            if not passed:
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
                    "strategy": spec.name,
                    "asset": str(asset),
                    "signal_timestamp": str(r["timestamp"]),
                    "start_idx": float(start_idx),
                    "end_idx": float(end_idx),
                    "signal_sign": float(trend_sign),
                    "signal_strength": float(max(1e-6, raw_strength)),
                    "hold_days": float(end_idx - sig_idx),
                }
            )
            last_end = end_idx
    return pd.DataFrame(rows)


def _simulate_fold(
    test_panel: pd.DataFrame, trades: pd.DataFrame, spy_df: pd.DataFrame, cfg: Config, fold_id: str, strategy: str
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
    out["strategy"] = strategy

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
            rt_cost = 2.0 * _one_way_cost_return(cfg)
            net_ret = gross_ret - rt_cost
            spy_entry = float(spy.loc[entry_ts]) if entry_ts in spy.index else float("nan")
            spy_exit = float(spy.loc[exit_ts]) if exit_ts in spy.index else float("nan")
            bench = (
                spy_exit / spy_entry - 1.0
                if np.isfinite(spy_entry) and np.isfinite(spy_exit) and spy_entry > 0
                else float("nan")
            )
            trade_rows.append(
                {
                    "fold_id": fold_id,
                    "strategy": strategy,
                    "asset": asset,
                    "entry_timestamp": str(entry_ts),
                    "exit_timestamp": str(exit_ts),
                    "hold_days": float(eidx - sidx),
                    "signal_strength": float(r["signal_strength"]),
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
            "mean_daily_alpha": float("nan"),
            "annualized_mean_alpha_daily": float("nan"),
            "avg_turnover": float("nan"),
            "avg_gross_exposure": float("nan"),
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
    mean_daily_alpha = float(pd.to_numeric(daily["alpha_return"], errors="coerce").mean())
    return {
        "n_days": float(len(daily)),
        "n_trades": float(len(trades)),
        "annual_return": ann,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "annualized_sharpe_net": sr_n,
        "annualized_sharpe_alpha": sr_a,
        "mean_daily_alpha": mean_daily_alpha,
        "annualized_mean_alpha_daily": float(mean_daily_alpha * 252.0),
        "avg_turnover": float(daily["turnover"].mean()),
        "avg_gross_exposure": float(daily["gross_exposure"].mean()),
    }


def main() -> None:
    out_root = Path("trading_research/examples/_output/32_rel2_pooled_percentile_study")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(list(UNIVERSE_170), period=cfg.period, interval=cfg.interval)
    spy_df = vendor.fetch_bars(["SPY"], period=cfg.period, interval=cfg.interval)[["timestamp", "close"]]
    if bars.empty or spy_df.empty:
        raise ValueError("Missing Yahoo data for requested universe or SPY benchmark.")

    panel = _build_panel(bars, spy_df, cfg)
    pooled_qs = sorted({float(s.pooled_q) for s in STRATEGIES if s.pooled_q is not None})
    thresholds = _build_pooled_threshold_frame(panel, pooled_qs, min_history_obs=cfg.pooled_min_history_obs)
    panel = panel.merge(thresholds, on="timestamp", how="left")

    years = sorted(panel["year"].dropna().unique().tolist())
    if len(years) <= cfg.min_train_years:
        raise ValueError("Insufficient years for anchored walk-forward.")

    all_daily_by_strategy: dict[str, list[pd.DataFrame]] = {s.name: [] for s in STRATEGIES}
    all_trades_by_strategy: dict[str, list[pd.DataFrame]] = {s.name: [] for s in STRATEGIES}
    fold_rows: list[dict[str, float | str]] = []

    for test_year in years[cfg.min_train_years :]:
        fold_id = f"fold_{int(test_year)}"
        train = panel[panel["year"] < test_year].copy()
        test = panel[panel["year"] == test_year].copy()
        if train.empty or test.empty:
            continue
        for spec in STRATEGIES:
            trades = _build_trades(test, cfg, spec, fold_id=fold_id)
            daily, trades_eval = _simulate_fold(test, trades, spy_df=spy_df, cfg=cfg, fold_id=fold_id, strategy=spec.name)
            if daily.empty:
                continue
            all_daily_by_strategy[spec.name].append(daily)
            all_trades_by_strategy[spec.name].append(trades_eval)
            fold_metrics = _portfolio_metrics(daily, trades_eval)
            fold_rows.append(
                {
                    "strategy": spec.name,
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
    for spec in STRATEGIES:
        name = spec.name
        daily_all = (
            pd.concat([d for d in all_daily_by_strategy[name] if not d.empty], ignore_index=True)
            if all_daily_by_strategy[name]
            else pd.DataFrame()
        )
        trade_parts = [t for t in all_trades_by_strategy[name] if not t.empty]
        trades_all = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
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
        overall_rows.append({"strategy": name, **_portfolio_metrics(daily_all, trades_all)})

    overall_df = pd.DataFrame(overall_rows).sort_values(
        ["annualized_sharpe_alpha", "annualized_sharpe_net", "annual_return"],
        ascending=[False, False, False],
    )
    yearly_df = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
    daily_oos = pd.concat(daily_all_list, ignore_index=True) if daily_all_list else pd.DataFrame()
    trades_oos = pd.concat(trades_all_list, ignore_index=True) if trades_all_list else pd.DataFrame()

    base = overall_df[overall_df["strategy"] == "rel2_tight_top8"]
    best = overall_df.iloc[0].to_dict() if not overall_df.empty else {}
    uplift = {}
    if not base.empty and best:
        b = base.iloc[0]
        uplift = {
            "best_strategy": str(best["strategy"]),
            "delta_annual_return_vs_rel2_top8": float(best["annual_return"] - b["annual_return"]),
            "delta_cagr_vs_rel2_top8": float(best["cagr"] - b["cagr"]),
            "delta_net_sharpe_vs_rel2_top8": float(best["annualized_sharpe_net"] - b["annualized_sharpe_net"]),
            "delta_alpha_sharpe_vs_rel2_top8": float(
                best["annualized_sharpe_alpha"] - b["annualized_sharpe_alpha"]
            ),
            "delta_maxdd_vs_rel2_top8": float(best["max_drawdown"] - b["max_drawdown"]),
            "delta_n_trades_vs_rel2_top8": float(best["n_trades"] - b["n_trades"]),
        }

    summary = {
        "universe_size_requested": len(UNIVERSE_170),
        "universe_size_fetched": int(panel["asset"].nunique()) if not panel.empty else 0,
        "history_start": str(panel["timestamp"].min()) if not panel.empty else None,
        "history_end": str(panel["timestamp"].max()) if not panel.empty else None,
        "n_walkforward_folds": int(fold_df["fold_id"].nunique()) if not fold_df.empty else 0,
        "config": {
            "period": cfg.period,
            "interval": cfg.interval,
            "signal_lookback_days": cfg.signal_lookback_days,
            "rolling_vwap_window": cfg.rolling_vwap_window,
            "vol_window_days": cfg.vol_window_days,
            "rel_strength_threshold": cfg.rel_strength_threshold,
            "hold_days": cfg.hold_days,
            "pooled_min_history_obs": cfg.pooled_min_history_obs,
            "one_way_cost_return": _one_way_cost_return(cfg),
        },
        "strategies": [s.__dict__ for s in STRATEGIES],
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

