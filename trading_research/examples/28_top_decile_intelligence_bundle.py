"""Top-decile intelligence bundle test (low-overfit ablation).

Compares baseline rel2 strategy vs incremental intelligence layers:
1) Adaptive top-tail selection (7-12% based on signal dispersion),
2) Correlation-aware entry filtering,
3) Vol-targeted position sizing,
4) Regime-aware trailing stop design.

All variants share the same data, costs, and portfolio simulator.
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
class BundleConfig:
    signal_lookback_days: int = 63
    rolling_vwap_window: int = 20
    rel_strength_threshold: float = 0.02
    test_ratio: float = 0.30
    gross_target: float = 1.0
    max_abs_weight_per_asset: float = 0.05
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    # top-decile base + adaptive band
    base_top_quantile: float = 0.90
    top_quantile_wide: float = 0.88
    top_quantile_tight: float = 0.93
    # corr selection / vol targeting
    corr_lookback_days: int = 63
    corr_cap: float = 0.55
    vol_window_days: int = 21
    vol_floor: float = 1e-4
    # stop design
    max_hold_days: int = 63
    min_hold_days: int = 5
    low_vol_trailing_stop: float = 0.10
    high_vol_trailing_stop: float = 0.15


@dataclass(frozen=True)
class StrategySpec:
    name: str
    adaptive_top: bool
    corr_filter: bool
    vol_target: bool
    regime_stop: bool


def _split_time(panel: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = sorted(panel["timestamp"].dropna().unique())
    split_idx = int((1.0 - test_ratio) * len(dates))
    split_idx = min(max(split_idx, 1), len(dates) - 1)
    split_ts = pd.Timestamp(dates[split_idx])
    train = panel[panel["timestamp"] < split_ts].copy()
    test = panel[panel["timestamp"] >= split_ts].copy()
    return train, test, split_ts


def _one_way_cost_return(cfg: BundleConfig) -> float:
    bps = cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side
    return bps / 10_000.0


def _build_panel(bars: pd.DataFrame, spy_close: pd.DataFrame, cfg: BundleConfig) -> pd.DataFrame:
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
    frame["vol_21"] = g["close"].pct_change().rolling(cfg.vol_window_days, min_periods=cfg.vol_window_days).std().reset_index(level=0, drop=True)
    frame["score"] = np.sign(frame["past_return"]) * (frame["close"] / frame["rolling_vwap"] - 1.0)
    frame["rank_pct"] = frame.groupby("timestamp")["score"].rank(pct=True, method="average")

    spy = spy_close.copy()
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True, errors="coerce")
    spy = spy.dropna(subset=["timestamp", "spy_close"]).sort_values("timestamp")
    spy["spy_ret_63"] = spy["spy_close"].pct_change(look)
    spy["spy_vol_21"] = spy["spy_close"].pct_change().rolling(cfg.vol_window_days, min_periods=cfg.vol_window_days).std()
    frame = frame.merge(spy[["timestamp", "spy_ret_63", "spy_vol_21"]], on="timestamp", how="left")
    frame["rel_strength_63"] = frame["past_return"] - frame["spy_ret_63"]
    frame = frame.dropna(
        subset=["close", "score", "rank_pct", "past_return", "rel_strength_63", "vol_21", "spy_vol_21"]
    ).reset_index(drop=True)
    return frame


def _top_quantile_for_date(
    ts: pd.Timestamp,
    adaptive: bool,
    cfg: BundleConfig,
    dispersion: pd.Series,
    disp_q_low: float,
    disp_q_high: float,
) -> float:
    if not adaptive:
        return cfg.base_top_quantile
    d = float(dispersion.get(ts, np.nan))
    if not np.isfinite(d):
        return cfg.base_top_quantile
    if d >= disp_q_high:
        return cfg.top_quantile_tight
    if d <= disp_q_low:
        return cfg.top_quantile_wide
    return cfg.base_top_quantile


def _build_trades_for_strategy(
    test_panel: pd.DataFrame,
    close_wide: pd.DataFrame,
    ret_wide: pd.DataFrame,
    spec: StrategySpec,
    cfg: BundleConfig,
    dispersion: pd.Series,
    disp_q_low: float,
    disp_q_high: float,
    spy_vol_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    test_panel = test_panel.sort_values(["timestamp", "asset"]).copy()
    all_dates = sorted(test_panel["timestamp"].unique())
    date_to_idx = {ts: i for i, ts in enumerate(all_dates)}
    n_dates = len(all_dates)
    last_end_idx: dict[str, int] = {}

    by_date = {ts: g.copy() for ts, g in test_panel.groupby("timestamp", sort=True)}
    for ts in all_dates:
        g = by_date.get(ts)
        if g is None or g.empty:
            continue
        top_q = _top_quantile_for_date(ts, spec.adaptive_top, cfg, dispersion, disp_q_low, disp_q_high)
        cand = g[(g["rank_pct"] >= top_q) & (g["rel_strength_63"] >= cfg.rel_strength_threshold)].copy()
        if cand.empty:
            continue
        cand = cand.sort_values("score", ascending=False).copy()
        cand["signal_sign"] = np.sign(cand["past_return"]).replace(0.0, np.nan)
        cand["signal_strength"] = (cand["rank_pct"] - top_q).clip(lower=1e-6)
        cand["risk_scale"] = 1.0 / cand["vol_21"].clip(lower=cfg.vol_floor)
        cand = cand.dropna(subset=["signal_sign", "signal_strength", "risk_scale"])
        if cand.empty:
            continue

        sig_idx = date_to_idx.get(ts)
        if sig_idx is None:
            continue
        cand["sig_idx"] = sig_idx
        cand = cand[cand["asset"].map(lambda a: sig_idx > int(last_end_idx.get(str(a), -1)))]
        if cand.empty:
            continue

        if spec.corr_filter:
            hist = ret_wide.loc[:ts].tail(cfg.corr_lookback_days)
            selected_assets: list[str] = []
            filtered_idx: list[int] = []
            corr = hist.corr() if len(hist) >= 20 else pd.DataFrame()
            for i, r in cand.iterrows():
                a = str(r["asset"])
                if not selected_assets:
                    selected_assets.append(a)
                    filtered_idx.append(i)
                    continue
                if corr.empty or a not in corr.index:
                    selected_assets.append(a)
                    filtered_idx.append(i)
                    continue
                vals = []
                for s in selected_assets:
                    if s in corr.columns:
                        v = float(corr.loc[a, s])
                        if np.isfinite(v):
                            vals.append(abs(v))
                mean_abs_corr = float(np.mean(vals)) if vals else 0.0
                if mean_abs_corr <= cfg.corr_cap:
                    selected_assets.append(a)
                    filtered_idx.append(i)
            cand = cand.loc[filtered_idx].copy()
            if cand.empty:
                continue

        for _, r in cand.iterrows():
            asset = str(r["asset"])
            sign = float(r["signal_sign"])
            strength = float(r["signal_strength"])
            risk_scale = float(r["risk_scale"])
            sig_idx = int(r["sig_idx"])
            start_idx = sig_idx + 1
            if start_idx >= n_dates:
                continue
            end_cap = min(sig_idx + cfg.max_hold_days, n_dates - 1)
            if end_cap <= start_idx:
                continue

            exit_idx = end_cap
            exit_reason = "max_hold"
            entry_price = float(r["close"])
            peak_signed_ret = 0.0
            if spec.regime_stop:
                spy_vol = float(r["spy_vol_21"])
                stop_thr = cfg.high_vol_trailing_stop if spy_vol >= spy_vol_threshold else cfg.low_vol_trailing_stop
                for j in range(start_idx, end_cap + 1):
                    ts_j = all_dates[j]
                    px = close_wide.loc[ts_j, asset] if asset in close_wide.columns and ts_j in close_wide.index else np.nan
                    if not np.isfinite(px) or entry_price <= 0:
                        continue
                    signed_ret_now = sign * (float(px) / entry_price - 1.0)
                    peak_signed_ret = max(peak_signed_ret, signed_ret_now)
                    if (j - sig_idx) < cfg.min_hold_days:
                        continue
                    if (peak_signed_ret - signed_ret_now) >= stop_thr:
                        exit_idx = j
                        exit_reason = "regime_trailing_stop"
                        break

            rows.append(
                {
                    "strategy": spec.name,
                    "asset": asset,
                    "signal_timestamp": str(ts),
                    "start_idx": float(start_idx),
                    "end_idx": float(exit_idx),
                    "signal_sign": sign,
                    "signal_strength": strength,
                    "risk_scale": risk_scale if spec.vol_target else 1.0,
                    "hold_days": float(max(1, exit_idx - sig_idx)),
                    "exit_reason": exit_reason,
                }
            )
            last_end_idx[asset] = exit_idx

    return pd.DataFrame(rows)


def _weights_from_active(active: pd.DataFrame, cfg: BundleConfig, assets: list[str]) -> pd.Series:
    w = pd.Series(0.0, index=assets, dtype=float)
    if active.empty:
        return w
    by_asset = active.groupby("asset", as_index=False).agg(
        raw=("signal_strength", lambda s: float(np.sum(s))),
        sign=("signal_sign", lambda s: float(np.sign(np.sum(s)))),
        risk=("risk_scale", lambda s: float(np.mean(s))),
    )
    by_asset["raw_signed"] = by_asset["raw"] * by_asset["sign"] * by_asset["risk"]
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
    cfg: BundleConfig,
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
        out = pd.DataFrame({"timestamp": rets.index, "net_return": 0.0, "spy_return": spy_ret.values, "alpha_return": -spy_ret.values})
        out["turnover"] = 0.0
        out["gross_exposure"] = 0.0
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
    trades = trades.dropna(subset=["start_idx", "end_idx", "asset", "signal_sign", "signal_strength", "risk_scale"]).copy()

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

    ann_return = float(np.exp(np.log1p(out["net_return"]).mean() * 252.0) - 1.0) if not out.empty else float("nan")
    if out.empty:
        cagr = max_dd = sharpe_n = sharpe_a = float("nan")
    else:
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
    out_root = Path("trading_research/examples/_output/28_top_decile_intelligence_bundle")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = BundleConfig()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(list(UNIVERSE_130), period="3y", interval="1d")
    spy = vendor.fetch_bars(["SPY"], period="3y", interval="1d")[["timestamp", "close"]].rename(columns={"close": "spy_close"})
    if bars.empty or spy.empty:
        raise ValueError("Missing data for intelligence-bundle test.")

    panel = _build_panel(bars, spy, cfg)
    train, test, split_ts = _split_time(panel, cfg.test_ratio)

    # train-derived calibration (avoid overfitting on test)
    disp_train = train.groupby("timestamp")["score"].std()
    disp_q_low = float(disp_train.quantile(0.33))
    disp_q_high = float(disp_train.quantile(0.67))
    disp_all = panel.groupby("timestamp")["score"].std()
    spy_vol_threshold = float(train["spy_vol_21"].median())

    close_wide = panel.pivot(index="timestamp", columns="asset", values="close").sort_index().ffill()
    ret_wide = close_wide.pct_change()

    strategies = [
        StrategySpec(name="baseline_rel2", adaptive_top=False, corr_filter=False, vol_target=False, regime_stop=False),
        StrategySpec(name="adaptive_top_rel2", adaptive_top=True, corr_filter=False, vol_target=False, regime_stop=False),
        StrategySpec(name="adaptive_corr_volsize_rel2", adaptive_top=True, corr_filter=True, vol_target=True, regime_stop=False),
        StrategySpec(name="full_intelligent_bundle_rel2", adaptive_top=True, corr_filter=True, vol_target=True, regime_stop=True),
    ]

    rows: list[dict[str, float | str]] = []
    for spec in strategies:
        trades = _build_trades_for_strategy(
            test_panel=test,
            close_wide=close_wide,
            ret_wide=ret_wide,
            spec=spec,
            cfg=cfg,
            dispersion=disp_all,
            disp_q_low=disp_q_low,
            disp_q_high=disp_q_high,
            spy_vol_threshold=spy_vol_threshold,
        )
        daily, perf = _simulate_portfolio(test, trades, spy_df=spy.rename(columns={"spy_close": "close"}), cfg=cfg)
        rows.append({"strategy": spec.name, **perf})
        trades.to_csv(reports_dir / f"trades_{spec.name}.csv", index=False)
        daily.to_csv(reports_dir / f"portfolio_daily_{spec.name}.csv", index=False)

    summary_df = pd.DataFrame(rows).sort_values(["cagr", "annualized_sharpe_alpha"], ascending=False).reset_index(drop=True)
    baseline = summary_df[summary_df["strategy"] == "baseline_rel2"]
    best = summary_df.iloc[0] if not summary_df.empty else None
    uplift = {}
    if best is not None and not baseline.empty:
        b = baseline.iloc[0]
        uplift = {
            "best_strategy": str(best["strategy"]),
            "delta_cagr_best_minus_baseline": float(best["cagr"] - b["cagr"]),
            "delta_alpha_sharpe_best_minus_baseline": float(best["annualized_sharpe_alpha"] - b["annualized_sharpe_alpha"]),
            "delta_net_sharpe_best_minus_baseline": float(best["annualized_sharpe_net"] - b["annualized_sharpe_net"]),
            "delta_max_dd_best_minus_baseline": float(best["max_drawdown"] - b["max_drawdown"]),
        }

    metadata = {
        "split_timestamp": str(split_ts),
        "universe_size_requested": len(UNIVERSE_130),
        "universe_size_fetched": int(pd.Series(bars["asset"]).nunique()),
        "config": {
            "signal_lookback_days": cfg.signal_lookback_days,
            "rolling_vwap_window": cfg.rolling_vwap_window,
            "rel_strength_threshold": cfg.rel_strength_threshold,
            "base_top_quantile": cfg.base_top_quantile,
            "top_quantile_wide": cfg.top_quantile_wide,
            "top_quantile_tight": cfg.top_quantile_tight,
            "corr_cap": cfg.corr_cap,
            "max_hold_days": cfg.max_hold_days,
            "regime_stops": {"low_vol": cfg.low_vol_trailing_stop, "high_vol": cfg.high_vol_trailing_stop},
            "one_way_cost_return": _one_way_cost_return(cfg),
        },
        "train_calibration": {
            "disp_q_low": disp_q_low,
            "disp_q_high": disp_q_high,
            "spy_vol_threshold": spy_vol_threshold,
        },
        "uplift": uplift,
    }

    summary_df.to_csv(reports_dir / "strategy_comparison.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("split_timestamp:", metadata["split_timestamp"])
    print(summary_df.to_string(index=False))
    print("\nuplift:", json.dumps(uplift, indent=2))


if __name__ == "__main__":
    main()

