"""Simple-first improvements for baseline_rel2 under rigorous walk-forward.

Compares a few low-complexity variants:
- baseline_rel2 (top decile + rel_strength >= +2%, hold 63d)
- + SPY trend gate (SPY > 200D MA at signal time)
- + SPY low-vol gate (SPY 21D vol <= fold train median)
- tighter rank tail (top 8% instead of top 10%)

All variants run on the same:
- ~170-name universe
- 10y daily history
- anchored yearly OOS folds (minimum 3 train years)
- portfolio simulation with costs and exposure caps
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
    hold_days: int = 63
    min_train_years: int = 3
    gross_target: float = 1.0
    max_abs_weight_per_asset: float = 0.04
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    spy_vol_scale_min: float = 0.50
    spy_vol_scale_max: float = 1.25


@dataclass(frozen=True)
class StrategySpec:
    name: str
    top_quantile: float
    require_spy_up: bool
    require_low_vol: bool
    vol_scaled_exposure: bool
    vol_scale_min: float | None = None
    vol_scale_max: float | None = None


STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(
        name="baseline_rel2",
        top_quantile=0.90,
        require_spy_up=False,
        require_low_vol=False,
        vol_scaled_exposure=False,
    ),
    StrategySpec(
        name="rel2_spyup",
        top_quantile=0.90,
        require_spy_up=True,
        require_low_vol=False,
        vol_scaled_exposure=False,
    ),
    StrategySpec(
        name="rel2_spyup_lowvol",
        top_quantile=0.90,
        require_spy_up=True,
        require_low_vol=True,
        vol_scaled_exposure=False,
    ),
    StrategySpec(
        name="rel2_spyup_tight_top8",
        top_quantile=0.92,
        require_spy_up=True,
        require_low_vol=False,
        vol_scaled_exposure=False,
    ),
    StrategySpec(
        name="rel2_tight_top8",
        top_quantile=0.92,
        require_spy_up=False,
        require_low_vol=False,
        vol_scaled_exposure=False,
    ),
    StrategySpec(
        name="rel2_tight_top8_spyvol_scaled",
        top_quantile=0.92,
        require_spy_up=False,
        require_low_vol=False,
        vol_scaled_exposure=True,
        vol_scale_min=0.50,
        vol_scale_max=1.25,
    ),
    StrategySpec(
        name="rel2_tight_top8_spyvol_scaled_60_120",
        top_quantile=0.92,
        require_spy_up=False,
        require_low_vol=False,
        vol_scaled_exposure=True,
        vol_scale_min=0.60,
        vol_scale_max=1.20,
    ),
    StrategySpec(
        name="rel2_tight_top8_spyvol_scaled_70_110",
        top_quantile=0.92,
        require_spy_up=False,
        require_low_vol=False,
        vol_scaled_exposure=True,
        vol_scale_min=0.70,
        vol_scale_max=1.10,
    ),
    StrategySpec(
        name="rel2_tight_top5",
        top_quantile=0.95,
        require_spy_up=False,
        require_low_vol=False,
        vol_scaled_exposure=False,
    ),
    StrategySpec(
        name="rel2_lowvol",
        top_quantile=0.90,
        require_spy_up=False,
        require_low_vol=True,
        vol_scaled_exposure=False,
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
    frame["score"] = np.sign(frame["past_return"]) * (frame["close"] / frame["rolling_vwap"] - 1.0)
    frame["rank_pct"] = frame.groupby("timestamp")["score"].rank(pct=True, method="average")

    spy = spy.copy()
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True, errors="coerce")
    spy = spy.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    spy["close"] = pd.to_numeric(spy["close"], errors="coerce")
    spy = spy.dropna(subset=["close"]).copy()
    spy["spy_ret_63"] = spy["close"] / spy["close"].shift(look) - 1.0
    spy["spy_ma200"] = spy["close"].rolling(200, min_periods=200).mean()
    spy["spy_up"] = (spy["close"] > spy["spy_ma200"]).astype(float)
    spy["spy_vol_21"] = spy["close"].pct_change().rolling(21, min_periods=21).std()
    spy = spy.rename(columns={"close": "spy_close"})
    frame = frame.merge(
        spy[["timestamp", "spy_close", "spy_ret_63", "spy_up", "spy_vol_21"]],
        on="timestamp",
        how="left",
    )
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
            "spy_up",
            "spy_vol_21",
        ]
    )
    frame["year"] = frame["timestamp"].dt.year
    return frame.reset_index(drop=True)


def _weights_from_active(active: pd.DataFrame, cfg: Config, assets: list[str], gross_target: float) -> pd.Series:
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
    by_asset["w"] = (by_asset["raw_signed"] / denom) * float(gross_target)
    by_asset["w"] = by_asset["w"].clip(-cfg.max_abs_weight_per_asset, cfg.max_abs_weight_per_asset)
    for _, r in by_asset.iterrows():
        a = str(r["asset"])
        if a in w.index:
            w.loc[a] = float(r["w"])
    return w


def _build_trades(
    test_panel: pd.DataFrame,
    cfg: Config,
    strategy: StrategySpec,
    fold_id: str,
    spy_vol_threshold: float,
) -> pd.DataFrame:
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
            spy_up = float(r["spy_up"]) > 0.5
            spy_vol = float(r["spy_vol_21"])

            if rank < strategy.top_quantile:
                continue
            if rel < cfg.rel_strength_threshold:
                continue
            if strategy.require_spy_up and not spy_up:
                continue
            if strategy.require_low_vol and (not np.isfinite(spy_vol) or spy_vol > spy_vol_threshold):
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
                    "strategy": strategy.name,
                    "asset": str(asset),
                    "signal_timestamp": str(r["timestamp"]),
                    "start_idx": float(start_idx),
                    "end_idx": float(end_idx),
                    "signal_sign": float(trend_sign),
                    "signal_strength": float(max(1e-6, rank - strategy.top_quantile)),
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
    strategy: StrategySpec,
    spy_vol_anchor: float,
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
    spy_vol_by_date = test_panel.groupby("timestamp")["spy_vol_21"].median()

    t = trades.copy()
    if t.empty:
        out = pd.DataFrame(
            {
                "timestamp": rets.index,
                "net_return": 0.0,
                "spy_return": spy_ret.values,
                "alpha_return": -spy_ret.values,
            }
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
            ts_i = pd.Timestamp(dates[i])
            if strategy.vol_scaled_exposure:
                spy_vol_now = float(spy_vol_by_date.get(ts_i, np.nan))
                if np.isfinite(spy_vol_now) and spy_vol_now > 0.0 and np.isfinite(spy_vol_anchor) and spy_vol_anchor > 0.0:
                    raw_scale = spy_vol_anchor / spy_vol_now
                    min_scale = float(strategy.vol_scale_min) if strategy.vol_scale_min is not None else cfg.spy_vol_scale_min
                    max_scale = float(strategy.vol_scale_max) if strategy.vol_scale_max is not None else cfg.spy_vol_scale_max
                    exposure_scale = float(np.clip(raw_scale, min_scale, max_scale))
                else:
                    exposure_scale = 1.0
            else:
                exposure_scale = 1.0
            gross_target_day = cfg.gross_target * exposure_scale
            w = _weights_from_active(active, cfg, assets, gross_target=gross_target_day)
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
                    "exposure_scale": exposure_scale,
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
    out["strategy"] = strategy.name

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
            bench = spy_exit / spy_entry - 1.0 if np.isfinite(spy_entry) and np.isfinite(spy_exit) and spy_entry > 0 else float("nan")
            trade_rows.append(
                {
                    "fold_id": fold_id,
                    "strategy": strategy.name,
                    "asset": asset,
                    "entry_timestamp": str(entry_ts),
                    "exit_timestamp": str(exit_ts),
                    "hold_days": float(eidx - sidx),
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
            "mean_trade_alpha": float("nan"),
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
    alpha_daily = pd.to_numeric(daily["alpha_return"], errors="coerce")
    alpha_daily_mean = float(alpha_daily.mean()) if len(alpha_daily) else float("nan")
    alpha_daily_ann_mean = float(alpha_daily_mean * 252.0) if np.isfinite(alpha_daily_mean) else float("nan")
    alpha_daily_hit = float((alpha_daily > 0).mean()) if len(alpha_daily) else float("nan")
    if "alpha_return" in trades.columns:
        alpha_trade = pd.to_numeric(trades["alpha_return"], errors="coerce")
    else:
        alpha_trade = pd.Series(dtype=float, index=trades.index)
    # Proxy for capital-weighted trade alpha: stronger signals held longer get larger weight.
    if "signal_strength" in trades.columns:
        strength = pd.to_numeric(trades["signal_strength"], errors="coerce").abs()
    else:
        strength = pd.Series(1.0, index=trades.index, dtype=float)
    if "hold_days" in trades.columns:
        hold_raw = pd.Series(pd.to_numeric(trades["hold_days"], errors="coerce"), index=trades.index, dtype=float)
    else:
        hold_raw = pd.Series(1.0, index=trades.index, dtype=float)
    hold = hold_raw.where(hold_raw >= 1.0, 1.0)
    trade_w = strength * hold
    mask = alpha_trade.notna() & trade_w.notna() & (trade_w > 0)
    alpha_trade_weighted_proxy = (
        float(np.average(alpha_trade[mask], weights=trade_w[mask]))
        if bool(mask.any())
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
        "mean_daily_alpha": alpha_daily_mean,
        "annualized_mean_alpha_daily": alpha_daily_ann_mean,
        "daily_alpha_hit_rate": alpha_daily_hit,
        "avg_turnover": float(daily["turnover"].mean()),
        "avg_gross_exposure": float(daily["gross_exposure"].mean()),
        "mean_trade_alpha": float(pd.to_numeric(trades.get("alpha_return"), errors="coerce").mean())
        if not trades.empty
        else float("nan"),
        "mean_trade_alpha_weighted_proxy": alpha_trade_weighted_proxy,
    }


def main() -> None:
    out_root = Path("trading_research/examples/_output/30_baseline_rel2_simple_improvements")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(list(UNIVERSE_170), period=cfg.period, interval=cfg.interval)
    spy_df = vendor.fetch_bars(["SPY"], period=cfg.period, interval=cfg.interval)[["timestamp", "close"]]
    if bars.empty or spy_df.empty:
        raise ValueError("Missing Yahoo data for universe or SPY benchmark.")

    panel = _build_panel(bars, spy_df, cfg)
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

        spy_vol_threshold = float(train["spy_vol_21"].median())
        for spec in STRATEGIES:
            trades = _build_trades(
                test_panel=test,
                cfg=cfg,
                strategy=spec,
                fold_id=fold_id,
                spy_vol_threshold=spy_vol_threshold,
            )
            daily, trades_eval = _simulate_fold(
                test_panel=test,
                trades=trades,
                spy_df=spy_df,
                cfg=cfg,
                fold_id=fold_id,
                strategy=spec,
                spy_vol_anchor=spy_vol_threshold,
            )
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

    overall_df = pd.DataFrame(overall_rows).sort_values(
        ["annualized_sharpe_alpha", "cagr"],
        ascending=[False, False],
    )
    yearly_df = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
    daily_oos = pd.concat(daily_all_list, ignore_index=True) if daily_all_list else pd.DataFrame()
    trades_oos = pd.concat(trades_all_list, ignore_index=True) if trades_all_list else pd.DataFrame()

    baseline = overall_df[overall_df["strategy"] == "baseline_rel2"]
    best = overall_df.iloc[0].to_dict() if not overall_df.empty else {}
    uplift = {}
    if not baseline.empty and best:
        b = baseline.iloc[0]
        uplift = {
            "best_strategy": str(best["strategy"]),
            "delta_cagr_vs_baseline": float(best["cagr"] - b["cagr"]),
            "delta_alpha_sharpe_vs_baseline": float(best["annualized_sharpe_alpha"] - b["annualized_sharpe_alpha"]),
            "delta_net_sharpe_vs_baseline": float(best["annualized_sharpe_net"] - b["annualized_sharpe_net"]),
            "delta_max_dd_vs_baseline": float(best["max_drawdown"] - b["max_drawdown"]),
        }

    summary = {
        "universe_size_requested": len(UNIVERSE_170),
        "universe_size_fetched": int(panel["asset"].nunique()) if not panel.empty else 0,
        "history_start": str(panel["timestamp"].min()) if not panel.empty else None,
        "history_end": str(panel["timestamp"].max()) if not panel.empty else None,
        "n_walkforward_folds": int(fold_df["fold_id"].nunique()) if not fold_df.empty else 0,
        "strategies": [s.name for s in STRATEGIES],
        "config": {
            "period": cfg.period,
            "interval": cfg.interval,
            "signal_lookback_days": cfg.signal_lookback_days,
            "rolling_vwap_window": cfg.rolling_vwap_window,
            "rel_strength_threshold": cfg.rel_strength_threshold,
            "hold_days": cfg.hold_days,
            "min_train_years": cfg.min_train_years,
            "one_way_cost_return": _one_way_cost_return(cfg),
            "spy_vol_scale_min": cfg.spy_vol_scale_min,
            "spy_vol_scale_max": cfg.spy_vol_scale_max,
        },
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
