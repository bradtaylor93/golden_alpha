"""Next low-hanging-fruit experiments on top of current best strategy.

Baseline stack (from prior winners):
- rel2_top8 entry
- trend exit: bull hold 98d, bear hold 42d, bear trailing 10%
- confidence sizing
- cluster-proxy exposure caps

New low-hanging-fruit variants (one-at-a-time + combined):
1) liquidity filter (ADV rank gate),
2) crowding cap (max new entries/day),
3) volatility-normalized bear trailing stop,
4) regime-aware edge/cost buffer,
5) re-entry cooldown.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data.vendors.yahoo import YahooMarketDataVendor

# Reuse same 170 universe as prior studies.
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
    vol_window_days: int = 21
    rel_strength_threshold: float = 0.02
    top_quantile: float = 0.92
    min_train_years: int = 3
    gross_target: float = 1.0
    max_abs_weight_per_asset: float = 0.04
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    vol_floor: float = 1e-4
    # Base strategy parameters.
    bull_hold_days: int = 98
    bear_hold_days: int = 42
    bear_trailing_stop_pct: float = 0.10
    # Cluster-cap baseline.
    cluster_corr_threshold: float = 0.75
    cluster_abs_weight_cap: float = 0.25
    # New low-hanging fruits.
    liquidity_min_rank_pct: float = 0.30
    crowding_max_new_entries_per_day: int = 25
    quality_cap_max_new_entries_per_day: int = 20
    vol_trail_mult: float = 3.0
    vol_trail_min: float = 0.06
    vol_trail_max: float = 0.16
    regime_buffer_base_edge: float = 0.015
    regime_buffer_additional: float = 0.010
    reentry_cooldown_days: int = 10
    reentry_cooldown_good_days: int = 5
    reentry_cooldown_bad_days: int = 15
    turnover_smoothing_lambda: float = 0.20
    portfolio_vol_target_annual: float = 0.18
    portfolio_vol_lookback_days: int = 63
    portfolio_vol_scale_min: float = 0.70
    portfolio_vol_scale_max: float = 1.30
    # Sharpe-first overlays.
    beta_window_days: int = 63
    beta_cap: float = 1.0
    beta_activate_threshold: float = 0.10
    hedge_cost_bps_per_side: float = 2.0
    stressed_gross_scale: float = 0.65
    calm_gross_scale: float = 1.05


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    use_liquidity_filter: bool = False
    use_crowding_cap: bool = False
    use_vol_norm_trail: bool = False
    use_regime_buffer: bool = False
    use_reentry_cooldown: bool = False
    use_turnover_smoothing: bool = False
    use_cluster_caps: bool = True
    use_adaptive_cooldown: bool = False
    use_dynamic_edge_floor: bool = False
    use_quality_priority_cap: bool = False
    use_portfolio_vol_target: bool = False
    use_side_aware_cooldown: bool = False
    use_beta_hedge: bool = False
    use_dynamic_gross_target: bool = False
    require_spy_up_for_longs: bool = False
    require_spy_down_for_shorts: bool = False
    top_quantile_override: float | None = None
    rel_strength_threshold_override: float | None = None
    cooldown_days_override: int | None = None


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    # Prior best baseline.
    ExperimentSpec(name="base_lhf5_cluster_caps"),
    ExperimentSpec(name="nhf5_reentry_cooldown", use_reentry_cooldown=True),
    # Sharpe-first candidates: no cash-yield assumptions, only signal/risk controls.
    ExperimentSpec(
        name="sf1_top5_rel3_cooldown",
        use_reentry_cooldown=True,
        top_quantile_override=0.95,
        rel_strength_threshold_override=0.03,
    ),
    ExperimentSpec(
        name="sf2_top5_rel3_regime_side",
        use_reentry_cooldown=True,
        use_regime_buffer=True,
        require_spy_up_for_longs=True,
        require_spy_down_for_shorts=True,
        top_quantile_override=0.95,
        rel_strength_threshold_override=0.03,
    ),
    ExperimentSpec(
        name="sf3_top5_rel3_regime_voltarget",
        use_reentry_cooldown=True,
        use_regime_buffer=True,
        use_portfolio_vol_target=True,
        require_spy_up_for_longs=True,
        require_spy_down_for_shorts=True,
        top_quantile_override=0.95,
        rel_strength_threshold_override=0.03,
    ),
    ExperimentSpec(
        name="sf4_top5_rel3_regime_voltarget_beta",
        use_reentry_cooldown=True,
        use_regime_buffer=True,
        use_portfolio_vol_target=True,
        use_beta_hedge=True,
        use_turnover_smoothing=True,
        use_dynamic_gross_target=True,
        require_spy_up_for_longs=True,
        require_spy_down_for_shorts=True,
        top_quantile_override=0.95,
        rel_strength_threshold_override=0.03,
    ),
    ExperimentSpec(
        name="sf5_top3_rel4_ultra_defensive",
        use_reentry_cooldown=True,
        use_regime_buffer=True,
        use_portfolio_vol_target=True,
        use_beta_hedge=True,
        use_turnover_smoothing=True,
        use_dynamic_gross_target=True,
        require_spy_up_for_longs=True,
        require_spy_down_for_shorts=True,
        top_quantile_override=0.97,
        rel_strength_threshold_override=0.04,
    ),
)


def _one_way_cost_return(cfg: Config) -> float:
    return (cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side) / 10_000.0


def _hedge_one_way_cost_return(cfg: Config) -> float:
    return cfg.hedge_cost_bps_per_side / 10_000.0


def _build_panel(bars: pd.DataFrame, spy: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "asset", "close"]).sort_values(["asset", "timestamp"])
    frame["asset"] = frame["asset"].astype(str)
    for c in ["open", "high", "low", "close", "volume"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
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
    frame["dollar_vol"] = frame["close"] * frame["volume"].fillna(0.0)
    frame["dollar_vol_21"] = (
        g["dollar_vol"]
        .rolling(cfg.vol_window_days, min_periods=cfg.vol_window_days)
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame["adv_rank_pct"] = frame.groupby("timestamp")["dollar_vol_21"].rank(pct=True, method="average")

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
    spy["spy_vol_21"] = spy["close"].pct_change().rolling(cfg.vol_window_days, min_periods=cfg.vol_window_days).std()
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
            "vol_21",
            "adv_rank_pct",
            "spy_close",
            "spy_up",
            "spy_vol_21",
            "rel_strength_63",
        ]
    )
    frame["year"] = frame["timestamp"].dt.year
    return frame.reset_index(drop=True)


def _confidence_from_row(row: pd.Series, cfg: Config) -> float:
    rank = float(row["rank_pct"])
    rel = float(row["rel_strength_63"])
    rank_conf = float(np.clip((rank - cfg.top_quantile) / max(1e-6, 1.0 - cfg.top_quantile), 0.0, 1.0))
    rel_conf = float(np.clip((rel - cfg.rel_strength_threshold) / 0.05, 0.0, 1.0))
    return float(0.5 + 1.5 * (0.5 * rank_conf + 0.5 * rel_conf))


def _build_trades(
    test_panel: pd.DataFrame,
    cfg: Config,
    exp: ExperimentSpec,
    fold_id: str,
    train_spy_vol_median: float,
) -> pd.DataFrame:
    close_wide = test_panel.pivot(index="timestamp", columns="asset", values="close").sort_index().ffill()
    all_dates = close_wide.index.to_list()
    date_to_idx = {ts: i for i, ts in enumerate(all_dates)}
    n_dates = len(all_dates)
    if n_dates < 3:
        return pd.DataFrame()

    rows: list[dict[str, float | str]] = []
    last_end_by_asset: dict[str, int] = {}
    cooldown_until_by_asset: dict[str, int] = {}
    by_ts = {ts: g.copy() for ts, g in test_panel.groupby("timestamp", sort=True)}
    top_q = float(exp.top_quantile_override) if exp.top_quantile_override is not None else cfg.top_quantile
    rel_thresh = (
        float(exp.rel_strength_threshold_override)
        if exp.rel_strength_threshold_override is not None
        else cfg.rel_strength_threshold
    )

    for ts in all_dates:
        sig_idx = date_to_idx[ts]
        daily = by_ts.get(ts)
        if daily is None or daily.empty:
            continue

        # Pre-filter and compute priority/confidence at this date.
        cands: list[dict[str, object]] = []
        for _, r in daily.iterrows():
            asset = str(r["asset"])
            rank = float(r["rank_pct"])
            rel = float(r["rel_strength_63"])
            if rank < top_q or rel < rel_thresh:
                continue
            if exp.use_liquidity_filter and float(r["adv_rank_pct"]) < cfg.liquidity_min_rank_pct:
                continue
            trend_sign = float(np.sign(float(r["past_return"])))
            if not np.isfinite(trend_sign) or trend_sign == 0.0:
                continue
            spy_up_now = float(r["spy_up"]) > 0.5
            if exp.require_spy_up_for_longs and trend_sign > 0 and not spy_up_now:
                continue
            if exp.require_spy_down_for_shorts and trend_sign < 0 and spy_up_now:
                continue
            if sig_idx <= int(last_end_by_asset.get(asset, -1)):
                continue
            if exp.use_reentry_cooldown and sig_idx <= int(cooldown_until_by_asset.get(asset, -1)):
                continue
            rank_conf = float(np.clip((rank - top_q) / max(1e-6, 1.0 - top_q), 0.0, 1.0))
            rel_conf = float(np.clip((rel - rel_thresh) / 0.05, 0.0, 1.0))
            conf = float(0.5 + 1.5 * (0.5 * rank_conf + 0.5 * rel_conf))
            strength = float(max(1e-6, rank - top_q))
            edge_proxy = strength * conf
            if exp.use_regime_buffer or exp.use_dynamic_edge_floor:
                spy_vol = float(r["spy_vol_21"])
                stressed = (not spy_up_now) or (
                    np.isfinite(spy_vol) and np.isfinite(train_spy_vol_median) and spy_vol > train_spy_vol_median
                )
                req = cfg.regime_buffer_base_edge + (cfg.regime_buffer_additional if stressed else 0.0)
                if edge_proxy < req:
                    continue
            cands.append(
                {
                    "asset": asset,
                    "row": r,
                    "trend_sign": trend_sign,
                    "confidence": conf,
                    "strength": strength,
                    "priority": edge_proxy,
                }
            )

        if not cands:
            continue
        cands.sort(key=lambda x: float(x["priority"]), reverse=True)
        if exp.use_quality_priority_cap:
            cands = cands[: cfg.quality_cap_max_new_entries_per_day]
        elif exp.use_crowding_cap:
            cands = cands[: cfg.crowding_max_new_entries_per_day]

        for c in cands:
            asset = str(c["asset"])
            r = c["row"]
            trend_sign = float(c["trend_sign"])
            start_idx = sig_idx + 1
            spy_up = float(r["spy_up"]) > 0.5
            hold_days = cfg.bull_hold_days if spy_up else cfg.bear_hold_days
            hard_end = min(sig_idx + hold_days, n_dates - 1)
            if start_idx >= n_dates or hard_end <= start_idx:
                continue

            entry_ts = all_dates[start_idx]
            entry_px = float(close_wide.loc[entry_ts, asset])
            if not np.isfinite(entry_px) or entry_px <= 0.0:
                continue

            exit_idx = hard_end
            exit_reason = "fixed_hold"
            peak_signed = 0.0
            bear_trail = cfg.bear_trailing_stop_pct
            if exp.use_vol_norm_trail and not spy_up:
                ent_vol = float(r["vol_21"])
                if np.isfinite(ent_vol):
                    bear_trail = float(np.clip(cfg.vol_trail_mult * ent_vol, cfg.vol_trail_min, cfg.vol_trail_max))

            for j in range(start_idx, hard_end + 1):
                ts_j = all_dates[j]
                px = float(close_wide.loc[ts_j, asset])
                if not np.isfinite(px) or px <= 0.0:
                    continue
                signed_ret = trend_sign * (px / entry_px - 1.0)
                peak_signed = max(peak_signed, signed_ret)
                if (not spy_up) and bear_trail is not None and (peak_signed - signed_ret) >= float(bear_trail):
                    exit_idx = j
                    exit_reason = "bear_trailing_dynamic" if exp.use_vol_norm_trail else "bear_trailing_10"
                    break

            rows.append(
                {
                    "fold_id": fold_id,
                    "asset": asset,
                    "signal_timestamp": str(ts),
                    "start_idx": float(start_idx),
                    "end_idx": float(exit_idx),
                    "signal_sign": trend_sign,
                    "signal_strength": float(c["strength"]),
                    "entry_vol_21": float(max(cfg.vol_floor, float(r["vol_21"]))),
                    "confidence_score": float(c["confidence"]),
                    "hold_days": float(exit_idx - sig_idx),
                    "exit_reason": exit_reason,
                }
            )
            last_end_by_asset[asset] = exit_idx
            if exp.use_reentry_cooldown:
                if exp.use_side_aware_cooldown:
                    pnl_sign = 1.0
                    if entry_px > 0:
                        exit_px = float(close_wide.loc[all_dates[exit_idx], asset])
                        pnl_sign = trend_sign * (exit_px / entry_px - 1.0)
                    cd_days = cfg.reentry_cooldown_good_days if pnl_sign > 0 else cfg.reentry_cooldown_bad_days
                elif exp.use_adaptive_cooldown:
                    spy_vol = float(r["spy_vol_21"])
                    if np.isfinite(spy_vol) and np.isfinite(train_spy_vol_median) and spy_vol > train_spy_vol_median:
                        cd_days = cfg.reentry_cooldown_bad_days
                    else:
                        cd_days = cfg.reentry_cooldown_good_days
                else:
                    cd_days = (
                        int(exp.cooldown_days_override)
                        if exp.cooldown_days_override is not None
                        else cfg.reentry_cooldown_days
                    )
                cooldown_until_by_asset[asset] = exit_idx + cd_days
    return pd.DataFrame(rows)


def _build_corr_clusters(train_panel: pd.DataFrame, corr_threshold: float) -> dict[str, int]:
    close = train_panel.pivot(index="timestamp", columns="asset", values="close").sort_index().ffill()
    rets = close.pct_change().dropna(how="all")
    if rets.empty or rets.shape[1] <= 1:
        return {str(a): i for i, a in enumerate(rets.columns.tolist())}
    corr = rets.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    assets = corr.columns.tolist()
    n = len(assets)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    vals = corr.to_numpy(dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(vals[i, j]) >= corr_threshold:
                union(i, j)
    roots = [find(i) for i in range(n)]
    uniq = {r: k for k, r in enumerate(sorted(set(roots)))}
    return {str(assets[i]): int(uniq[roots[i]]) for i in range(n)}


def _apply_cluster_caps(weights: pd.Series, cluster_map: dict[str, int], cap_abs: float) -> pd.Series:
    out = weights.copy()
    if out.empty:
        return out
    by_cluster: dict[int, list[str]] = {}
    for a in out.index:
        c = int(cluster_map.get(str(a), -1))
        by_cluster.setdefault(c, []).append(str(a))
    for members in by_cluster.values():
        cluster_abs = float(np.sum(np.abs(out.reindex(members).fillna(0.0).to_numpy(dtype=float))))
        if cluster_abs > cap_abs and cluster_abs > 1e-12:
            out.loc[members] = out.loc[members] * (cap_abs / cluster_abs)
    return out


def _weights_from_active(
    active: pd.DataFrame,
    cfg: Config,
    assets: list[str],
    cluster_map: dict[str, int] | None,
    use_cluster_caps: bool,
) -> pd.Series:
    w = pd.Series(0.0, index=assets, dtype=float)
    if active.empty:
        return w
    a = active.copy()
    for c in ["signal_strength", "signal_sign", "entry_vol_21", "confidence_score"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a["entry_vol_21"] = a["entry_vol_21"].clip(lower=cfg.vol_floor)
    a["confidence_score"] = a["confidence_score"].clip(lower=0.25, upper=3.0)
    a = a.dropna(subset=["signal_strength", "signal_sign", "entry_vol_21", "confidence_score"])
    if a.empty:
        return w
    a["raw"] = a["signal_strength"] * a["confidence_score"]
    by_asset = a.groupby("asset", as_index=False).agg(
        raw=("raw", "sum"),
        sign=("signal_sign", lambda s: float(np.sign(np.sum(s)))),
    )
    by_asset["raw_signed"] = by_asset["raw"] * by_asset["sign"]
    denom = float(np.sum(np.abs(by_asset["raw_signed"])))
    if denom <= 0.0:
        return w
    by_asset["weight"] = (by_asset["raw_signed"] / denom) * cfg.gross_target
    by_asset["weight"] = by_asset["weight"].clip(-cfg.max_abs_weight_per_asset, cfg.max_abs_weight_per_asset)
    for _, r in by_asset.iterrows():
        name = str(r["asset"])
        if name in w.index:
            w.loc[name] = float(r["weight"])
    if use_cluster_caps and cluster_map is not None:
        w = _apply_cluster_caps(w, cluster_map=cluster_map, cap_abs=cfg.cluster_abs_weight_cap)
    return w


def _simulate_fold(
    test_panel: pd.DataFrame,
    trades: pd.DataFrame,
    spy_df: pd.DataFrame,
    cfg: Config,
    fold_id: str,
    strategy_name: str,
    exp: ExperimentSpec,
    train_spy_vol_median: float,
    cluster_map: dict[str, int] | None,
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
    spy["close"] = pd.to_numeric(spy["close"], errors="coerce")
    spy = spy.dropna(subset=["close"])
    spy = spy.set_index("timestamp")["close"].reindex(rets.index).ffill()
    spy_ret = spy.pct_change().fillna(0.0)
    spy_ma200 = spy.rolling(200, min_periods=200).mean()
    spy_up = (spy > spy_ma200).astype(float).fillna(0.0)
    spy_vol21 = spy.pct_change().rolling(cfg.vol_window_days, min_periods=cfg.vol_window_days).std()

    t = trades.copy()
    if t.empty:
        out = pd.DataFrame(
            {"timestamp": rets.index, "net_return": 0.0, "spy_return": spy_ret.values, "alpha_return": -spy_ret.values}
        )
        out["turnover"] = 0.0
        out["gross_exposure"] = 0.0
        out["hedge_beta"] = 0.0
    else:
        t["start_idx"] = pd.to_numeric(t["start_idx"], errors="coerce").astype("Int64")
        t["end_idx"] = pd.to_numeric(t["end_idx"], errors="coerce").astype("Int64")
        t = t.dropna(subset=["start_idx", "end_idx", "asset", "signal_sign", "signal_strength", "entry_vol_21", "confidence_score"])
        prev_w = pd.Series(0.0, index=assets, dtype=float)
        one_way = _one_way_cost_return(cfg)
        hedge_one_way = _hedge_one_way_cost_return(cfg)
        prev_hedge = 0.0
        rows: list[dict[str, float | str]] = []
        hist_net: list[float] = []
        hist_strat: list[float] = []
        hist_spy: list[float] = []
        for i in range(1, len(dates)):
            active = t[(t["start_idx"] <= i) & (t["end_idx"] >= i)]
            target_w = _weights_from_active(
                active=active,
                cfg=cfg,
                assets=assets,
                cluster_map=cluster_map,
                use_cluster_caps=exp.use_cluster_caps,
            )
            if exp.use_turnover_smoothing:
                w = (1.0 - cfg.turnover_smoothing_lambda) * target_w + cfg.turnover_smoothing_lambda * prev_w
            else:
                w = target_w

            if exp.use_dynamic_gross_target:
                curr_spy_up = bool(float(spy_up.iloc[i]) > 0.5) if np.isfinite(spy_up.iloc[i]) else False
                curr_spy_vol = float(spy_vol21.iloc[i]) if np.isfinite(spy_vol21.iloc[i]) else float("nan")
                stressed = (not curr_spy_up) or (
                    np.isfinite(curr_spy_vol)
                    and np.isfinite(train_spy_vol_median)
                    and curr_spy_vol > train_spy_vol_median
                )
                gross_scale = cfg.stressed_gross_scale if stressed else cfg.calm_gross_scale
                w = w * float(gross_scale)
                w = w.clip(-cfg.max_abs_weight_per_asset, cfg.max_abs_weight_per_asset)

            if exp.use_portfolio_vol_target:
                if len(hist_net) >= cfg.portfolio_vol_lookback_days:
                    rv = float(np.std(hist_net[-cfg.portfolio_vol_lookback_days :], ddof=1)) * math.sqrt(252.0)
                    if np.isfinite(rv) and rv > 1e-9:
                        scale = float(
                            np.clip(
                                cfg.portfolio_vol_target_annual / rv,
                                cfg.portfolio_vol_scale_min,
                                cfg.portfolio_vol_scale_max,
                            )
                        )
                        w = w * scale
                # Re-apply per-asset cap after scaling.
                w = w.clip(-cfg.max_abs_weight_per_asset, cfg.max_abs_weight_per_asset)

            gross = float(np.dot(w.values, rets.iloc[i].reindex(assets).fillna(0.0).values))
            turnover = float(np.abs(w - prev_w).sum())
            cost = turnover * one_way
            strat_net_pre = gross - cost
            spy_r = float(spy_ret.iloc[i]) if np.isfinite(spy_ret.iloc[i]) else 0.0

            hedge_beta = 0.0
            hedge_turnover = 0.0
            hedge_cost = 0.0
            if exp.use_beta_hedge:
                if len(hist_strat) >= cfg.beta_window_days and len(hist_spy) >= cfg.beta_window_days:
                    y = np.asarray(hist_strat[-cfg.beta_window_days :], dtype=float)
                    x = np.asarray(hist_spy[-cfg.beta_window_days :], dtype=float)
                    vx = float(np.var(x, ddof=1))
                    if np.isfinite(vx) and vx > 1e-12:
                        beta_est = float(np.cov(y, x, ddof=1)[0, 1] / vx)
                        if np.isfinite(beta_est) and beta_est >= cfg.beta_activate_threshold:
                            hedge_beta = float(np.clip(beta_est, 0.0, cfg.beta_cap))
                hedge_turnover = abs(hedge_beta - prev_hedge)
                hedge_cost = hedge_turnover * hedge_one_way
                prev_hedge = hedge_beta
            net = strat_net_pre - hedge_beta * spy_r - hedge_cost
            alpha = net - max(0.0, 1.0 - hedge_beta) * spy_r

            rows.append(
                {
                    "timestamp": str(dates[i]),
                    "net_return": net,
                    "spy_return": spy_r,
                    "alpha_return": alpha,
                    "turnover": float(turnover + hedge_turnover),
                    "gross_exposure": float(np.abs(w).sum() + abs(hedge_beta)),
                    "hedge_beta": hedge_beta,
                }
            )
            prev_w = w
            hist_net.append(float(net))
            hist_strat.append(float(strat_net_pre))
            hist_spy.append(float(spy_r))
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
                    "net_return": net_ret,
                    "benchmark_return": bench,
                    "alpha_return": net_ret - bench if np.isfinite(bench) else float("nan"),
                }
            )
    return out, pd.DataFrame(trade_rows)


def _simulate_buy_hold_all_fold(test_panel: pd.DataFrame, spy_df: pd.DataFrame, cfg: Config, fold_id: str) -> pd.DataFrame:
    close = test_panel.pivot(index="timestamp", columns="asset", values="close").sort_index().ffill()
    rets = close.pct_change().fillna(0.0)
    if rets.empty:
        return pd.DataFrame()
    n_assets = rets.shape[1]
    if n_assets == 0:
        return pd.DataFrame()
    one_way = _one_way_cost_return(cfg)
    w = np.repeat(1.0 / n_assets, n_assets)
    gross = rets.to_numpy() @ w
    turnover = np.zeros_like(gross)
    if len(gross) > 0:
        turnover[0] = float(np.sum(np.abs(w)))
    net = gross - turnover * one_way
    out = pd.DataFrame(
        {"timestamp": rets.index, "net_return": net, "turnover": turnover, "gross_exposure": 1.0, "hedge_beta": 0.0}
    )
    spy = spy_df.copy()
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True, errors="coerce")
    spy = spy.dropna(subset=["timestamp"]).sort_values("timestamp")
    spy = spy.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(out["timestamp"]), method="ffill")
    spy_ret = spy.pct_change().fillna(0.0).to_numpy()
    out["spy_return"] = spy_ret
    out["alpha_return"] = pd.to_numeric(out["net_return"], errors="coerce") - pd.Series(spy_ret, index=out.index)
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    out["running_max"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["running_max"] - 1.0
    out["fold_id"] = fold_id
    out["strategy"] = "buy_hold_all"
    return out.reset_index(drop=True)


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
            "avg_hedge_beta": float("nan"),
        }
    net = pd.to_numeric(daily["net_return"], errors="coerce").fillna(0.0)
    ann = float(np.exp(np.log1p(net).mean() * 252.0) - 1.0)
    start, end = daily["timestamp"].iloc[0], daily["timestamp"].iloc[-1]
    years = max(1e-9, (end - start).total_seconds() / (365.25 * 24 * 3600))
    # Rebuild stitched OOS equity from daily returns so metrics are robust
    # even when per-fold equity paths are concatenated.
    equity = (1.0 + net).cumprod()
    eq_end = float(equity.iloc[-1])
    cagr = float(eq_end ** (1.0 / years) - 1.0) if eq_end > 0 else float("nan")
    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1.0).min())
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
        "avg_hedge_beta": float(pd.to_numeric(daily.get("hedge_beta"), errors="coerce").mean()),
    }


def main() -> None:
    out_root = Path("trading_research/examples/_output/37_rel2_top8_next_low_hanging")
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
        raise ValueError("Insufficient years for anchored walk-forward.")

    strategy_names = [e.name for e in EXPERIMENTS] + ["buy_hold_all"]
    all_daily_by_strategy: dict[str, list[pd.DataFrame]] = {n: [] for n in strategy_names}
    all_trades_by_strategy: dict[str, list[pd.DataFrame]] = {n: [] for n in strategy_names}
    fold_rows: list[dict[str, float | str]] = []

    for test_year in years[cfg.min_train_years :]:
        fold_id = f"fold_{int(test_year)}"
        train = panel[panel["year"] < test_year].copy()
        test = panel[panel["year"] == test_year].copy()
        if train.empty or test.empty:
            continue
        train_spy_vol_median = float(train["spy_vol_21"].median())
        cluster_map = _build_corr_clusters(train, corr_threshold=cfg.cluster_corr_threshold)

        for exp in EXPERIMENTS:
            trades = _build_trades(
                test_panel=test,
                cfg=cfg,
                exp=exp,
                fold_id=fold_id,
                train_spy_vol_median=train_spy_vol_median,
            )
            daily, trades_eval = _simulate_fold(
                test_panel=test,
                trades=trades,
                spy_df=spy_df,
                cfg=cfg,
                fold_id=fold_id,
                strategy_name=exp.name,
                exp=exp,
                train_spy_vol_median=train_spy_vol_median,
                cluster_map=cluster_map,
            )
            if daily.empty:
                continue
            all_daily_by_strategy[exp.name].append(daily)
            all_trades_by_strategy[exp.name].append(trades_eval)
            fold_rows.append(
                {
                    "strategy": exp.name,
                    "fold_id": fold_id,
                    "test_year": int(test_year),
                    "train_years": int(train["year"].nunique()),
                    "assets_in_test": int(test["asset"].nunique()),
                    **_portfolio_metrics(daily, trades_eval),
                }
            )

        bh_daily = _simulate_buy_hold_all_fold(test_panel=test, spy_df=spy_df, cfg=cfg, fold_id=fold_id)
        if not bh_daily.empty:
            all_daily_by_strategy["buy_hold_all"].append(bh_daily)
            all_trades_by_strategy["buy_hold_all"].append(pd.DataFrame())
            fold_rows.append(
                {
                    "strategy": "buy_hold_all",
                    "fold_id": fold_id,
                    "test_year": int(test_year),
                    "train_years": int(train["year"].nunique()),
                    "assets_in_test": int(test["asset"].nunique()),
                    **_portfolio_metrics(bh_daily, pd.DataFrame()),
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

    base = overall_df[overall_df["strategy"] == "base_lhf5_cluster_caps"]
    best = overall_df.iloc[0].to_dict() if not overall_df.empty else {}
    uplift = {}
    if not base.empty and best:
        b = base.iloc[0]
        uplift = {
            "best_strategy": str(best["strategy"]),
            "delta_annual_return_vs_base": float(best["annual_return"] - b["annual_return"]),
            "delta_cagr_vs_base": float(best["cagr"] - b["cagr"]),
            "delta_net_sharpe_vs_base": float(best["annualized_sharpe_net"] - b["annualized_sharpe_net"]),
            "delta_alpha_sharpe_vs_base": float(best["annualized_sharpe_alpha"] - b["annualized_sharpe_alpha"]),
            "delta_maxdd_vs_base": float(best["max_drawdown"] - b["max_drawdown"]),
            "delta_n_trades_vs_base": float(best["n_trades"] - b["n_trades"]),
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
            "top_quantile": cfg.top_quantile,
            "one_way_cost_return": _one_way_cost_return(cfg),
        },
        "experiments": [e.__dict__ for e in EXPERIMENTS],
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

