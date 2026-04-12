"""Multi-sleeve alpha study: rel2 + vol compression breakout + EMA crossover.

Adds two new sleeves and combines sleeves with an adaptive allocator:
- Sleeve A: rel2 ranked opportunity (existing baseline family).
- Sleeve B: volatility compression breakout.
- Sleeve C: EMA crossover trend sleeve.

Combination is "intelligent" in the sense that sleeve weights adapt daily using:
1) trailing sleeve Sharpe proxy,
2) inverse-vol preference,
3) correlation penalty across sleeves,
4) allocation smoothing and explicit allocation turnover cost.
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

SLEEVE_REL2 = "sleeve_rel2"
SLEEVE_VOL_COMP = "sleeve_vol_compression_breakout"
SLEEVE_EMA = "sleeve_ema_crossover"
SLEEVE_COMBO = "combo_dynamic_allocator"


@dataclass(frozen=True)
class Config:
    period: str = "10y"
    interval: str = "1d"
    signal_lookback_days: int = 63
    rolling_vwap_window: int = 20
    vol_window_days: int = 21
    min_train_years: int = 3
    gross_target: float = 1.0
    max_abs_weight_per_asset: float = 0.04
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    vol_floor: float = 1e-4
    # rel2 sleeve
    rel2_top_quantile: float = 0.92
    rel2_strength_threshold: float = 0.02
    rel2_bull_hold_days: int = 98
    rel2_bear_hold_days: int = 42
    rel2_bear_trailing_stop_pct: float = 0.10
    rel2_reentry_cooldown_days: int = 10
    # volatility compression sleeve
    bb_window_days: int = 20
    bb_k: float = 2.0
    compression_quantile: float = 0.20
    breakout_lookback_days: int = 20
    comp_hold_days: int = 42
    comp_trailing_stop_pct: float = 0.10
    comp_reentry_cooldown_days: int = 5
    comp_allow_shorts: bool = False
    comp_require_spy_up: bool = True
    comp_min_rel_strength: float = 0.0
    # EMA sleeve
    ema_fast_span: int = 20
    ema_slow_span: int = 100
    ema_min_gap: float = 0.003
    ema_max_hold_days: int = 84
    ema_reentry_cooldown_days: int = 10
    ema_allow_shorts: bool = False
    ema_require_spy_up: bool = True
    # Combination allocator
    combo_lookback_days: int = 63
    combo_min_history_days: int = 40
    combo_corr_penalty: float = 0.75
    combo_alloc_smoothing_lambda: float = 0.50
    combo_alloc_cost_bps_per_side: float = 1.0
    combo_rebalance_days: int = 5
    combo_disable_sharpe_threshold: float = 0.10
    combo_core_min_weight: float = 0.60


def _one_way_cost_return(cfg: Config) -> float:
    return (cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side) / 10_000.0


def _combo_alloc_cost_return(cfg: Config) -> float:
    return cfg.combo_alloc_cost_bps_per_side / 10_000.0


def _build_panel(bars: pd.DataFrame, spy_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "asset", "close"]).sort_values(["asset", "timestamp"])
    frame["asset"] = frame["asset"].astype(str)
    for c in ["open", "high", "low", "close", "volume"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame.dropna(subset=["close", "high", "low"])
    frame = frame[frame["close"] > 0].copy()

    g_close = frame.groupby("asset", sort=False)["close"]
    look = cfg.signal_lookback_days

    # rel2 baseline features
    typ = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    frame["pv"] = typ * frame["volume"].fillna(0.0)
    frame["pv_roll"] = frame.groupby("asset", sort=False)["pv"].transform(
        lambda s: s.rolling(cfg.rolling_vwap_window, min_periods=cfg.rolling_vwap_window).sum()
    )
    frame["v_roll"] = frame.groupby("asset", sort=False)["volume"].transform(
        lambda s: s.fillna(0.0).rolling(cfg.rolling_vwap_window, min_periods=cfg.rolling_vwap_window).sum()
    )
    frame["rolling_vwap"] = frame["pv_roll"] / frame["v_roll"].replace(0.0, np.nan)
    frame["past_return"] = g_close.pct_change(look)
    frame["ret_1d"] = g_close.pct_change()
    frame["vol_21"] = frame.groupby("asset", sort=False)["ret_1d"].transform(
        lambda s: s.rolling(cfg.vol_window_days, min_periods=cfg.vol_window_days).std()
    )
    frame["score"] = np.sign(frame["past_return"]) * (frame["close"] / frame["rolling_vwap"] - 1.0)
    frame["rank_pct"] = frame.groupby("timestamp")["score"].rank(pct=True, method="average")

    # Vol-compression features
    frame["bb_mid"] = g_close.transform(lambda s: s.rolling(cfg.bb_window_days, min_periods=cfg.bb_window_days).mean())
    frame["bb_std"] = g_close.transform(lambda s: s.rolling(cfg.bb_window_days, min_periods=cfg.bb_window_days).std())
    frame["bb_width"] = (2.0 * cfg.bb_k * frame["bb_std"]) / frame["bb_mid"].replace(0.0, np.nan)
    frame["breakout_high"] = g_close.transform(
        lambda s: s.rolling(cfg.breakout_lookback_days, min_periods=cfg.breakout_lookback_days).max().shift(1)
    )
    frame["breakout_low"] = g_close.transform(
        lambda s: s.rolling(cfg.breakout_lookback_days, min_periods=cfg.breakout_lookback_days).min().shift(1)
    )

    # EMA sleeve features
    frame["ema_fast"] = g_close.transform(lambda s: s.ewm(span=cfg.ema_fast_span, adjust=False).mean())
    frame["ema_slow"] = g_close.transform(lambda s: s.ewm(span=cfg.ema_slow_span, adjust=False).mean())
    frame["ema_gap"] = frame["ema_fast"] / frame["ema_slow"].replace(0.0, np.nan) - 1.0
    frame["ema_gap_prev"] = frame.groupby("asset", sort=False)["ema_gap"].shift(1)
    frame["ema_cross_up"] = (
        (frame["ema_gap_prev"] <= 0.0) & (frame["ema_gap"] > 0.0) & (frame["ema_gap"].abs() >= cfg.ema_min_gap)
    ).astype(float)
    frame["ema_cross_down"] = (
        (frame["ema_gap_prev"] >= 0.0) & (frame["ema_gap"] < 0.0) & (frame["ema_gap"].abs() >= cfg.ema_min_gap)
    ).astype(float)

    spy = spy_df.copy()
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True, errors="coerce")
    spy = spy.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    spy["close"] = pd.to_numeric(spy["close"], errors="coerce")
    spy = spy.dropna(subset=["close"]).copy()
    spy["spy_ret_63"] = spy["close"].pct_change(look)
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
            "ret_1d",
            "vol_21",
            "score",
            "rank_pct",
            "bb_width",
            "breakout_high",
            "breakout_low",
            "ema_gap",
            "ema_cross_up",
            "ema_cross_down",
            "spy_up",
            "spy_vol_21",
            "rel_strength_63",
        ]
    )
    frame["year"] = frame["timestamp"].dt.year
    return frame.reset_index(drop=True)


def _confidence_rel2(rank_pct: float, rel_strength: float, cfg: Config) -> float:
    rank_conf = float(
        np.clip(
            (rank_pct - cfg.rel2_top_quantile) / max(1e-6, 1.0 - cfg.rel2_top_quantile),
            0.0,
            1.0,
        )
    )
    rel_conf = float(np.clip((rel_strength - cfg.rel2_strength_threshold) / 0.05, 0.0, 1.0))
    return float(0.5 + 1.5 * (0.5 * rank_conf + 0.5 * rel_conf))


def _build_rel2_trades(test_panel: pd.DataFrame, cfg: Config, fold_id: str) -> pd.DataFrame:
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

    for ts in all_dates:
        sig_idx = date_to_idx[ts]
        daily = by_ts.get(ts)
        if daily is None or daily.empty:
            continue
        cands: list[dict[str, object]] = []
        for _, r in daily.iterrows():
            asset = str(r["asset"])
            rank = float(r["rank_pct"])
            rel = float(r["rel_strength_63"])
            if rank < cfg.rel2_top_quantile or rel < cfg.rel2_strength_threshold:
                continue
            sign = float(np.sign(float(r["past_return"])))
            if not np.isfinite(sign) or sign == 0.0:
                continue
            if sig_idx <= int(last_end_by_asset.get(asset, -1)):
                continue
            if sig_idx <= int(cooldown_until_by_asset.get(asset, -1)):
                continue
            strength = float(max(1e-6, rank - cfg.rel2_top_quantile))
            conf = _confidence_rel2(rank_pct=rank, rel_strength=rel, cfg=cfg)
            cands.append(
                {
                    "asset": asset,
                    "row": r,
                    "sign": sign,
                    "strength": strength,
                    "confidence": conf,
                    "priority": strength * conf,
                }
            )
        if not cands:
            continue
        cands.sort(key=lambda x: float(x["priority"]), reverse=True)

        for c in cands:
            asset = str(c["asset"])
            r = c["row"]
            sign = float(c["sign"])
            start_idx = sig_idx + 1
            spy_up = bool(float(r["spy_up"]) > 0.5)
            hold_days = cfg.rel2_bull_hold_days if spy_up else cfg.rel2_bear_hold_days
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
            for j in range(start_idx, hard_end + 1):
                px = float(close_wide.loc[all_dates[j], asset])
                if not np.isfinite(px) or px <= 0.0:
                    continue
                signed_ret = sign * (px / entry_px - 1.0)
                peak_signed = max(peak_signed, signed_ret)
                if (not spy_up) and (peak_signed - signed_ret) >= cfg.rel2_bear_trailing_stop_pct:
                    exit_idx = j
                    exit_reason = "bear_trailing_stop"
                    break

            rows.append(
                {
                    "fold_id": fold_id,
                    "sleeve": SLEEVE_REL2,
                    "asset": asset,
                    "signal_timestamp": str(ts),
                    "start_idx": float(start_idx),
                    "end_idx": float(exit_idx),
                    "signal_sign": sign,
                    "signal_strength": float(c["strength"]),
                    "risk_scale": 1.0 / float(max(cfg.vol_floor, float(r["vol_21"]))),
                    "confidence_score": float(c["confidence"]),
                    "hold_days": float(exit_idx - sig_idx),
                    "exit_reason": exit_reason,
                }
            )
            last_end_by_asset[asset] = exit_idx
            cooldown_until_by_asset[asset] = exit_idx + cfg.rel2_reentry_cooldown_days
    return pd.DataFrame(rows)


def _build_vol_compression_trades(
    test_panel: pd.DataFrame,
    compression_thresholds: dict[str, float],
    cfg: Config,
    fold_id: str,
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

    for ts in all_dates:
        sig_idx = date_to_idx[ts]
        daily = by_ts.get(ts)
        if daily is None or daily.empty:
            continue
        cands: list[dict[str, object]] = []
        for _, r in daily.iterrows():
            asset = str(r["asset"])
            thr = float(compression_thresholds.get(asset, np.nan))
            if not np.isfinite(thr):
                continue
            bb_width = float(r["bb_width"])
            if not (np.isfinite(bb_width) and bb_width <= thr):
                continue
            close = float(r["close"])
            b_hi = float(r["breakout_high"])
            b_lo = float(r["breakout_low"])
            sign = 0.0
            if np.isfinite(close) and np.isfinite(b_hi) and close > b_hi:
                sign = 1.0
            elif cfg.comp_allow_shorts and np.isfinite(close) and np.isfinite(b_lo) and close < b_lo:
                sign = -1.0
            if sign == 0.0:
                continue
            spy_up = float(r["spy_up"]) > 0.5
            if cfg.comp_require_spy_up and sign > 0 and not spy_up:
                continue
            rel = float(r["rel_strength_63"])
            if sign > 0 and np.isfinite(rel) and rel < cfg.comp_min_rel_strength:
                continue
            if sig_idx <= int(last_end_by_asset.get(asset, -1)):
                continue
            if sig_idx <= int(cooldown_until_by_asset.get(asset, -1)):
                continue
            breakout_mag = abs(close / (b_hi if sign > 0 else b_lo) - 1.0) if (b_hi > 0 and b_lo > 0) else 0.0
            compression_mag = max(0.0, (thr - bb_width) / max(1e-6, thr))
            strength = float(max(1e-6, breakout_mag + compression_mag))
            conf = float(0.75 + min(1.0, 2.0 * strength))
            cands.append(
                {
                    "asset": asset,
                    "row": r,
                    "sign": sign,
                    "strength": strength,
                    "confidence": conf,
                    "priority": strength * conf,
                }
            )
        if not cands:
            continue
        cands.sort(key=lambda x: float(x["priority"]), reverse=True)
        for c in cands:
            asset = str(c["asset"])
            r = c["row"]
            sign = float(c["sign"])
            start_idx = sig_idx + 1
            hard_end = min(sig_idx + cfg.comp_hold_days, n_dates - 1)
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
                px = float(close_wide.loc[all_dates[j], asset])
                if not np.isfinite(px) or px <= 0.0:
                    continue
                signed_ret = sign * (px / entry_px - 1.0)
                peak_signed = max(peak_signed, signed_ret)
                if (peak_signed - signed_ret) >= cfg.comp_trailing_stop_pct:
                    exit_idx = j
                    exit_reason = "trailing_stop"
                    break

            rows.append(
                {
                    "fold_id": fold_id,
                    "sleeve": SLEEVE_VOL_COMP,
                    "asset": asset,
                    "signal_timestamp": str(ts),
                    "start_idx": float(start_idx),
                    "end_idx": float(exit_idx),
                    "signal_sign": sign,
                    "signal_strength": float(c["strength"]),
                    "risk_scale": 1.0 / float(max(cfg.vol_floor, float(r["vol_21"]))),
                    "confidence_score": float(c["confidence"]),
                    "hold_days": float(exit_idx - sig_idx),
                    "exit_reason": exit_reason,
                }
            )
            last_end_by_asset[asset] = exit_idx
            cooldown_until_by_asset[asset] = exit_idx + cfg.comp_reentry_cooldown_days
    return pd.DataFrame(rows)


def _build_ema_trades(test_panel: pd.DataFrame, cfg: Config, fold_id: str) -> pd.DataFrame:
    close_wide = test_panel.pivot(index="timestamp", columns="asset", values="close").sort_index().ffill()
    up_wide = test_panel.pivot(index="timestamp", columns="asset", values="ema_cross_up").sort_index().fillna(0.0)
    dn_wide = test_panel.pivot(index="timestamp", columns="asset", values="ema_cross_down").sort_index().fillna(0.0)
    gap_wide = test_panel.pivot(index="timestamp", columns="asset", values="ema_gap").sort_index().fillna(0.0)
    vol_wide = test_panel.pivot(index="timestamp", columns="asset", values="vol_21").sort_index().ffill()
    all_dates = close_wide.index.to_list()
    date_to_idx = {ts: i for i, ts in enumerate(all_dates)}
    n_dates = len(all_dates)
    if n_dates < 3:
        return pd.DataFrame()

    rows: list[dict[str, float | str]] = []
    last_end_by_asset: dict[str, int] = {}
    cooldown_until_by_asset: dict[str, int] = {}
    by_ts = {ts: g.copy() for ts, g in test_panel.groupby("timestamp", sort=True)}

    for ts in all_dates:
        sig_idx = date_to_idx[ts]
        daily = by_ts.get(ts)
        if daily is None or daily.empty:
            continue
        cands: list[dict[str, object]] = []
        for _, r in daily.iterrows():
            asset = str(r["asset"])
            up = float(r["ema_cross_up"]) > 0.5
            down = float(r["ema_cross_down"]) > 0.5
            if not up and not down:
                continue
            sign = 1.0 if up else -1.0
            if sign < 0 and not cfg.ema_allow_shorts:
                continue
            spy_up = float(r["spy_up"]) > 0.5
            if cfg.ema_require_spy_up and sign > 0 and not spy_up:
                continue
            if sig_idx <= int(last_end_by_asset.get(asset, -1)):
                continue
            if sig_idx <= int(cooldown_until_by_asset.get(asset, -1)):
                continue
            gap = float(abs(r["ema_gap"]))
            strength = float(max(1e-6, gap))
            conf = float(0.75 + min(1.0, 10.0 * gap))
            cands.append(
                {
                    "asset": asset,
                    "row": r,
                    "sign": sign,
                    "strength": strength,
                    "confidence": conf,
                    "priority": strength * conf,
                }
            )
        if not cands:
            continue
        cands.sort(key=lambda x: float(x["priority"]), reverse=True)

        for c in cands:
            asset = str(c["asset"])
            r = c["row"]
            sign = float(c["sign"])
            start_idx = sig_idx + 1
            hard_end = min(sig_idx + cfg.ema_max_hold_days, n_dates - 1)
            if start_idx >= n_dates or hard_end <= start_idx:
                continue
            exit_idx = hard_end
            exit_reason = "max_hold"
            # Exit on opposite cross.
            for j in range(start_idx, hard_end + 1):
                ts_j = all_dates[j]
                up_now = float(up_wide.loc[ts_j, asset]) > 0.5 if asset in up_wide.columns else False
                dn_now = float(dn_wide.loc[ts_j, asset]) > 0.5 if asset in dn_wide.columns else False
                if (sign > 0 and dn_now) or (sign < 0 and up_now):
                    exit_idx = j
                    exit_reason = "opposite_cross"
                    break
            rows.append(
                {
                    "fold_id": fold_id,
                    "sleeve": SLEEVE_EMA,
                    "asset": asset,
                    "signal_timestamp": str(ts),
                    "start_idx": float(start_idx),
                    "end_idx": float(exit_idx),
                    "signal_sign": sign,
                    "signal_strength": float(c["strength"]),
                    "risk_scale": 1.0 / float(max(cfg.vol_floor, float(vol_wide.loc[ts, asset]))),
                    "confidence_score": float(c["confidence"]),
                    "hold_days": float(exit_idx - sig_idx),
                    "exit_reason": exit_reason,
                    "ema_gap_at_signal": float(gap_wide.loc[ts, asset]),
                }
            )
            last_end_by_asset[asset] = exit_idx
            cooldown_until_by_asset[asset] = exit_idx + cfg.ema_reentry_cooldown_days
    return pd.DataFrame(rows)


def _weights_from_active(active: pd.DataFrame, cfg: Config, assets: list[str]) -> pd.Series:
    w = pd.Series(0.0, index=assets, dtype=float)
    if active.empty:
        return w
    a = active.copy()
    for c in ["signal_strength", "signal_sign", "risk_scale", "confidence_score"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a.dropna(subset=["signal_strength", "signal_sign", "risk_scale", "confidence_score"])
    if a.empty:
        return w
    a["raw_signed"] = a["signal_sign"] * a["signal_strength"] * a["risk_scale"] * a["confidence_score"]
    by_asset = a.groupby("asset", as_index=False).agg(raw=("raw_signed", "sum"))
    denom = float(np.sum(np.abs(by_asset["raw"])))
    if denom <= 0.0:
        return w
    by_asset["weight"] = (by_asset["raw"] / denom) * cfg.gross_target
    by_asset["weight"] = by_asset["weight"].clip(-cfg.max_abs_weight_per_asset, cfg.max_abs_weight_per_asset)
    for _, r in by_asset.iterrows():
        asset = str(r["asset"])
        if asset in w.index:
            w.loc[asset] = float(r["weight"])
    return w


def _simulate_sleeve_daily(
    test_panel: pd.DataFrame,
    trades: pd.DataFrame,
    spy_df: pd.DataFrame,
    cfg: Config,
    fold_id: str,
    sleeve_name: str,
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

    t = trades.copy()
    if t.empty:
        out = pd.DataFrame(
            {
                "timestamp": rets.index,
                "net_return": 0.0,
                "spy_return": spy_ret.values,
                "alpha_return": -spy_ret.values,
                "turnover": 0.0,
                "gross_exposure": 0.0,
                "strategy": sleeve_name,
                "fold_id": fold_id,
            }
        )
        out["equity"] = (1.0 + out["net_return"]).cumprod()
        out["running_max"] = out["equity"].cummax()
        out["drawdown"] = out["equity"] / out["running_max"] - 1.0
        return out, pd.DataFrame()

    t["start_idx"] = pd.to_numeric(t["start_idx"], errors="coerce").astype("Int64")
    t["end_idx"] = pd.to_numeric(t["end_idx"], errors="coerce").astype("Int64")
    t = t.dropna(
        subset=["start_idx", "end_idx", "asset", "signal_sign", "signal_strength", "risk_scale", "confidence_score"]
    ).copy()

    prev_w = pd.Series(0.0, index=assets, dtype=float)
    one_way = _one_way_cost_return(cfg)
    rows: list[dict[str, float | str]] = []
    for i in range(1, len(dates)):
        active = t[(t["start_idx"] <= i) & (t["end_idx"] >= i)]
        w = _weights_from_active(active=active, cfg=cfg, assets=assets)
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
                "strategy": sleeve_name,
                "fold_id": fold_id,
            }
        )
        prev_w = w

    out = pd.DataFrame(rows)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    out["running_max"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["running_max"] - 1.0

    trade_rows: list[dict[str, float | str]] = []
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
        bench = spy_exit / spy_entry - 1.0 if np.isfinite(spy_entry) and np.isfinite(spy_exit) and spy_entry > 0 else float(
            "nan"
        )
        trade_rows.append(
            {
                "fold_id": fold_id,
                "strategy": sleeve_name,
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


def _combine_sleeves_daily(
    sleeve_dailies: dict[str, pd.DataFrame],
    cfg: Config,
    fold_id: str,
) -> pd.DataFrame:
    if not sleeve_dailies:
        return pd.DataFrame()
    sleeve_names = list(sleeve_dailies.keys())
    base = sleeve_dailies[sleeve_names[0]][["timestamp", "spy_return"]].copy()
    base = base.sort_values("timestamp").drop_duplicates("timestamp")
    for name in sleeve_names:
        d = sleeve_dailies[name][["timestamp", "net_return", "turnover", "gross_exposure"]].copy()
        d = d.rename(
            columns={
                "net_return": f"{name}_ret",
                "turnover": f"{name}_turn",
                "gross_exposure": f"{name}_gross",
            }
        )
        base = base.merge(d, on="timestamp", how="left")
    base = base.sort_values("timestamp").reset_index(drop=True)
    for name in sleeve_names:
        base[f"{name}_ret"] = pd.to_numeric(base[f"{name}_ret"], errors="coerce").fillna(0.0)
        base[f"{name}_turn"] = pd.to_numeric(base[f"{name}_turn"], errors="coerce").fillna(0.0)
        base[f"{name}_gross"] = pd.to_numeric(base[f"{name}_gross"], errors="coerce").fillna(0.0)
    base["spy_return"] = pd.to_numeric(base["spy_return"], errors="coerce").fillna(0.0)

    alloc_cost = _combo_alloc_cost_return(cfg)
    prev_alloc = pd.Series(1.0 / len(sleeve_names), index=sleeve_names, dtype=float)
    rows: list[dict[str, float | str]] = []

    for i in range(len(base)):
        if i < cfg.combo_min_history_days:
            alloc = pd.Series(0.0, index=sleeve_names, dtype=float)
            alloc.loc[SLEEVE_REL2] = 1.0
            prev_alloc = alloc.copy() if i == 0 else prev_alloc
        elif (i % max(1, cfg.combo_rebalance_days)) != 0:
            alloc = prev_alloc.copy()
        else:
            start = max(0, i - cfg.combo_lookback_days)
            hist = pd.DataFrame(
                {name: base[f"{name}_ret"].iloc[start:i].to_numpy(dtype=float) for name in sleeve_names},
                dtype=float,
            )
            mu = hist.mean()
            vol = hist.std(ddof=1).replace(0.0, np.nan).fillna(cfg.vol_floor).clip(lower=cfg.vol_floor)
            sharpe_proxy = (mu / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # Disable sleeves that have weak/negative recent edge.
            sharpe_proxy = sharpe_proxy.where(sharpe_proxy >= cfg.combo_disable_sharpe_threshold, 0.0)
            corr = hist.corr().fillna(0.0)
            corr_pen = pd.Series(1.0, index=sleeve_names, dtype=float)
            if len(sleeve_names) > 1:
                for name in sleeve_names:
                    others = [o for o in sleeve_names if o != name]
                    avg_abs_corr = float(corr.loc[name, others].abs().mean()) if others else 0.0
                    corr_pen.loc[name] = 1.0 + cfg.combo_corr_penalty * avg_abs_corr
            raw = (sharpe_proxy / corr_pen) * (1.0 / vol)
            if SLEEVE_REL2 in raw.index:
                raw.loc[SLEEVE_REL2] = max(float(raw.loc[SLEEVE_REL2]), 1e-6)
            if float(raw.sum()) <= 1e-12:
                alloc_target = pd.Series(0.0, index=sleeve_names, dtype=float)
                alloc_target.loc[SLEEVE_REL2] = 1.0
            else:
                alloc_target = raw / max(1e-12, float(raw.sum()))
            # Keep rel2 as core sleeve with a minimum structural weight.
            if SLEEVE_REL2 in alloc_target.index and alloc_target.loc[SLEEVE_REL2] < cfg.combo_core_min_weight:
                others = [n for n in sleeve_names if n != SLEEVE_REL2]
                rem = max(1e-12, 1.0 - cfg.combo_core_min_weight)
                alloc_target.loc[SLEEVE_REL2] = cfg.combo_core_min_weight
                other_sum = float(alloc_target.loc[others].sum()) if others else 0.0
                if others and other_sum > 1e-12:
                    alloc_target.loc[others] = alloc_target.loc[others] * (rem / other_sum)
                elif others:
                    alloc_target.loc[others] = rem / len(others)
            alloc = (1.0 - cfg.combo_alloc_smoothing_lambda) * alloc_target + cfg.combo_alloc_smoothing_lambda * prev_alloc
            alloc = alloc / max(1e-12, float(alloc.sum()))

        sleeve_ret = pd.Series({name: float(base.loc[i, f"{name}_ret"]) for name in sleeve_names}, dtype=float)
        sleeve_turn = pd.Series({name: float(base.loc[i, f"{name}_turn"]) for name in sleeve_names}, dtype=float)
        sleeve_gross = pd.Series({name: float(base.loc[i, f"{name}_gross"]) for name in sleeve_names}, dtype=float)
        alloc_turn = float(np.abs(alloc - prev_alloc).sum()) if i > 0 else float(np.abs(alloc).sum())
        alloc_switch_cost = alloc_turn * alloc_cost
        net = float(np.dot(alloc.values, sleeve_ret.values)) - alloc_switch_cost
        spy_r = float(base.loc[i, "spy_return"])
        out = {
            "timestamp": str(base.loc[i, "timestamp"]),
            "net_return": net,
            "spy_return": spy_r,
            "alpha_return": net - spy_r,
            "turnover": float(np.dot(alloc.values, sleeve_turn.values) + alloc_turn),
            "gross_exposure": float(np.dot(alloc.values, sleeve_gross.values)),
            "strategy": SLEEVE_COMBO,
            "fold_id": fold_id,
        }
        for name in sleeve_names:
            out[f"alloc_{name}"] = float(alloc.loc[name])
        rows.append(out)
        prev_alloc = alloc.copy()

    out = pd.DataFrame(rows)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    out["running_max"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["running_max"] - 1.0
    return out


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
        {"timestamp": rets.index, "net_return": net, "turnover": turnover, "gross_exposure": 1.0}
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
        }
    net = pd.to_numeric(daily["net_return"], errors="coerce").fillna(0.0)
    ann = float(np.exp(np.log1p(net).mean() * 252.0) - 1.0)
    start, end = daily["timestamp"].iloc[0], daily["timestamp"].iloc[-1]
    years = max(1e-9, (end - start).total_seconds() / (365.25 * 24 * 3600))
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
        "avg_turnover": float(pd.to_numeric(daily["turnover"], errors="coerce").mean()),
        "avg_gross_exposure": float(pd.to_numeric(daily["gross_exposure"], errors="coerce").mean()),
    }


def main() -> None:
    out_root = Path("trading_research/examples/_output/38_multi_sleeve_alpha_combo")
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

    strategy_names = [SLEEVE_REL2, SLEEVE_VOL_COMP, SLEEVE_EMA, SLEEVE_COMBO, "buy_hold_all"]
    all_daily: dict[str, list[pd.DataFrame]] = {n: [] for n in strategy_names}
    all_trades: dict[str, list[pd.DataFrame]] = {n: [] for n in strategy_names}
    fold_rows: list[dict[str, float | str]] = []

    for fold_idx, test_year in enumerate(years[cfg.min_train_years :], start=1):
        train = panel[panel["year"] < test_year].copy()
        test = panel[panel["year"] == test_year].copy()
        if train.empty or test.empty:
            continue
        fold_id = f"fold_{test_year}"
        compression_thr = (
            train.groupby("asset")["bb_width"].quantile(cfg.compression_quantile).replace([np.inf, -np.inf], np.nan).to_dict()
        )

        trades_rel2 = _build_rel2_trades(test_panel=test, cfg=cfg, fold_id=fold_id)
        trades_comp = _build_vol_compression_trades(
            test_panel=test,
            compression_thresholds=compression_thr,
            cfg=cfg,
            fold_id=fold_id,
        )
        trades_ema = _build_ema_trades(test_panel=test, cfg=cfg, fold_id=fold_id)

        daily_rel2, trade_eval_rel2 = _simulate_sleeve_daily(
            test_panel=test, trades=trades_rel2, spy_df=spy_df, cfg=cfg, fold_id=fold_id, sleeve_name=SLEEVE_REL2
        )
        daily_comp, trade_eval_comp = _simulate_sleeve_daily(
            test_panel=test, trades=trades_comp, spy_df=spy_df, cfg=cfg, fold_id=fold_id, sleeve_name=SLEEVE_VOL_COMP
        )
        daily_ema, trade_eval_ema = _simulate_sleeve_daily(
            test_panel=test, trades=trades_ema, spy_df=spy_df, cfg=cfg, fold_id=fold_id, sleeve_name=SLEEVE_EMA
        )
        daily_combo = _combine_sleeves_daily(
            sleeve_dailies={SLEEVE_REL2: daily_rel2, SLEEVE_VOL_COMP: daily_comp, SLEEVE_EMA: daily_ema},
            cfg=cfg,
            fold_id=fold_id,
        )
        daily_bh = _simulate_buy_hold_all_fold(test_panel=test, spy_df=spy_df, cfg=cfg, fold_id=fold_id)

        fold_map = {
            SLEEVE_REL2: (daily_rel2, trade_eval_rel2),
            SLEEVE_VOL_COMP: (daily_comp, trade_eval_comp),
            SLEEVE_EMA: (daily_ema, trade_eval_ema),
            SLEEVE_COMBO: (daily_combo, pd.DataFrame()),
            "buy_hold_all": (daily_bh, pd.DataFrame()),
        }
        for name, (daily_df, trades_df) in fold_map.items():
            if daily_df.empty:
                continue
            all_daily[name].append(daily_df)
            if not trades_df.empty:
                all_trades[name].append(trades_df)
            fold_rows.append(
                {
                    "strategy": name,
                    "fold_id": fold_id,
                    "test_year": int(test_year),
                    "train_years": int(train["year"].nunique()),
                    "assets_in_test": int(test["asset"].nunique()),
                    **_portfolio_metrics(daily_df, trades_df),
                }
            )
        print(f"[fold {fold_idx}] {fold_id}: rel2={len(trades_rel2)} comp={len(trades_comp)} ema={len(trades_ema)}")

    fold_df = pd.DataFrame(fold_rows).sort_values(["strategy", "test_year"]).reset_index(drop=True)
    overall_rows: list[dict[str, float | str]] = []
    daily_all_list: list[pd.DataFrame] = []
    trades_all_list: list[pd.DataFrame] = []

    for name in strategy_names:
        daily_concat = pd.concat([d for d in all_daily[name] if not d.empty], ignore_index=True) if all_daily[name] else pd.DataFrame()
        trade_concat = (
            pd.concat([t for t in all_trades[name] if not t.empty], ignore_index=True) if all_trades[name] else pd.DataFrame()
        )
        if not daily_concat.empty:
            daily_concat = daily_concat.sort_values("timestamp").reset_index(drop=True)
            daily_all_list.append(daily_concat)
        if not trade_concat.empty:
            trades_all_list.append(trade_concat)
        overall_rows.append({"strategy": name, **_portfolio_metrics(daily_concat, trade_concat)})

    overall_df = pd.DataFrame(overall_rows).sort_values(
        ["annualized_sharpe_net", "annual_return", "cagr"], ascending=[False, False, False]
    )
    daily_oos = pd.concat(daily_all_list, ignore_index=True) if daily_all_list else pd.DataFrame()
    trades_oos = pd.concat(trades_all_list, ignore_index=True) if trades_all_list else pd.DataFrame()

    combo_alloc = daily_oos[daily_oos["strategy"] == SLEEVE_COMBO].copy() if not daily_oos.empty else pd.DataFrame()
    summary = {
        "universe_size_requested": len(UNIVERSE_170),
        "universe_size_fetched": int(panel["asset"].nunique()) if not panel.empty else 0,
        "history_start": str(panel["timestamp"].min()) if not panel.empty else None,
        "history_end": str(panel["timestamp"].max()) if not panel.empty else None,
        "n_walkforward_folds": int(fold_df["fold_id"].nunique()) if not fold_df.empty else 0,
        "sleeves": [SLEEVE_REL2, SLEEVE_VOL_COMP, SLEEVE_EMA, SLEEVE_COMBO],
        "config": {
            "period": cfg.period,
            "interval": cfg.interval,
            "lookback": cfg.signal_lookback_days,
            "one_way_cost_return": _one_way_cost_return(cfg),
            "combo_alloc_cost_return": _combo_alloc_cost_return(cfg),
            "combo_lookback_days": cfg.combo_lookback_days,
        },
    }

    fold_df.to_csv(reports_dir / "walkforward_fold_metrics.csv", index=False)
    overall_df.to_csv(reports_dir / "strategy_overall_comparison.csv", index=False)
    daily_oos.to_csv(reports_dir / "daily_oos_all_strategies.csv", index=False)
    trades_oos.to_csv(reports_dir / "trades_oos_all_strategies.csv", index=False)
    combo_alloc.to_csv(reports_dir / "combo_allocator_daily.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("overall comparison:")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
