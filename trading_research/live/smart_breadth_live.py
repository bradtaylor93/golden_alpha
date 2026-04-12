"""Live signal generation for smart-breadth strategy.

This module extracts the latest signal state from daily bars and converts it
into target portfolio weights compatible with live broker execution.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data.vendors.yahoo import YahooMarketDataVendor


def _load_strategy_module():
    """Load the strategy script as a module without requiring package import."""
    module_name = "trading_research.examples.live_rel2_44"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = Path(__file__).resolve().parents[1] / "examples" / "44_rel2_smart_breadth_leverage3_grid.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load strategy module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


strat = _load_strategy_module()


@dataclass(frozen=True)
class LiveSignalConfig:
    strategy_name: str = "smart_breadth_quality_3x"
    history_period: str = "1y"
    interval: str = "1d"
    gross_target: float = 3.0
    max_abs_weight: float = 0.12
    top_quantile: float = 0.92
    rel_threshold: float = 0.02
    adv_rank_min: float = 0.40
    short_gross_fraction: float = 0.70


def _strategy_from_name(name: str) -> strat.ExperimentSpec:
    for exp in strat.EXPERIMENTS:
        if exp.name == name:
            return exp
    raise ValueError(f"Unknown strategy name: {name}")


def _latest_snapshot(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    if panel.empty:
        raise ValueError("Panel is empty.")
    ts = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce").max()
    snap = panel[pd.to_datetime(panel["timestamp"], utc=True, errors="coerce") == ts].copy()
    if snap.empty:
        raise ValueError("No latest snapshot rows found.")
    return snap, ts


def _filter_candidates(
    snapshot: pd.DataFrame,
    cfg: strat.Config,
    exp: strat.ExperimentSpec,
    efficacy_map: dict[str, float],
    efficacy_threshold: float | None,
) -> pd.DataFrame:
    df = snapshot.copy()
    top_q = float(exp.top_quantile_override) if exp.top_quantile_override is not None else cfg.top_quantile
    rel_thresh = (
        float(exp.rel_strength_threshold_override)
        if exp.rel_strength_threshold_override is not None
        else cfg.rel_strength_threshold
    )
    df = df[(df["rank_pct"] >= top_q) & (df["rel_strength_63"] >= rel_thresh)].copy()
    if df.empty:
        return df
    if exp.use_liquidity_filter:
        df = df[df["adv_rank_pct"] >= cfg.smart_liquidity_min_rank_pct].copy()
    if exp.use_asset_efficacy_filter and efficacy_map:
        eff = df["asset"].map(efficacy_map).astype(float)
        if efficacy_threshold is not None and np.isfinite(efficacy_threshold):
            df = df[eff >= float(efficacy_threshold)].copy()
    if df.empty:
        return df
    df["trend_sign"] = np.sign(df["past_return"])
    df = df[df["trend_sign"] != 0.0].copy()
    if not exp.allow_shorts:
        df = df[df["trend_sign"] > 0.0].copy()
    if exp.require_spy_up_for_longs:
        df = df[~((df["trend_sign"] > 0.0) & (df["spy_up"] <= 0.5))].copy()
    if exp.require_spy_down_for_shorts:
        df = df[~((df["trend_sign"] < 0.0) & (df["spy_up"] > 0.5))].copy()
    if exp.use_asymmetric_short_filter:
        short_top_q = (
            float(exp.short_top_quantile_override)
            if exp.short_top_quantile_override is not None
            else min(0.995, top_q + cfg.short_rank_buffer)
        )
        short_rel_th = (
            float(exp.short_rel_strength_threshold_override)
            if exp.short_rel_strength_threshold_override is not None
            else rel_thresh + cfg.short_rel_strength_buffer
        )
        keep = ~(
            (df["trend_sign"] < 0.0) & ((df["rank_pct"] < short_top_q) | (df["rel_strength_63"] < short_rel_th))
        )
        df = df[keep].copy()
    return df


def build_target_weights(
    *,
    universe: tuple[str, ...],
    cfg: strat.Config,
    live_cfg: LiveSignalConfig,
    vendor: YahooMarketDataVendor | None = None,
) -> tuple[pd.Series, pd.Timestamp]:
    """Compute today's target portfolio weights for the selected strategy."""
    exp = _strategy_from_name(live_cfg.strategy_name)
    md_vendor = vendor or YahooMarketDataVendor()
    bars = md_vendor.fetch_bars(list(universe), period=live_cfg.history_period, interval=live_cfg.interval)
    spy = md_vendor.fetch_bars(["SPY"], period=live_cfg.history_period, interval=live_cfg.interval)[["timestamp", "close"]]
    if bars.empty or spy.empty:
        raise ValueError("Missing market data for universe/SPY.")
    panel = strat._build_panel(bars, spy, cfg)
    if panel.empty:
        raise ValueError("Panel is empty after feature construction.")

    snapshot, ts = _latest_snapshot(panel)
    train = panel[pd.to_datetime(panel["timestamp"], utc=True, errors="coerce") < ts].copy()
    efficacy_map, efficacy_thr = ({}, None)
    if exp.use_asset_efficacy_filter and not train.empty:
        eff_q = (
            float(exp.asset_efficacy_quantile_override)
            if exp.asset_efficacy_quantile_override is not None
            else cfg.asset_efficacy_quantile
        )
        efficacy_map, efficacy_thr = strat._compute_asset_efficacy(
            train_panel=train,
            horizon_days=cfg.asset_efficacy_horizon_days,
            min_obs=cfg.asset_efficacy_min_obs,
            quantile_threshold=eff_q,
        )

    cands = _filter_candidates(snapshot, cfg=cfg, exp=exp, efficacy_map=efficacy_map, efficacy_threshold=efficacy_thr)
    if cands.empty:
        return pd.Series(0.0, index=sorted(snapshot["asset"].astype(str).unique()), dtype=float), ts

    top_q = (
        float(exp.top_quantile_override)
        if exp.top_quantile_override is not None
        else float(live_cfg.top_quantile)
    )
    rel_thresh = (
        float(exp.rel_strength_threshold_override)
        if exp.rel_strength_threshold_override is not None
        else float(live_cfg.rel_threshold)
    )
    rank_conf = np.clip((cands["rank_pct"] - top_q) / max(1e-6, 1.0 - top_q), 0.0, 1.0)
    rel_conf = np.clip((cands["rel_strength_63"] - rel_thresh) / 0.05, 0.0, 1.0)
    if exp.use_confidence_calibration:
        vol_norm = np.clip((cands["vol_21"] / max(1e-6, float(cands["spy_vol_21"].median()))) - 1.0, 0.0, 2.0)
        signal = cfg.confidence_calib_rel_weight * rel_conf + cfg.confidence_calib_rank_weight * rank_conf - (
            cfg.confidence_calib_vol_penalty * vol_norm
        )
        conf = np.clip(cfg.confidence_calib_base + cfg.confidence_calib_scale * signal, 0.25, 3.0)
    else:
        conf = 0.5 + 1.5 * (0.5 * rank_conf + 0.5 * rel_conf)
    cands = cands.copy()
    cands["confidence_score"] = conf.astype(float)
    cands["signal_sign"] = cands["trend_sign"].astype(float)
    cands["signal_strength"] = np.maximum(1e-6, cands["rank_pct"] - top_q)
    cands["entry_vol_21"] = np.maximum(cfg.vol_floor, cands["vol_21"].astype(float))

    if exp.use_quality_priority_cap:
        cands = cands.sort_values(["signal_strength", "confidence_score"], ascending=False).head(
            cfg.quality_cap_max_new_entries_per_day
        )
    elif exp.use_crowding_cap:
        cands = cands.sort_values(["signal_strength", "confidence_score"], ascending=False).head(
            cfg.crowding_max_new_entries_per_day
        )

    gross_target = float(live_cfg.gross_target)
    max_abs_w = float(live_cfg.max_abs_weight)
    weights = strat._weights_from_active(
        active=cands[["asset", "signal_strength", "signal_sign", "entry_vol_21", "confidence_score"]],
        cfg=cfg,
        assets=sorted(snapshot["asset"].astype(str).unique()),
        cluster_map=None,
        use_cluster_caps=False,
        gross_target=gross_target,
        max_abs_weight_per_asset=max_abs_w,
    )
    if exp.use_asymmetric_short_filter:
        short_frac = (
            float(exp.short_gross_fraction_override)
            if exp.short_gross_fraction_override is not None
            else float(live_cfg.short_gross_fraction)
        )
        short_w = float(np.abs(weights[weights < 0.0]).sum())
        long_w = float(np.abs(weights[weights > 0.0]).sum())
        max_short = short_frac * long_w
        if short_w > max_short and short_w > 1e-12:
            scale = max_short / short_w
            weights.loc[weights < 0.0] = weights.loc[weights < 0.0] * scale
    return weights.astype(float), ts


def default_universe_170() -> tuple[str, ...]:
    """Return the legacy 170-name universe used in studies."""
    return tuple(strat.UNIVERSE_170)


def explain_rebalance(
    *,
    target_weights: dict[str, float],
    positions: dict[str, float],
    quotes: dict[str, float],
    equity: float,
) -> dict[str, object]:
    """Build a human-readable rebalance summary."""
    rows: list[dict[str, float | str]] = []
    syms = sorted(set(target_weights.keys()) | set(positions.keys()))
    for sym in syms:
        px = float(quotes.get(sym, float("nan")))
        tgt_w = float(target_weights.get(sym, 0.0))
        cur_q = float(positions.get(sym, 0.0))
        cur_notional = cur_q * px if np.isfinite(px) else float("nan")
        tgt_notional = tgt_w * float(equity)
        delta_notional = tgt_notional - (cur_notional if np.isfinite(cur_notional) else 0.0)
        rows.append(
            {
                "symbol": sym,
                "price": px,
                "target_weight": tgt_w,
                "target_notional": tgt_notional,
                "current_qty": cur_q,
                "current_notional": cur_notional,
                "delta_notional": delta_notional,
            }
        )
    rows = sorted(rows, key=lambda r: abs(float(r["delta_notional"])), reverse=True)
    return {
        "n_symbols": len(rows),
        "sum_abs_target_weight": float(sum(abs(float(v)) for v in target_weights.values())),
        "top_rebalance_rows": rows[:50],
    }

