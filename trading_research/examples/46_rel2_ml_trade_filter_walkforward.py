"""Ablation stack to improve nhf5_reentry_cooldown.

Goal:
- Start from the known best baseline (`nhf5_reentry_cooldown`)
- Test ranked improvement levers one-by-one and in combinations:
  1) regime-aware exposure throttle
  2) tail-loss control (vol-normalized trailing + hard stop)
  3) adaptive cooldown
  4) expected-edge floor filter
  5) dynamic cluster caps
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:  # pragma: no cover - optional dependency at runtime.
    HistGradientBoostingClassifier = None  # type: ignore[assignment]

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


def _extract_symbols_from_tables(tables: list[pd.DataFrame]) -> list[str]:
    out: list[str] = []
    for table in tables:
        symbols_col = None
        for col in table.columns:
            col_name = str(col).strip().lower()
            if "symbol" in col_name or "ticker" in col_name:
                symbols_col = col
                break
        if symbols_col is None:
            continue
        raw = table[symbols_col].astype(str).tolist()
        # Yahoo convention uses '-' for share class tickers.
        cleaned = [s.strip().upper().replace(".", "-") for s in raw if s and s != "nan"]
        # Guard against non-ticker text in loose tables.
        cleaned = [s for s in cleaned if 1 <= len(s) <= 8 and " " not in s]
        out.extend(cleaned)
    return out


def _load_broad_us_universe(max_assets: int) -> tuple[str, ...]:
    """Load broad US equity universe (S&P 500/400/600 + Nasdaq-100).

    Uses CSV data sources first (no optional HTML parser dependency).
    """
    if int(max_assets) <= len(UNIVERSE_170):
        return tuple(list(UNIVERSE_170)[: max(1, int(max_assets))])

    csv_urls = (
        "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://raw.githubusercontent.com/nasdaq/nasdaq100/master/data/constituents.csv",
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt",
    )
    all_symbols: list[str] = []
    for url in csv_urls:
        try:
            if url.endswith(".txt"):
                df = pd.read_csv(url, header=None)
                raw = df.iloc[:, 0].astype(str).tolist()
            else:
                df = pd.read_csv(url)
                symbol_col = None
                for c in df.columns:
                    name = str(c).strip().lower()
                    if "symbol" in name or "ticker" in name:
                        symbol_col = c
                        break
                if symbol_col is None:
                    continue
                raw = df[symbol_col].astype(str).tolist()
            cleaned = [s.strip().upper().replace(".", "-") for s in raw if s and s != "nan"]
            # Keep plain US-style tickers and share class variants.
            cleaned = [s for s in cleaned if 1 <= len(s) <= 8 and " " not in s and s.replace("-", "").isalnum()]
            all_symbols.extend(cleaned)
        except Exception:
            continue
    # Stable fallback add-on.
    all_symbols.extend(list(UNIVERSE_170))
    uniq = tuple(dict.fromkeys(all_symbols))
    if not uniq:
        return tuple(list(UNIVERSE_170)[: max(1, int(max_assets))])
    return uniq[: max(1, int(max_assets))]


@dataclass(frozen=True)
class Config:
    period: str = "7y"
    interval: str = "1d"
    signal_lookback_days: int = 63
    rolling_vwap_window: int = 20
    vol_window_days: int = 21
    rel_strength_threshold: float = 0.02
    top_quantile: float = 0.92
    min_train_years: int = 3
    # 3x leverage run defaults.
    gross_target: float = 3.0
    max_abs_weight_per_asset: float = 0.12
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    vol_floor: float = 1e-4
    # Base strategy parameters.
    bull_hold_days: int = 98
    bear_hold_days: int = 42
    bear_trailing_stop_pct: float = 0.10
    # Cluster-cap baseline.
    cluster_corr_threshold: float = 0.75
    cluster_abs_weight_cap: float = 0.75
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
    # Improvement-stack controls.
    hard_stop_loss_pct: float = 0.20
    edge_floor_base: float = 0.015
    edge_floor_stressed: float = 0.030
    dynamic_cluster_abs_weight_cap_stressed: float = 0.54
    dynamic_cluster_abs_weight_cap_calm: float = 0.90
    # Smart-breadth controls.
    max_universe_assets: int = 170
    asset_efficacy_horizon_days: int = 21
    asset_efficacy_min_obs: int = 120
    asset_efficacy_quantile: float = 0.55
    smart_liquidity_min_rank_pct: float = 0.40
    # Calibrated confidence and crowding controls.
    confidence_calib_rel_weight: float = 1.0
    confidence_calib_rank_weight: float = 0.7
    confidence_calib_vol_penalty: float = 0.6
    confidence_calib_base: float = 0.9
    confidence_calib_scale: float = 0.8
    corr_lookback_days: int = 63
    corr_scale_min: float = 0.75
    corr_scale_max: float = 1.05
    short_rel_strength_buffer: float = 0.01
    short_rank_buffer: float = 0.02
    profit_lock_trigger: float = 0.10
    profit_lock_trail: float = 0.06
    # ML trade filter controls.
    trade_filter_min_train_trades: int = 120
    trade_filter_prob_quantile: float = 0.60
    trade_filter_soft_scale: float = 0.50
    trade_filter_ensemble_hgb_weight: float = 0.60
    trade_filter_ensemble_rf_weight: float = 0.40
    trade_filter_recency_half_life_days: int = 252
    trade_filter_edge_boost: float = 0.40
    trade_filter_threshold_min_keep_fraction: float = 0.25
    trade_filter_threshold_max_keep_fraction: float = 0.85
    trade_filter_threshold_min_selected: int = 40
    trade_filter_threshold_grid_size: int = 21
    # Universe restriction by cross-sectional cashflow proxy (21d dollar volume rank).
    cashflow_quartile_rank_threshold: float = 0.75
    cashflow_soft_rank_floor: float = 0.55
    cashflow_soft_min_scale: float = 0.70
    # Market stop-loss controls (using SPY drawdown from rolling peak).
    market_stop_drawdown: float = 0.10
    market_stop_recovery_drawdown: float = 0.04
    market_stop_scale: float = 0.0
    market_stop_trigger_days: int = 1
    market_stop_vol_adj_min: float = 0.75
    market_stop_vol_adj_max: float = 1.25
    # Global risk-on gate: do not open trades unless SPY 3-month return is positive.
    require_spy_3m_positive: bool = True


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
    allow_shorts: bool = True
    use_custom_edge_threshold: bool = False
    custom_base_edge_threshold: float | None = None
    custom_additional_edge_threshold: float | None = None
    gross_target_override: float | None = None
    max_abs_weight_per_asset_override: float | None = None
    portfolio_vol_target_annual_override: float | None = None
    top_quantile_override: float | None = None
    rel_strength_threshold_override: float | None = None
    cooldown_days_override: int | None = None
    # New ablation toggles.
    use_hard_stop_loss: bool = False
    use_dynamic_cluster_caps: bool = False
    stressed_gross_scale_override: float | None = None
    calm_gross_scale_override: float | None = None
    hard_stop_loss_pct_override: float | None = None
    use_asset_efficacy_filter: bool = False
    asset_efficacy_quantile_override: float | None = None
    # Four-step Sharpe improvement pack.
    use_confidence_calibration: bool = False
    use_corr_aware_gross_scaler: bool = False
    use_asymmetric_short_filter: bool = False
    short_gross_fraction_override: float | None = None
    short_rel_strength_threshold_override: float | None = None
    short_top_quantile_override: float | None = None
    use_profit_lock_exit: bool = False
    profit_lock_threshold_override: float | None = None
    profit_lock_trail_override: float | None = None
    # ML trade filter (trained on train-fold win/loss trades).
    use_trade_filter_model: bool = False
    trade_filter_prob_quantile_override: float | None = None
    use_trade_filter_hard_gate: bool = True
    use_trade_filter_soft_weighting: bool = False
    trade_filter_backfill_fraction_override: float | None = None
    use_trade_filter_recency_weighting: bool = False
    use_trade_filter_adaptive_threshold: bool = False
    # Additional robustness variants requested by user.
    use_cashflow_quartile_restrictor: bool = False
    cashflow_quartile_threshold_override: float | None = None
    use_soft_cashflow_weighting: bool = False
    cashflow_soft_floor_override: float | None = None
    cashflow_soft_min_scale_override: float | None = None
    use_ml_topk_selector: bool = False
    ml_topk_per_day_override: int | None = None
    use_market_stop_loss: bool = False
    market_stop_drawdown_override: float | None = None
    market_stop_recovery_override: float | None = None
    market_stop_scale_override: float | None = None
    use_market_stop_vol_adjustment: bool = False
    market_stop_trigger_days_override: int | None = None


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        name="smart_breadth_quality_3x",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_filter_q60",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        trade_filter_prob_quantile_override=0.60,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_filter_q70",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        trade_filter_prob_quantile_override=0.70,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_filter_q55",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        trade_filter_prob_quantile_override=0.55,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_filter_q58",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        trade_filter_prob_quantile_override=0.58,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_filter_q55_softw",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_soft_weighting=True,
        trade_filter_prob_quantile_override=0.55,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_filter_q58_softw",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_soft_weighting=True,
        trade_filter_prob_quantile_override=0.58,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_filter_q55_hybrid",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=False,
        use_trade_filter_soft_weighting=True,
        trade_filter_prob_quantile_override=0.55,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_filter_q60_hybrid",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=False,
        use_trade_filter_soft_weighting=True,
        trade_filter_prob_quantile_override=0.60,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_filter_q60_hybrid_backfill90",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=False,
        use_trade_filter_soft_weighting=True,
        trade_filter_backfill_fraction_override=0.90,
        trade_filter_prob_quantile_override=0.60,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_filter_q60_backfill85",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.85,
        trade_filter_prob_quantile_override=0.60,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_filter_q60_backfill90",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.90,
        trade_filter_prob_quantile_override=0.60,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_v2",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        use_trade_filter_soft_weighting=True,
        trade_filter_backfill_fraction_override=0.88,
        trade_filter_prob_quantile_override=0.58,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_v2b",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_trade_filter_recency_weighting=False,
        use_trade_filter_adaptive_threshold=False,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.12,
        market_stop_recovery_override=0.05,
        use_soft_cashflow_weighting=True,
        cashflow_soft_floor_override=0.55,
        cashflow_soft_min_scale_override=0.70,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_v3_softw",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        use_trade_filter_soft_weighting=True,
        trade_filter_backfill_fraction_override=0.86,
        trade_filter_prob_quantile_override=0.57,
        use_trade_filter_recency_weighting=True,
        use_trade_filter_adaptive_threshold=True,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_market_stop_10",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.10,
        market_stop_recovery_override=0.04,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_market_stop_12",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.12,
        market_stop_recovery_override=0.05,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_stop12_cashflow_q4",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.12,
        market_stop_recovery_override=0.05,
        use_cashflow_quartile_restrictor=True,
        cashflow_quartile_threshold_override=0.75,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_stop12_cashflow_soft",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.12,
        market_stop_recovery_override=0.05,
        use_soft_cashflow_weighting=True,
        cashflow_soft_floor_override=0.55,
        cashflow_soft_min_scale_override=0.70,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_stop12_voladj",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.12,
        market_stop_recovery_override=0.05,
        use_market_stop_vol_adjustment=True,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_stop12_grace2",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.12,
        market_stop_recovery_override=0.05,
        market_stop_trigger_days_override=2,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_stop12_ml_topk20",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.12,
        market_stop_recovery_override=0.05,
        use_ml_topk_selector=True,
        ml_topk_per_day_override=20,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_stop12_ml_topk20_cashflow_soft",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.12,
        market_stop_recovery_override=0.05,
        use_ml_topk_selector=True,
        ml_topk_per_day_override=20,
        use_soft_cashflow_weighting=True,
        cashflow_soft_floor_override=0.55,
        cashflow_soft_min_scale_override=0.70,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_market_stop_11",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.11,
        market_stop_recovery_override=0.045,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_market_stop_13",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.13,
        market_stop_recovery_override=0.06,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_market_stop_12_recover_04",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.12,
        market_stop_recovery_override=0.04,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_market_stop_12_scale_20",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.12,
        market_stop_recovery_override=0.05,
        market_stop_scale_override=0.20,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_cashflow_q4",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_cashflow_quartile_restrictor=True,
        cashflow_quartile_threshold_override=0.75,
    ),
    ExperimentSpec(
        name="smart_breadth_quality_3x_ml_champion_market_stop_10_cashflow_q4",
        use_reentry_cooldown=False,
        use_liquidity_filter=True,
        use_asset_efficacy_filter=True,
        use_dynamic_cluster_caps=True,
        use_dynamic_edge_floor=True,
        use_custom_edge_threshold=True,
        custom_base_edge_threshold=0.010,
        custom_additional_edge_threshold=0.010,
        gross_target_override=3.0,
        max_abs_weight_per_asset_override=0.12,
        use_trade_filter_model=True,
        use_trade_filter_hard_gate=True,
        trade_filter_backfill_fraction_override=0.82,
        trade_filter_prob_quantile_override=0.57,
        use_market_stop_loss=True,
        market_stop_drawdown_override=0.10,
        market_stop_recovery_override=0.04,
        use_cashflow_quartile_restrictor=True,
        cashflow_quartile_threshold_override=0.75,
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
    spy["spy_ret_126"] = spy["close"] / spy["close"].shift(63) - 1.0
    spy["spy_3m_positive"] = (spy["spy_ret_126"] > 0.0).astype(float)
    spy["spy_ma200"] = spy["close"].rolling(200, min_periods=200).mean()
    spy["spy_up"] = (spy["close"] > spy["spy_ma200"]).astype(float)
    spy["spy_vol_21"] = spy["close"].pct_change().rolling(cfg.vol_window_days, min_periods=cfg.vol_window_days).std()
    spy = spy.rename(columns={"close": "spy_close"})
    frame = frame.merge(
        spy[["timestamp", "spy_close", "spy_ret_63", "spy_ret_126", "spy_3m_positive", "spy_up", "spy_vol_21"]],
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
            "spy_ret_126",
            "spy_3m_positive",
            "spy_up",
            "spy_vol_21",
            "rel_strength_63",
        ]
    )
    frame["year"] = frame["timestamp"].dt.year
    return frame.reset_index(drop=True)


def _compute_asset_efficacy(
    train_panel: pd.DataFrame,
    horizon_days: int,
    min_obs: int,
    quantile_threshold: float,
) -> tuple[dict[str, float], float]:
    """Estimate per-asset signal efficacy from train data only."""
    if train_panel.empty:
        return {}, float("-inf")
    h = max(1, int(horizon_days))
    p = train_panel[["timestamp", "asset", "close", "past_return", "spy_close"]].copy()
    p = p.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    p["signal_sign"] = np.sign(pd.to_numeric(p["past_return"], errors="coerce"))
    p["asset_fwd"] = (
        p.groupby("asset", sort=False)["close"].shift(-h) / p["close"].replace(0.0, np.nan) - 1.0
    )
    spy_ts = p[["timestamp", "spy_close"]].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    spy_ts["spy_fwd"] = spy_ts["spy_close"].shift(-h) / spy_ts["spy_close"].replace(0.0, np.nan) - 1.0
    p = p.merge(spy_ts[["timestamp", "spy_fwd"]], on="timestamp", how="left")
    p["eff_sample"] = p["signal_sign"] * (p["asset_fwd"] - p["spy_fwd"])
    p = p.replace([np.inf, -np.inf], np.nan).dropna(subset=["eff_sample"])
    if p.empty:
        return {}, float("-inf")
    g = p.groupby("asset", as_index=False).agg(efficacy=("eff_sample", "mean"), n=("eff_sample", "count"))
    g = g[g["n"] >= int(min_obs)].copy()
    if g.empty:
        return {}, float("-inf")
    scores = pd.to_numeric(g["efficacy"], errors="coerce").dropna()
    if scores.empty:
        return {}, float("-inf")
    thr = float(scores.quantile(float(np.clip(quantile_threshold, 0.0, 1.0))))
    return {str(a): float(e) for a, e in zip(g["asset"], g["efficacy"])}, thr


def _confidence_from_row(row: pd.Series, cfg: Config) -> float:
    rank = float(row["rank_pct"])
    rel = float(row["rel_strength_63"])
    rank_conf = float(np.clip((rank - cfg.top_quantile) / max(1e-6, 1.0 - cfg.top_quantile), 0.0, 1.0))
    rel_conf = float(np.clip((rel - cfg.rel_strength_threshold) / 0.05, 0.0, 1.0))
    return float(0.5 + 1.5 * (0.5 * rank_conf + 0.5 * rel_conf))


def _build_confidence_calibrator(train_panel: pd.DataFrame) -> tuple[float, float]:
    """Fit simple linear confidence calibration from train fold only."""
    if train_panel.empty:
        return 0.0, 0.0
    p = train_panel[["asset", "timestamp", "close", "rank_pct", "rel_strength_63", "past_return"]].copy()
    p = p.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    p["rank_pct"] = pd.to_numeric(p["rank_pct"], errors="coerce")
    p["rel_strength_63"] = pd.to_numeric(p["rel_strength_63"], errors="coerce")
    p["past_return"] = pd.to_numeric(p["past_return"], errors="coerce")
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    p = p.replace([np.inf, -np.inf], np.nan).dropna(subset=["rank_pct", "rel_strength_63", "past_return", "close"])
    if p.empty:
        return 0.0, 0.0

    p["trend_sign"] = np.sign(p["past_return"])
    p = p[p["trend_sign"] != 0.0].copy()
    if p.empty:
        return 0.0, 0.0
    p["fwd_ret"] = p.groupby("asset", sort=False)["close"].shift(-1) / p["close"].replace(0.0, np.nan) - 1.0
    p = p.replace([np.inf, -np.inf], np.nan).dropna(subset=["fwd_ret"])
    if p.empty:
        return 0.0, 0.0

    rank_centered = np.clip((p["rank_pct"] - 0.90) / 0.10, 0.0, 1.0)
    rel_centered = np.clip((p["rel_strength_63"] - 0.01) / 0.05, 0.0, 1.0)
    p["conf_signal"] = 0.5 + 1.5 * (0.5 * rank_centered + 0.5 * rel_centered)
    p["edge_1d"] = p["trend_sign"] * p["fwd_ret"]
    p = p.replace([np.inf, -np.inf], np.nan).dropna(subset=["conf_signal", "edge_1d"])
    if len(p) < 50:
        return 0.0, 0.0

    x = p["conf_signal"].to_numpy(dtype=float)
    y = p["edge_1d"].to_numpy(dtype=float)
    x_mu = float(np.mean(x))
    y_mu = float(np.mean(y))
    var_x = float(np.var(x, ddof=1))
    if not np.isfinite(var_x) or var_x <= 1e-12:
        return 0.0, y_mu
    cov_xy = float(np.cov(x, y, ddof=1)[0, 1])
    slope = cov_xy / var_x
    intercept = y_mu - slope * x_mu
    if not np.isfinite(slope):
        slope = 0.0
    if not np.isfinite(intercept):
        intercept = y_mu if np.isfinite(y_mu) else 0.0
    return float(slope), float(intercept)


TRADE_FILTER_FEATURES: tuple[str, ...] = (
    "rank_pct",
    "rel_strength_63",
    "score",
    "past_return",
    "vol_21",
    "adv_rank_pct",
    "spy_up",
    "spy_vol_21",
    "signal_strength",
    "confidence_score",
    "edge_proxy",
    "trend_sign",
    # richer interaction / normalization features
    "abs_score",
    "score_x_rel",
    "score_over_vol",
    "past_return_over_vol",
    "rel_over_spy_vol",
    "vol_vs_spy_vol",
    "edge_over_vol",
    "conf_x_rank",
    "conf_x_rel",
    "rank_x_rel",
    "score_sq",
    "rel_sq",
    "spy_ret_63",
    "spy_ret_126",
    "liquidity_edge",
    "rank_minus_adv",
    "rel_minus_score",
    "rank_x_score",
    "sign_x_score",
    "sign_x_rel",
)


def _trade_feature_row(
    row: pd.Series,
    *,
    trend_sign: float,
    strength: float,
    conf: float,
    edge_proxy: float,
) -> dict[str, float]:
    score = float(row["score"])
    rel = float(row["rel_strength_63"])
    past = float(row["past_return"])
    vol = float(row["vol_21"])
    adv = float(row["adv_rank_pct"])
    spy_vol = float(row["spy_vol_21"])
    vol_safe = max(1e-6, abs(vol))
    spy_vol_safe = max(1e-6, abs(spy_vol))
    return {
        "rank_pct": float(row["rank_pct"]),
        "rel_strength_63": rel,
        "score": score,
        "past_return": past,
        "vol_21": vol,
        "adv_rank_pct": adv,
        "spy_up": float(row["spy_up"]),
        "spy_vol_21": spy_vol,
        "signal_strength": float(strength),
        "confidence_score": float(conf),
        "edge_proxy": float(edge_proxy),
        "trend_sign": float(trend_sign),
        "abs_score": float(abs(score)),
        "score_x_rel": float(score * rel),
        "score_over_vol": float(score / vol_safe),
        "past_return_over_vol": float(past / vol_safe),
        "rel_over_spy_vol": float(rel / spy_vol_safe),
        "vol_vs_spy_vol": float(vol / spy_vol_safe),
        "edge_over_vol": float(edge_proxy / vol_safe),
        "conf_x_rank": float(conf * float(row["rank_pct"])),
        "conf_x_rel": float(conf * rel),
        "rank_x_rel": float(float(row["rank_pct"]) * rel),
        "score_sq": float(score * score),
        "rel_sq": float(rel * rel),
        "spy_ret_63": float(row["spy_ret_63"]),
        "spy_ret_126": float(row["spy_ret_126"]),
        "liquidity_edge": float((1.0 - np.clip(adv, 0.0, 1.0)) * edge_proxy),
        "rank_minus_adv": float(float(row["rank_pct"]) - adv),
        "rel_minus_score": float(rel - score),
        "rank_x_score": float(float(row["rank_pct"]) * score),
        "sign_x_score": float(trend_sign * score),
        "sign_x_rel": float(trend_sign * rel),
    }


def _select_trade_filter_threshold(
    *,
    prob: np.ndarray,
    net_ret: np.ndarray,
    cfg: Config,
    q: float,
    use_adaptive: bool,
) -> float:
    q_use = float(np.clip(q, 0.0, 0.95))
    base = float(np.quantile(prob, q_use))
    if (not use_adaptive) or prob.size < 120 or net_ret.size != prob.size:
        return base if np.isfinite(base) else 0.50

    grid_size = max(7, int(cfg.trade_filter_threshold_grid_size))
    q_grid = np.linspace(0.05, 0.95, grid_size)
    thresholds = np.unique(np.quantile(prob, q_grid))
    if thresholds.size == 0:
        return base if np.isfinite(base) else 0.50

    n_total = int(prob.size)
    min_keep = max(
        int(cfg.trade_filter_threshold_min_selected),
        int(round(n_total * float(np.clip(cfg.trade_filter_threshold_min_keep_fraction, 0.0, 1.0)))),
    )
    max_keep = max(
        min_keep + 1,
        int(round(n_total * float(np.clip(cfg.trade_filter_threshold_max_keep_fraction, 0.0, 1.0)))),
    )
    target_keep = float(np.clip(1.0 - q_use, 0.05, 0.95))

    best_thr = base
    best_obj = float("-inf")
    for thr in thresholds.tolist():
        mask = prob >= float(thr)
        n_sel = int(mask.sum())
        if n_sel < min_keep or n_sel > max_keep:
            continue
        selected = net_ret[mask]
        if selected.size < 2:
            continue
        mu = float(np.mean(selected))
        sigma = float(np.std(selected, ddof=1))
        if not np.isfinite(mu):
            continue
        cov = float(n_sel) / float(max(1, n_total))
        keep_gap = abs(cov - target_keep)
        # Sharpe-like term + mean-return term, with a mild penalty for extreme keep rates.
        obj = (mu / max(1e-6, sigma)) * math.sqrt(cov) + 40.0 * mu - 0.10 * keep_gap
        if obj > best_obj:
            best_obj = obj
            best_thr = float(thr)
    if not np.isfinite(best_thr):
        return 0.50
    return float(best_thr)


def _fit_trade_filter_model(
    *,
    train_panel: pd.DataFrame,
    cfg: Config,
    exp: ExperimentSpec,
    train_spy_vol_median: float,
    asset_efficacy_map: dict[str, float],
    asset_efficacy_threshold: float | None,
    conf_cal_slope: float,
    conf_cal_intercept: float,
) -> tuple[HistGradientBoostingClassifier | None, float | None]:
    if HistGradientBoostingClassifier is None:
        return None, None
    if train_panel.empty:
        return None, None
    train_exp = replace(exp, use_trade_filter_model=False)
    train_trades = _build_trades(
        test_panel=train_panel,
        cfg=cfg,
        exp=train_exp,
        fold_id="train_filter",
        train_spy_vol_median=train_spy_vol_median,
        asset_efficacy_map=asset_efficacy_map,
        asset_efficacy_threshold=asset_efficacy_threshold,
        conf_cal_slope=conf_cal_slope,
        conf_cal_intercept=conf_cal_intercept,
    )
    if train_trades.empty or len(train_trades) < int(cfg.trade_filter_min_train_trades):
        return None, None

    panel = train_panel.copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    panel["asset"] = panel["asset"].astype(str)
    lookup = panel.set_index(["timestamp", "asset"], drop=False)
    close = panel.pivot(index="timestamp", columns="asset", values="close").sort_index().ffill()
    dates = close.index.to_list()

    one_way = _one_way_cost_return(cfg)
    x_rows: list[dict[str, float]] = []
    y_rows: list[int] = []
    net_ret_rows: list[float] = []
    edge_rows: list[float] = []
    signal_ts_rows: list[pd.Timestamp] = []
    for _, tr in train_trades.iterrows():
        asset = str(tr["asset"])
        try:
            sidx = int(tr["start_idx"])
            eidx = int(tr["end_idx"])
        except Exception:
            continue
        if sidx < 0 or eidx >= len(dates) or sidx >= eidx or asset not in close.columns:
            continue
        signal_ts = pd.to_datetime(tr["signal_timestamp"], utc=True, errors="coerce")
        if not pd.notna(signal_ts):
            continue
        key = (signal_ts, asset)
        if key not in lookup.index:
            continue
        row = lookup.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        sign = float(tr["signal_sign"])
        entry_px = float(close.loc[dates[sidx], asset])
        exit_px = float(close.loc[dates[eidx], asset])
        if not (np.isfinite(entry_px) and np.isfinite(exit_px) and entry_px > 0.0):
            continue
        gross_ret = sign * (exit_px / entry_px - 1.0)
        net_ret = gross_ret - 2.0 * one_way
        strength = float(tr["signal_strength"])
        conf = float(tr["confidence_score"])
        edge_proxy = float(strength * conf)
        x_rows.append(
            _trade_feature_row(
                row,
                trend_sign=sign,
                strength=strength,
                conf=conf,
                edge_proxy=edge_proxy,
            )
        )
        y_rows.append(1 if net_ret > 0.0 else 0)
        net_ret_rows.append(float(net_ret))
        edge_rows.append(float(edge_proxy))
        signal_ts_rows.append(pd.Timestamp(signal_ts))

    if len(x_rows) < int(cfg.trade_filter_min_train_trades):
        return None, None
    y = np.asarray(y_rows, dtype=int)
    if len(np.unique(y)) < 2:
        return None, None

    x = pd.DataFrame(x_rows)
    for c in TRADE_FILTER_FEATURES:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return None, None
    keep_idx = x.index.to_numpy(dtype=int)
    y = y[keep_idx]
    net_ret_arr = np.asarray(net_ret_rows, dtype=float)[keep_idx]
    edge_arr = np.asarray(edge_rows, dtype=float)[keep_idx]
    ts_arr = pd.to_datetime(pd.Series(signal_ts_rows), utc=True, errors="coerce").iloc[keep_idx].reset_index(drop=True)

    sample_weight = np.ones(len(y), dtype=float)
    if exp.use_trade_filter_recency_weighting and len(y) > 0:
        age_days = np.zeros(len(y), dtype=float)
        if not ts_arr.empty and ts_arr.notna().any():
            latest_ts = ts_arr.max()
            age_days = (
                (latest_ts - ts_arr).dt.total_seconds().fillna(0.0).to_numpy(dtype=float) / 86400.0
            )
            age_days = np.clip(age_days, 0.0, None)
        rec_half = max(1.0, float(cfg.trade_filter_recency_half_life_days))
        recency_w = np.power(0.5, age_days / rec_half)
        edge_abs = np.abs(edge_arr)
        edge_scale = float(np.nanpercentile(edge_abs, 80.0)) if edge_abs.size else 1.0
        if (not np.isfinite(edge_scale)) or edge_scale <= 1e-8:
            edge_scale = 1.0
        edge_w = 1.0 + float(cfg.trade_filter_edge_boost) * np.clip(edge_abs / edge_scale, 0.0, 2.0)
        sample_weight = recency_w * edge_w
        sample_weight = np.where(np.isfinite(sample_weight), sample_weight, 1.0)
        sample_weight = np.clip(sample_weight, 0.10, 10.0)
        sample_weight = sample_weight / max(1e-6, float(sample_weight.mean()))

    model = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_depth=5,
        max_iter=420,
        min_samples_leaf=24,
        l2_regularization=0.04,
        random_state=42,
    )
    feature_cols = list(TRADE_FILTER_FEATURES)
    model.fit(x[feature_cols], y, sample_weight=sample_weight)
    prob = model.predict_proba(x[feature_cols])[:, 1]
    q = (
        float(exp.trade_filter_prob_quantile_override)
        if exp.trade_filter_prob_quantile_override is not None
        else float(cfg.trade_filter_prob_quantile)
    )
    threshold = _select_trade_filter_threshold(
        prob=np.asarray(prob, dtype=float),
        net_ret=np.asarray(net_ret_arr, dtype=float),
        cfg=cfg,
        q=q,
        use_adaptive=exp.use_trade_filter_adaptive_threshold,
    )
    if not np.isfinite(threshold):
        threshold = 0.50
    return model, threshold


def _build_trades(
    test_panel: pd.DataFrame,
    cfg: Config,
    exp: ExperimentSpec,
    fold_id: str,
    train_spy_vol_median: float,
    asset_efficacy_map: dict[str, float] | None = None,
    asset_efficacy_threshold: float | None = None,
    conf_cal_slope: float = 0.0,
    conf_cal_intercept: float = 0.0,
    trade_filter_model: HistGradientBoostingClassifier | None = None,
    trade_filter_threshold: float | None = None,
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
        rejected_by_filter: list[dict[str, object]] = []
        for _, r in daily.iterrows():
            asset = str(r["asset"])
            rank = float(r["rank_pct"])
            rel = float(r["rel_strength_63"])
            if rank < top_q or rel < rel_thresh:
                continue
            cf_rank = float(r["adv_rank_pct"])
            if exp.use_cashflow_quartile_restrictor:
                cf_q = (
                    float(exp.cashflow_quartile_threshold_override)
                    if exp.cashflow_quartile_threshold_override is not None
                    else float(cfg.cashflow_quartile_rank_threshold)
                )
                if cf_rank < float(np.clip(cf_q, 0.0, 1.0)):
                    continue
            if exp.use_liquidity_filter and cf_rank < cfg.smart_liquidity_min_rank_pct:
                continue
            if exp.use_asset_efficacy_filter and asset_efficacy_map:
                eff = float(asset_efficacy_map.get(asset, float("nan")))
                if (not np.isfinite(eff)) or (
                    asset_efficacy_threshold is not None and eff < float(asset_efficacy_threshold)
                ):
                    continue
            trend_sign = float(np.sign(float(r["past_return"])))
            if not np.isfinite(trend_sign) or trend_sign == 0.0:
                continue
            if (not exp.allow_shorts) and trend_sign < 0:
                continue
            spy_up_now = float(r["spy_up"]) > 0.5
            if exp.require_spy_up_for_longs and trend_sign > 0 and not spy_up_now:
                continue
            if exp.require_spy_down_for_shorts and trend_sign < 0 and spy_up_now:
                continue
            if exp.use_asymmetric_short_filter and trend_sign < 0:
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
                if rank < short_top_q or rel < short_rel_th:
                    continue
            if sig_idx <= int(last_end_by_asset.get(asset, -1)):
                continue
            # Cooldown disabled per latest rule set.
            if False and exp.use_reentry_cooldown and sig_idx <= int(cooldown_until_by_asset.get(asset, -1)):
                continue
            if cfg.require_spy_3m_positive and float(r.get("spy_3m_positive", 0.0)) <= 0.0:
                continue
            if exp.use_confidence_calibration:
                rank_conf = float(np.clip((rank - top_q) / max(1e-6, 1.0 - top_q), 0.0, 1.0))
                rel_conf = float(np.clip((rel - rel_thresh) / 0.05, 0.0, 1.0))
                vol = float(r["vol_21"])
                vol_norm = float(np.clip((vol / max(1e-6, train_spy_vol_median)) - 1.0, 0.0, 2.0))
                base_signal = (
                    cfg.confidence_calib_rel_weight * rel_conf
                    + cfg.confidence_calib_rank_weight * rank_conf
                    - cfg.confidence_calib_vol_penalty * vol_norm
                )
                conf = float(np.clip(cfg.confidence_calib_base + cfg.confidence_calib_scale * base_signal, 0.25, 3.0))
                # Fold-local calibration learned on train fold only.
                edge_hat = float(conf_cal_slope * conf + conf_cal_intercept)
                conf = float(np.clip(conf * (1.0 + float(np.clip(edge_hat / 0.005, -0.4, 0.6))), 0.25, 3.0))
            else:
                rank_conf = float(np.clip((rank - top_q) / max(1e-6, 1.0 - top_q), 0.0, 1.0))
                rel_conf = float(np.clip((rel - rel_thresh) / 0.05, 0.0, 1.0))
                conf = float(0.5 + 1.5 * (0.5 * rank_conf + 0.5 * rel_conf))
            strength = float(max(1e-6, rank - top_q))
            edge_proxy = strength * conf
            if exp.use_soft_cashflow_weighting:
                cf_floor = (
                    float(exp.cashflow_soft_floor_override)
                    if exp.cashflow_soft_floor_override is not None
                    else float(cfg.cashflow_soft_rank_floor)
                )
                cf_min_scale = (
                    float(exp.cashflow_soft_min_scale_override)
                    if exp.cashflow_soft_min_scale_override is not None
                    else float(cfg.cashflow_soft_min_scale)
                )
                cf_floor = float(np.clip(cf_floor, 0.0, 0.99))
                cf_min_scale = float(np.clip(cf_min_scale, 0.25, 1.0))
                cf_norm = float(np.clip((cf_rank - cf_floor) / max(1e-6, 1.0 - cf_floor), 0.0, 1.0))
                cf_scale = float(cf_min_scale + (1.0 - cf_min_scale) * cf_norm)
                conf = float(np.clip(conf * cf_scale, 0.25, 3.0))
                edge_proxy = strength * conf
            trade_feature = _trade_feature_row(
                r,
                trend_sign=trend_sign,
                strength=strength,
                conf=conf,
                edge_proxy=edge_proxy,
            )
            if exp.use_regime_buffer or exp.use_dynamic_edge_floor:
                spy_vol = float(r["spy_vol_21"])
                stressed = (not spy_up_now) or (
                    np.isfinite(spy_vol) and np.isfinite(train_spy_vol_median) and spy_vol > train_spy_vol_median
                )
                if exp.use_dynamic_edge_floor:
                    req = cfg.edge_floor_stressed if stressed else cfg.edge_floor_base
                else:
                    base_req = (
                        float(exp.custom_base_edge_threshold)
                        if exp.use_custom_edge_threshold and exp.custom_base_edge_threshold is not None
                        else cfg.regime_buffer_base_edge
                    )
                    add_req = (
                        float(exp.custom_additional_edge_threshold)
                        if exp.use_custom_edge_threshold and exp.custom_additional_edge_threshold is not None
                        else cfg.regime_buffer_additional
                    )
                    req = base_req + (add_req if stressed else 0.0)
                if edge_proxy < req:
                    continue
            prob_keep = float("nan")
            if trade_filter_model is not None and trade_filter_threshold is not None:
                feat = pd.DataFrame([trade_feature], columns=list(TRADE_FILTER_FEATURES))
                prob_keep = float(trade_filter_model.predict_proba(feat)[0, 1])
                if not np.isfinite(prob_keep):
                    prob_keep = 0.50
                if exp.use_trade_filter_hard_gate and prob_keep < float(trade_filter_threshold):
                    rejected_by_filter.append(
                        {
                            "asset": asset,
                            "row": r,
                            "trend_sign": trend_sign,
                            "confidence": conf,
                            "strength": strength,
                            "priority": edge_proxy,
                            "trade_filter_prob": prob_keep,
                        }
                    )
                    continue
                if exp.use_trade_filter_soft_weighting:
                    pivot = float(trade_filter_threshold) if np.isfinite(float(trade_filter_threshold)) else 0.50
                    prob_scale = float(
                        np.clip(
                            (prob_keep - pivot) / max(1e-6, 1.0 - pivot),
                            float(cfg.trade_filter_soft_scale),
                            1.0,
                        )
                    )
                    conf = float(np.clip(conf * prob_scale, 0.25, 3.0))
                    edge_proxy = strength * conf
            ml_score = float(prob_keep * edge_proxy) if np.isfinite(prob_keep) else float(edge_proxy)
            cands.append(
                {
                    "asset": asset,
                    "row": r,
                    "trend_sign": trend_sign,
                    "confidence": conf,
                    "strength": strength,
                    "priority": edge_proxy,
                    "trade_filter_prob": prob_keep,
                    "ml_score": ml_score,
                }
            )

        if exp.trade_filter_backfill_fraction_override is not None and rejected_by_filter:
            pre_filter_count = len(cands) + len(rejected_by_filter)
            target_count = int(
                max(
                    len(cands),
                    round(pre_filter_count * float(np.clip(exp.trade_filter_backfill_fraction_override, 0.0, 1.0))),
                )
            )
            if target_count > len(cands):
                rejected_by_filter.sort(key=lambda x: float(x.get("priority", 0.0)), reverse=True)
                cands.extend(rejected_by_filter[: max(0, target_count - len(cands))])

        if not cands:
            continue
        if exp.use_ml_topk_selector:
            topk = (
                int(exp.ml_topk_per_day_override)
                if exp.ml_topk_per_day_override is not None
                else int(cfg.quality_cap_max_new_entries_per_day)
            )
            topk = max(1, topk)
            cands.sort(key=lambda x: float(x.get("ml_score", x["priority"])), reverse=True)
            cands = cands[:topk]
        else:
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
            lock_active = False
            bear_trail = cfg.bear_trailing_stop_pct
            hard_stop_loss_local = (
                float(exp.hard_stop_loss_pct_override)
                if exp.hard_stop_loss_pct_override is not None
                else float(cfg.hard_stop_loss_pct)
            )
            profit_lock_trigger = (
                float(exp.profit_lock_threshold_override)
                if exp.profit_lock_threshold_override is not None
                else float(cfg.profit_lock_trigger)
            )
            profit_lock_trail = (
                float(exp.profit_lock_trail_override)
                if exp.profit_lock_trail_override is not None
                else float(cfg.profit_lock_trail)
            )
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
                if exp.use_profit_lock_exit and (not lock_active) and peak_signed >= profit_lock_trigger:
                    lock_active = True
                if exp.use_hard_stop_loss and signed_ret <= -hard_stop_loss_local:
                    exit_idx = j
                    exit_reason = "hard_stop_loss"
                    break
                if lock_active and (peak_signed - signed_ret) >= profit_lock_trail:
                    exit_idx = j
                    exit_reason = "profit_lock_trail"
                    break
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
                    "entry_px": float(entry_px),
                    "exit_reason": exit_reason,
                    "trade_filter_prob": float(c.get("trade_filter_prob", float("nan"))),
                }
            )
            last_end_by_asset[asset] = exit_idx
            if False and exp.use_reentry_cooldown:
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
    cluster_cap_override: float | None = None,
    gross_target: float | None = None,
    max_abs_weight_per_asset: float | None = None,
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
    gross_target_local = float(gross_target) if gross_target is not None else cfg.gross_target
    max_abs_weight_local = (
        float(max_abs_weight_per_asset)
        if max_abs_weight_per_asset is not None
        else cfg.max_abs_weight_per_asset
    )
    by_asset["weight"] = (by_asset["raw_signed"] / denom) * gross_target_local
    by_asset["weight"] = by_asset["weight"].clip(-max_abs_weight_local, max_abs_weight_local)
    for _, r in by_asset.iterrows():
        name = str(r["asset"])
        if name in w.index:
            w.loc[name] = float(r["weight"])
    if use_cluster_caps and cluster_map is not None:
        cap_abs = float(cluster_cap_override) if cluster_cap_override is not None else cfg.cluster_abs_weight_cap
        w = _apply_cluster_caps(w, cluster_map=cluster_map, cap_abs=cap_abs)
    return w


def _apply_short_budget(weights: pd.Series, short_fraction: float) -> pd.Series:
    """Shrink short book to target gross fraction of long book."""
    out = weights.copy()
    if out.empty:
        return out
    sf = float(np.clip(short_fraction, 0.0, 1.0))
    long_gross = float(out[out > 0.0].sum())
    short_gross = float(np.abs(out[out < 0.0].sum()))
    if long_gross <= 1e-12 or short_gross <= 1e-12:
        return out
    max_short = sf * long_gross
    if short_gross > max_short and max_short >= 0.0:
        scale = max_short / short_gross if short_gross > 1e-12 else 1.0
        out.loc[out < 0.0] = out.loc[out < 0.0] * scale
    return out


def _cross_sectional_abs_corr(active: pd.DataFrame, returns_frame: pd.DataFrame, idx: int, lookback: int) -> float:
    if active.empty or idx <= 1 or returns_frame.empty:
        return 0.0
    assets = [str(a) for a in active["asset"].dropna().astype(str).unique().tolist()]
    if len(assets) < 2:
        return 0.0
    hist = returns_frame.reindex(columns=assets).iloc[max(1, idx - lookback) : idx].copy()
    hist = hist.dropna(axis=1, how="any")
    if hist.shape[1] < 2 or hist.shape[0] < max(10, lookback // 3):
        return 0.0
    corr = hist.corr().to_numpy(dtype=float)
    if corr.shape[0] < 2:
        return 0.0
    tri = np.triu_indices(corr.shape[0], k=1)
    vals = np.abs(corr[tri])
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size > 0 else 0.0


def _apply_short_gross_fraction(weights: pd.Series, short_fraction: float) -> pd.Series:
    out = weights.copy()
    sf = float(np.clip(short_fraction, 0.0, 1.0))
    if out.empty:
        return out
    long_mask = out > 0.0
    short_mask = out < 0.0
    long_gross = float(out[long_mask].sum())
    short_gross = float(np.abs(out[short_mask]).sum())
    if short_gross <= 1e-12:
        return out
    max_short = sf * long_gross if long_gross > 1e-12 else 0.0
    if short_gross <= max_short + 1e-12:
        return out
    scale = max_short / short_gross if short_gross > 0.0 else 0.0
    out.loc[short_mask] = out.loc[short_mask] * scale
    return out


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
    spy_drawdown = spy / spy.cummax() - 1.0

    gross_target_local = (
        float(exp.gross_target_override) if exp.gross_target_override is not None else cfg.gross_target
    )
    max_abs_weight_local = (
        float(exp.max_abs_weight_per_asset_override)
        if exp.max_abs_weight_per_asset_override is not None
        else cfg.max_abs_weight_per_asset
    )
    portfolio_vol_target_local = (
        float(exp.portfolio_vol_target_annual_override)
        if exp.portfolio_vol_target_annual_override is not None
        else cfg.portfolio_vol_target_annual
    )

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
        market_stop_active = False
        market_stop_breach_count = 0
        rows: list[dict[str, float | str]] = []
        hist_net: list[float] = []
        hist_strat: list[float] = []
        hist_spy: list[float] = []
        for i in range(1, len(dates)):
            active = t[(t["start_idx"] <= i) & (t["end_idx"] >= i)]
            curr_spy_up = bool(float(spy_up.iloc[i]) > 0.5) if np.isfinite(spy_up.iloc[i]) else False
            curr_spy_vol = float(spy_vol21.iloc[i]) if np.isfinite(spy_vol21.iloc[i]) else float("nan")
            stressed = (not curr_spy_up) or (
                np.isfinite(curr_spy_vol) and np.isfinite(train_spy_vol_median) and curr_spy_vol > train_spy_vol_median
            )
            cluster_cap_local: float | None = None
            if exp.use_dynamic_cluster_caps:
                cluster_cap_local = (
                    float(cfg.dynamic_cluster_abs_weight_cap_stressed)
                    if stressed
                    else float(cfg.dynamic_cluster_abs_weight_cap_calm)
                )
            target_w = _weights_from_active(
                active=active,
                cfg=cfg,
                assets=assets,
                cluster_map=cluster_map,
                use_cluster_caps=exp.use_cluster_caps,
                cluster_cap_override=cluster_cap_local,
                gross_target=gross_target_local,
                max_abs_weight_per_asset=max_abs_weight_local,
            )
            if exp.use_turnover_smoothing:
                w = (1.0 - cfg.turnover_smoothing_lambda) * target_w + cfg.turnover_smoothing_lambda * prev_w
            else:
                w = target_w

            if exp.use_dynamic_gross_target:
                stressed_scale = (
                    float(exp.stressed_gross_scale_override)
                    if exp.stressed_gross_scale_override is not None
                    else float(cfg.stressed_gross_scale)
                )
                calm_scale = (
                    float(exp.calm_gross_scale_override)
                    if exp.calm_gross_scale_override is not None
                    else float(cfg.calm_gross_scale)
                )
                gross_scale = stressed_scale if stressed else calm_scale
                w = w * float(gross_scale)
                w = w.clip(-max_abs_weight_local, max_abs_weight_local)

            if exp.use_portfolio_vol_target:
                if len(hist_net) >= cfg.portfolio_vol_lookback_days:
                    rv = float(np.std(hist_net[-cfg.portfolio_vol_lookback_days :], ddof=1)) * math.sqrt(252.0)
                    if np.isfinite(rv) and rv > 1e-9:
                        scale = float(
                            np.clip(
                                portfolio_vol_target_local / rv,
                                cfg.portfolio_vol_scale_min,
                                cfg.portfolio_vol_scale_max,
                            )
                        )
                        w = w * scale
                # Re-apply per-asset cap after scaling.
                w = w.clip(-max_abs_weight_local, max_abs_weight_local)
            curr_spy_dd = float(spy_drawdown.iloc[i]) if np.isfinite(spy_drawdown.iloc[i]) else float("nan")
            if exp.use_market_stop_loss:
                stop_dd = (
                    float(exp.market_stop_drawdown_override)
                    if exp.market_stop_drawdown_override is not None
                    else float(cfg.market_stop_drawdown)
                )
                rec_dd = (
                    float(exp.market_stop_recovery_override)
                    if exp.market_stop_recovery_override is not None
                    else float(cfg.market_stop_recovery_drawdown)
                )
                stop_scale = (
                    float(exp.market_stop_scale_override)
                    if exp.market_stop_scale_override is not None
                    else float(cfg.market_stop_scale)
                )
                trigger_days = (
                    int(exp.market_stop_trigger_days_override)
                    if exp.market_stop_trigger_days_override is not None
                    else int(cfg.market_stop_trigger_days)
                )
                stop_dd = float(max(1e-6, stop_dd))
                if exp.use_market_stop_vol_adjustment and np.isfinite(curr_spy_vol) and np.isfinite(train_spy_vol_median):
                    vol_ratio = float(train_spy_vol_median / max(1e-8, curr_spy_vol))
                    vol_adj = float(np.clip(vol_ratio, cfg.market_stop_vol_adj_min, cfg.market_stop_vol_adj_max))
                    stop_dd = float(stop_dd * vol_adj)
                    rec_dd = float(rec_dd * vol_adj)
                rec_dd = float(np.clip(rec_dd, 0.0, stop_dd))
                trigger_days = max(1, trigger_days)
                if not market_stop_active:
                    if np.isfinite(curr_spy_dd) and curr_spy_dd <= -stop_dd:
                        market_stop_breach_count += 1
                    else:
                        market_stop_breach_count = 0
                    if market_stop_breach_count >= trigger_days:
                        market_stop_active = True
                elif market_stop_active and np.isfinite(curr_spy_dd) and curr_spy_dd >= -rec_dd:
                    market_stop_active = False
                    market_stop_breach_count = 0
                if market_stop_active:
                    w = w * float(np.clip(stop_scale, 0.0, 1.0))

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
                    "spy_drawdown": curr_spy_dd,
                    "market_stop_active": float(1.0 if market_stop_active else 0.0),
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


def _rolling_corr_scalar(ret_wide: pd.DataFrame, end_idx: int, lookback: int) -> float:
    if ret_wide.empty:
        return float("nan")
    lb = max(10, int(lookback))
    if end_idx <= 1:
        return float("nan")
    start_idx = max(0, end_idx - lb + 1)
    sl = ret_wide.iloc[start_idx : end_idx + 1]
    if sl.shape[0] < 10 or sl.shape[1] < 2:
        return float("nan")
    corr = sl.corr().replace([np.inf, -np.inf], np.nan)
    if corr.empty:
        return float("nan")
    vals = corr.to_numpy(dtype=float)
    mask = np.triu(np.ones(vals.shape, dtype=bool), k=1)
    pair = vals[mask]
    pair = pair[np.isfinite(pair)]
    if pair.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pair)))


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
    out_root = Path("trading_research/examples/_output/46_rel2_ml_trade_filter_walkforward")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    requested_universe = _load_broad_us_universe(max_assets=cfg.max_universe_assets)
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(list(requested_universe), period=cfg.period, interval=cfg.interval)
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
        base_efficacy_map, base_efficacy_thr = _compute_asset_efficacy(
            train_panel=train,
            horizon_days=cfg.asset_efficacy_horizon_days,
            min_obs=cfg.asset_efficacy_min_obs,
            quantile_threshold=cfg.asset_efficacy_quantile,
        )

        for exp in EXPERIMENTS:
            eff_q = (
                float(exp.asset_efficacy_quantile_override)
                if exp.asset_efficacy_quantile_override is not None
                else cfg.asset_efficacy_quantile
            )
            if exp.use_asset_efficacy_filter and exp.asset_efficacy_quantile_override is not None:
                asset_efficacy_map, asset_efficacy_thr = _compute_asset_efficacy(
                    train_panel=train,
                    horizon_days=cfg.asset_efficacy_horizon_days,
                    min_obs=cfg.asset_efficacy_min_obs,
                    quantile_threshold=eff_q,
                )
            else:
                asset_efficacy_map, asset_efficacy_thr = base_efficacy_map, base_efficacy_thr
            conf_cal_slope, conf_cal_intercept = (0.0, 0.0)
            if exp.use_confidence_calibration:
                conf_cal_slope, conf_cal_intercept = _build_confidence_calibrator(train)
            trade_filter_model, trade_filter_threshold = (None, None)
            if exp.use_trade_filter_model:
                trade_filter_model, trade_filter_threshold = _fit_trade_filter_model(
                    train_panel=train,
                    cfg=cfg,
                    exp=exp,
                    train_spy_vol_median=train_spy_vol_median,
                    asset_efficacy_map=asset_efficacy_map,
                    asset_efficacy_threshold=asset_efficacy_thr,
                    conf_cal_slope=conf_cal_slope,
                    conf_cal_intercept=conf_cal_intercept,
                )
            trades = _build_trades(
                test_panel=test,
                cfg=cfg,
                exp=exp,
                fold_id=fold_id,
                train_spy_vol_median=train_spy_vol_median,
                asset_efficacy_map=asset_efficacy_map,
                asset_efficacy_threshold=asset_efficacy_thr,
                conf_cal_slope=conf_cal_slope,
                conf_cal_intercept=conf_cal_intercept,
                trade_filter_model=trade_filter_model,
                trade_filter_threshold=trade_filter_threshold,
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

    base = overall_df[overall_df["strategy"] == "smart_breadth_quality_3x"]
    champion = overall_df[overall_df["strategy"] == "smart_breadth_quality_3x_ml_champion"]
    if champion.empty and not overall_df.empty:
        champion = overall_df[overall_df["strategy"].astype(str).str.contains("ml_champion", na=False)]
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
        "universe_size_requested": len(requested_universe),
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
        "champion_strategy": (
            champion.iloc[0].to_dict() if not champion.empty else None
        ),
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

