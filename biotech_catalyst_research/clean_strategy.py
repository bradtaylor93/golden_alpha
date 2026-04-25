#!/usr/bin/env python3
"""
Clean-signal strategy: no look-ahead bias.

All signals use ONLY information available at or before entry time.
Focus is on:
  1. Event selection (which catalysts to trade)
  2. Trade structure (entry/exit/direction)
  3. Risk management (sizing, stops)

Key constraint: the signal at entry (day -N) can only use prices
up to day -N, plus pre-knowable trial metadata.
"""

import logging
import sys

import pandas as pd
import numpy as np

from config import DATA_DIR, OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_prepare() -> pd.DataFrame:
    """Load events and merge trial metadata. All clean — no post-entry data used."""
    events = pd.read_csv(
        OUTPUT_DIR / "scored_events.csv",
        parse_dates=["event_date", "nearest_trading_date"],
    )
    trials = pd.read_csv(DATA_DIR / "trials_raw.csv")

    meta_cols = ["nct_id", "phase", "enrollment", "allocation", "masking",
                 "primary_purpose", "n_primary_outcomes", "condition_category",
                 "n_arms", "intervention_types"]
    meta = trials[[c for c in meta_cols if c in trials.columns]].drop_duplicates(subset=["nct_id"])
    df = events.merge(meta, on="nct_id", how="left")

    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    df["is_pdufa"] = df["nct_id"].str.startswith("PDUFA_", na=False)
    df["is_phase3"] = df["phase"].str.contains("PHASE3", case=False, na=False)
    df["is_randomized"] = df["allocation"] == "RANDOMIZED"
    df["is_blinded"] = df["masking"].isin(["DOUBLE", "TRIPLE", "QUADRUPLE"])
    df["is_oncology"] = df["condition_category"].str.contains("oncology", case=False, na=False)
    df["is_rare"] = df["condition_category"].str.contains("rare", case=False, na=False)
    df["is_neuro"] = df["condition_category"].str.contains("neurology", case=False, na=False)
    df["is_small_stock"] = df["event_close"] < 20
    df["is_micro_stock"] = df["event_close"] < 5
    df["large_trial"] = df["enrollment"] > 200

    return df


# ======================================================================
# CLEAN SIGNAL DEFINITIONS
# All computed using only data available at entry time
# ======================================================================

def compute_clean_signals(df: pd.DataFrame, entry_day: int = 3) -> pd.DataFrame:
    """
    Compute signals using ONLY prices available at entry (day -entry_day).
    No event_close, no post-entry data.
    """
    df = df.copy()

    # Drift: return from day -20 to day -entry (all before entry)
    if f"pre_20d_close" in df.columns and f"pre_{entry_day}d_close" in df.columns:
        df["drift_20_to_entry"] = df[f"pre_{entry_day}d_close"] / df["pre_20d_close"] - 1

    if f"pre_10d_close" in df.columns and f"pre_{entry_day}d_close" in df.columns:
        df["drift_10_to_entry"] = df[f"pre_{entry_day}d_close"] / df["pre_10d_close"] - 1

    if f"pre_5d_close" in df.columns and f"pre_{entry_day}d_close" in df.columns:
        df["drift_5_to_entry"] = df[f"pre_{entry_day}d_close"] / df["pre_5d_close"] - 1

    # Acceleration: is the drift speeding up?
    # Compare recent drift (5d to entry) vs earlier drift (20d to 5d)
    if "pre_20d_close" in df.columns and "pre_5d_close" in df.columns:
        earlier = df["pre_5d_close"] / df["pre_20d_close"] - 1
        recent = df[f"pre_{entry_day}d_close"] / df["pre_5d_close"] - 1
        df["acceleration"] = recent - earlier

    return df


# ======================================================================
# TRADE OUTCOMES (clean — entry to exit, no signal leakage)
# ======================================================================

def compute_trade_returns(
    df: pd.DataFrame,
    entry_day: int = 3,
    exit_day: int = 1,
) -> pd.DataFrame:
    """Compute actual trade return from entry to exit. No leakage."""
    entry_col = f"pre_{entry_day}d_close"
    exit_col = f"post_{exit_day}d_close"

    df = df.dropna(subset=[entry_col, exit_col]).copy()
    df["raw_long_return"] = df[exit_col] / df[entry_col] - 1
    df["raw_short_return"] = -(df[exit_col] / df[entry_col] - 1)
    return df


# ======================================================================
# STRATEGY EVALUATION
# ======================================================================

def evaluate_strategy(
    df: pd.DataFrame,
    filter_mask: pd.Series,
    signal_col: str,
    long_thr: float,
    short_thr: float,
    tp: float = None,
    sl: float = None,
    cost: float = 0.003,
    label: str = "",
) -> dict:
    """Evaluate a clean strategy on the filtered subset."""
    sub = df[filter_mask].copy()
    if len(sub) < 10:
        return {}

    long_mask = sub[signal_col] > long_thr
    short_mask = sub[signal_col] < -short_thr

    long_returns = sub.loc[long_mask, "raw_long_return"].copy()
    short_returns = sub.loc[short_mask, "raw_short_return"].copy()

    if tp is not None:
        long_returns = long_returns.clip(upper=tp)
        short_returns = short_returns.clip(upper=tp)
    if sl is not None:
        long_returns = long_returns.clip(lower=-sl)
        short_returns = short_returns.clip(lower=-sl)

    long_returns = long_returns - cost
    short_returns = short_returns - cost

    all_returns = pd.concat([long_returns, short_returns])
    if len(all_returns) < 10:
        return {}

    dates = pd.to_datetime(sub.loc[all_returns.index, "event_date"])
    span = max((dates.max() - dates.min()).days / 365.25, 0.5)
    tpy = len(all_returns) / span
    sharpe_t = all_returns.mean() / all_returns.std() if all_returns.std() > 0 else 0
    ann_sharpe = sharpe_t * np.sqrt(tpy)

    return {
        "label": label,
        "n": len(all_returns),
        "n_long": len(long_returns),
        "n_short": len(short_returns),
        "per_yr": round(tpy, 1),
        "wr%": round((all_returns > 0).mean() * 100, 1),
        "mean%": round(all_returns.mean() * 100, 2),
        "med%": round(all_returns.median() * 100, 2),
        "std%": round(all_returns.std() * 100, 2),
        "sharpe_t": round(sharpe_t, 3),
        "ann_sharpe": round(ann_sharpe, 2),
        "best%": round(all_returns.max() * 100, 1),
        "worst%": round(all_returns.min() * 100, 1),
        "l_wr%": round((long_returns > 0).mean() * 100, 1) if len(long_returns) > 0 else None,
        "s_wr%": round((short_returns > 0).mean() * 100, 1) if len(short_returns) > 0 else None,
        "l_mean%": round(long_returns.mean() * 100, 2) if len(long_returns) > 0 else None,
        "s_mean%": round(short_returns.mean() * 100, 2) if len(short_returns) > 0 else None,
    }


def run_analysis(entry_day: int = 3, exit_day: int = 1):
    """Run the full clean-signal analysis."""
    df = load_and_prepare()
    df = compute_clean_signals(df, entry_day=entry_day)
    df = compute_trade_returns(df, entry_day=entry_day, exit_day=exit_day)

    logger.info("Events: %d, with clean signals: %d", len(df),
                df["drift_20_to_entry"].notna().sum())

    all_results = []

    # ------------------------------------------------------------------
    # PART 1: BASELINE — all events, various signals and thresholds
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"CLEAN STRATEGY ANALYSIS (entry d-{entry_day}, exit d+{exit_day})")
    print("=" * 90)

    print("\n--- BASELINES: Signal comparison across all events ---")
    for sig_name, sig_col in [
        ("drift 5→entry (2d)", "drift_5_to_entry"),
        ("drift 10→entry (7d)", "drift_10_to_entry"),
        ("drift 20→entry (17d)", "drift_20_to_entry"),
    ]:
        for thr in [0.02, 0.03, 0.05, 0.07, 0.10, 0.15]:
            if sig_col not in df.columns:
                continue
            mask = df[sig_col].notna()
            r = evaluate_strategy(df, mask, sig_col, thr, thr,
                                  label=f"{sig_name} thr={thr*100:.0f}%")
            if r:
                all_results.append(r)

    if all_results:
        baseline = pd.DataFrame(all_results)
        cols = ["label", "n", "per_yr", "wr%", "mean%", "sharpe_t", "ann_sharpe"]
        print(baseline[cols].to_string(index=False))

    # ------------------------------------------------------------------
    # PART 2: EVENT SELECTION — which events are worth trading?
    # ------------------------------------------------------------------
    print("\n--- EVENT SELECTION: Which catalyst types produce bigger moves? ---")

    # First just look at absolute move sizes by category (no signal needed)
    abs_ret = df["raw_long_return"].abs()
    categories = {
        "All events": df.index.notna(),
        "PDUFA events": df["is_pdufa"],
        "Phase 3": df["is_phase3"],
        "Phase 2 only": ~df["is_phase3"],
        "Randomized": df["is_randomized"].fillna(False),
        "Blinded": df["is_blinded"].fillna(False),
        "Oncology": df["is_oncology"].fillna(False),
        "Rare disease": df["is_rare"].fillna(False),
        "Neurology": df["is_neuro"].fillna(False),
        "Stock <$20": df["is_small_stock"],
        "Stock <$5": df["is_micro_stock"],
        "Enrollment >200": df["large_trial"].fillna(False),
        "Single primary outcome": (df["n_primary_outcomes"] == 1),
    }

    print(f"\n{'Category':>25} {'N':>5} {'Abs move mean%':>15} {'Abs move med%':>14} {'Std%':>7}")
    print("-" * 70)
    for name, mask in categories.items():
        sub = df.loc[mask, "raw_long_return"].dropna()
        if len(sub) < 10:
            continue
        print(f"{name:>25} {len(sub):>5} {sub.abs().mean()*100:>14.2f}% {sub.abs().median()*100:>13.2f}% {sub.std()*100:>6.2f}%")

    # ------------------------------------------------------------------
    # PART 3: CLEAN SIGNAL + EVENT SELECTION COMBINATIONS
    # ------------------------------------------------------------------
    print("\n--- STRATEGY MATRIX: Signal × Event filter × Risk management ---")

    strat_results = []

    sig_col = "drift_20_to_entry"
    if sig_col not in df.columns:
        print("Signal column missing")
        return

    event_filters = {
        "all": df[sig_col].notna(),
        "pdufa_only": df["is_pdufa"] & df[sig_col].notna(),
        "phase3": df["is_phase3"] & df[sig_col].notna(),
        "small_stock": df["is_small_stock"] & df[sig_col].notna(),
        "micro_stock": df["is_micro_stock"] & df[sig_col].notna(),
        "onco+rare": (df["is_oncology"] | df["is_rare"]) & df[sig_col].notna(),
        "blinded_p3": df["is_blinded"].fillna(False) & df["is_phase3"] & df[sig_col].notna(),
        "small+p3": df["is_small_stock"] & df["is_phase3"] & df[sig_col].notna(),
        "small+onco": df["is_small_stock"] & df["is_oncology"] & df[sig_col].notna(),
        "pdufa+small": df["is_pdufa"] & df["is_small_stock"] & df[sig_col].notna(),
        "large_trial+p3": df["large_trial"].fillna(False) & df["is_phase3"] & df[sig_col].notna(),
    }

    thresholds = [0.03, 0.05, 0.07, 0.10, 0.15]
    risk_configs = [
        (None, None, "no_risk_mgmt"),
        (None, 0.05, "SL5%"),
        (None, 0.10, "SL10%"),
        (0.15, 0.05, "TP15_SL5"),
        (0.20, 0.05, "TP20_SL5"),
        (0.20, 0.10, "TP20_SL10"),
        (0.30, 0.10, "TP30_SL10"),
    ]

    # Also try asymmetric: long only and short only
    directions = ["both", "long_only", "short_only"]

    for filt_name, filt_mask in event_filters.items():
        for thr in thresholds:
            for tp, sl, risk_name in risk_configs:
                for direction in directions:
                    if direction == "long_only":
                        lt, st = thr, 999  # never short
                    elif direction == "short_only":
                        lt, st = 999, thr  # never long
                    else:
                        lt, st = thr, thr

                    r = evaluate_strategy(
                        df, filt_mask, sig_col, lt, st,
                        tp=tp, sl=sl,
                        label=f"{filt_name}|{thr*100:.0f}%|{risk_name}|{direction}",
                    )
                    if r:
                        r["filter"] = filt_name
                        r["thr"] = thr * 100
                        r["risk"] = risk_name
                        r["direction"] = direction
                        strat_results.append(r)

    if not strat_results:
        print("No valid strategies found")
        return

    strats = pd.DataFrame(strat_results)
    strats.to_csv(OUTPUT_DIR / "clean_strategy_sweep.csv", index=False)

    # Filter to meaningful sample sizes
    valid = strats[strats["n"] >= 15].copy()

    print(f"\nTotal configs tested: {len(strats)}, valid (n>=15): {len(valid)}")

    # Best by Sharpe
    print(f"\n--- TOP 20 BY ANNUALIZED SHARPE (n>=15) ---")
    top = valid.nlargest(20, "ann_sharpe")
    cols = ["filter", "thr", "risk", "direction", "n", "per_yr",
            "wr%", "mean%", "ann_sharpe", "worst%"]
    print(top[cols].to_string(index=False))

    # Best by win rate
    print(f"\n--- TOP 10 BY WIN RATE (n>=20) ---")
    valid20 = valid[valid["n"] >= 20]
    if not valid20.empty:
        top_wr = valid20.nlargest(10, "wr%")
        print(top_wr[cols].to_string(index=False))

    # Best with risk controls
    print(f"\n--- TOP 10 WITH RISK CONTROLS (TP+SL set, n>=15) ---")
    controlled = valid[(valid["risk"].str.contains("TP")) & (valid["n"] >= 15)]
    if not controlled.empty:
        top_ctrl = controlled.nlargest(10, "ann_sharpe")
        cols_ext = cols + ["l_wr%", "s_wr%", "l_mean%", "s_mean%"]
        print(top_ctrl[[c for c in cols_ext if c in top_ctrl.columns]].to_string(index=False))

    # Long-only vs short-only
    print(f"\n--- LONG ONLY vs SHORT ONLY vs BOTH (aggregated, n>=15) ---")
    for d in ["long_only", "short_only", "both"]:
        sub = valid[valid["direction"] == d]
        if sub.empty:
            continue
        print(f"  {d:>12}: {len(sub)} configs, "
              f"median sharpe={sub['ann_sharpe'].median():.2f}, "
              f"best sharpe={sub['ann_sharpe'].max():.2f}, "
              f"median WR={sub['wr%'].median():.1f}%")

    # Event filter comparison
    print(f"\n--- EVENT FILTER COMPARISON (best config per filter, n>=15) ---")
    for filt_name in event_filters:
        sub = valid[valid["filter"] == filt_name]
        if sub.empty:
            continue
        best = sub.loc[sub["ann_sharpe"].idxmax()]
        print(f"  {filt_name:>18}: n={int(best['n']):>4}, wr={best['wr%']:>5.1f}%, "
              f"mean={best['mean%']:>6.2f}%, sharpe={best['ann_sharpe']:>5.2f} "
              f"({best['direction']}, thr={best['thr']:.0f}%, {best['risk']})")

    # ------------------------------------------------------------------
    # PART 4: ENTRY/EXIT TIMING SWEEP
    # ------------------------------------------------------------------
    print(f"\n--- ENTRY/EXIT TIMING (best config across timing combos) ---")
    timing_results = []
    for ed in [1, 3, 5]:
        for xd in [1, 3, 5, 10]:
            df_t = compute_trade_returns(df, entry_day=ed, exit_day=xd)
            df_t = compute_clean_signals(df_t, entry_day=ed)
            sig = "drift_20_to_entry"
            if sig not in df_t.columns:
                continue
            mask = df_t[sig].notna()
            for thr in [0.05, 0.10]:
                for tp, sl, rn in [(None, 0.05, "SL5"), (0.20, 0.05, "TP20_SL5")]:
                    r = evaluate_strategy(df_t, mask, sig, thr, thr, tp=tp, sl=sl,
                                          label=f"e{ed}x{xd}|{thr*100:.0f}%|{rn}")
                    if r and r["n"] >= 15:
                        r["entry"] = ed
                        r["exit"] = xd
                        timing_results.append(r)

    if timing_results:
        timing = pd.DataFrame(timing_results)
        top_timing = timing.nlargest(15, "ann_sharpe")
        print(top_timing[["label", "entry", "exit", "n", "per_yr",
                          "wr%", "mean%", "ann_sharpe", "worst%"]].to_string(index=False))

    print("\n" + "=" * 90)


if __name__ == "__main__":
    run_analysis()
