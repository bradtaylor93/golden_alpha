"""
Module 5: Catalyst importance grading.

Grades each catalyst event by its likely impact on the stock price,
using ONLY information knowable BEFORE the result is announced.

This avoids look-ahead bias — we never use the actual outcome.

Grading dimensions (all pre-event knowable):

1. TRIAL DESIGN RIGOR
   - Phase 3 > Phase 2 (more definitive, binary outcome)
   - Randomized > single-arm (more credible, harder to dismiss)
   - Double-blind > open-label (less prone to bias)
   - Large enrollment > small (more statistical power, more conviction)
   - Placebo-controlled > active comparator

2. THERAPEUTIC AREA SIGNIFICANCE
   - Oncology and rare disease trials move stocks more
   - First-in-class mechanisms > me-too drugs
   - Unmet medical need (few/no approved therapies)

3. PIPELINE CONCENTRATION
   - Single/lead asset companies: result IS the company
   - Diversified pipeline: one trial among many
   (Proxy: market cap — smaller companies are more concentrated)

4. REGULATORY PROXIMITY
   - FDA-regulated > non-FDA (US market is largest)
   - PDUFA date set = approval decision imminent
   - Phase 3 pivotal = the definitive trial

5. MARKET ATTENTION
   - Enrollment size as proxy for trial scale/visibility
   - Number of primary outcomes (fewer = cleaner read)
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def grade_trial_design(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score trial design rigor (0-4 points).
    Higher = more rigorous = more definitive result = larger expected move.
    """
    df = df.copy()
    score = pd.Series(0.0, index=df.index)

    # Phase: 3 gets 1 point, 2 gets 0
    if "phase" in df.columns:
        score += df["phase"].str.contains("PHASE3", case=False, na=False).astype(float)

    # Randomized: 1 point
    if "allocation" in df.columns:
        score += df["allocation"].str.contains("RANDOM", case=False, na=False).astype(float)

    # Double-blind: 1 point
    if "masking" in df.columns:
        score += df["masking"].isin(["DOUBLE", "TRIPLE", "QUADRUPLE"]).astype(float)

    # Large enrollment (>200): 1 point
    if "enrollment" in df.columns:
        enrollment = pd.to_numeric(df["enrollment"], errors="coerce")
        score += (enrollment > 200).astype(float)

    df["design_score"] = score
    return df


def grade_therapeutic_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score therapeutic area significance (0-3 points).
    Oncology and rare disease catalysts historically produce larger moves.
    """
    df = df.copy()
    score = pd.Series(0.0, index=df.index)

    if "condition_category" in df.columns:
        cats = df["condition_category"].fillna("")
        score += cats.str.contains("oncology", case=False).astype(float) * 1.5
        score += cats.str.contains("rare_disease", case=False).astype(float) * 1.5
        score += cats.str.contains("neurology", case=False).astype(float) * 1.0
        score += cats.str.contains("immunology", case=False).astype(float) * 0.5

    # Cap at 3
    df["therapeutic_score"] = score.clip(upper=3.0)
    return df


def grade_pipeline_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score pipeline concentration (0-3 points).
    Smaller market cap = more concentrated = bigger expected move.
    Uses event_close as proxy for size if market cap not available.
    """
    df = df.copy()
    score = pd.Series(1.0, index=df.index)  # default middle score

    if "event_close" in df.columns:
        price = pd.to_numeric(df["event_close"], errors="coerce")
        # Low-priced biotech (<$20) often micro-cap, single-asset
        score = np.where(price < 10, 3.0,
                np.where(price < 30, 2.0,
                np.where(price < 100, 1.0, 0.5)))

    df["concentration_score"] = score
    return df


def grade_regulatory_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score regulatory proximity (0-2 points).
    FDA-regulated trials with PDUFA dates are the most impactful.
    """
    df = df.copy()
    score = pd.Series(0.0, index=df.index)

    if "is_fda_regulated" in df.columns:
        score += df["is_fda_regulated"].fillna(False).astype(float)

    if "nct_id" in df.columns:
        score += df["nct_id"].str.startswith("PDUFA_", na=False).astype(float)

    if "primary_purpose" in df.columns:
        score += df["primary_purpose"].isin(["TREATMENT", "PREVENTION"]).astype(float) * 0.5

    df["regulatory_score"] = score.clip(upper=2.0)
    return df


def grade_outcome_clarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score outcome clarity (0-2 points).
    Fewer primary outcomes = cleaner binary read = bigger expected move.
    """
    df = df.copy()
    score = pd.Series(1.0, index=df.index)

    if "n_primary_outcomes" in df.columns:
        n = pd.to_numeric(df["n_primary_outcomes"], errors="coerce").fillna(1)
        score = np.where(n == 1, 2.0,
                np.where(n == 2, 1.5,
                np.where(n <= 3, 1.0, 0.5)))

    df["clarity_score"] = score
    return df


def compute_catalyst_grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite catalyst importance grade.

    Total score range: 0-14 points
    Grades:
        A (>=10): Major catalyst — expect large price move
        B (7-9.9): Significant catalyst
        C (4-6.9): Moderate catalyst
        D (<4):    Minor catalyst — small expected move
    """
    df = grade_trial_design(df)
    df = grade_therapeutic_area(df)
    df = grade_pipeline_concentration(df)
    df = grade_regulatory_proximity(df)
    df = grade_outcome_clarity(df)

    component_cols = [
        "design_score", "therapeutic_score", "concentration_score",
        "regulatory_score", "clarity_score",
    ]
    existing = [c for c in component_cols if c in df.columns]
    df["catalyst_score"] = df[existing].sum(axis=1)

    df["catalyst_grade"] = pd.cut(
        df["catalyst_score"],
        bins=[-0.1, 4, 7, 10, 20],
        labels=["D", "C", "B", "A"],
    )

    return df


def analyze_grades_vs_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare price behavior across catalyst grades.
    Shows whether higher-graded catalysts produce larger moves.
    """
    if "catalyst_grade" not in df.columns:
        return pd.DataFrame()

    agg_cols = {}
    for col in ["pre_5d_return", "post_1d_return", "post_5d_return", "pre_10d_return"]:
        if col in df.columns:
            agg_cols[col] = ["count", "mean", "median", "std"]

    if "volume_ratio_event_day" in df.columns:
        agg_cols["volume_ratio_event_day"] = ["mean", "median"]

    if not agg_cols:
        return pd.DataFrame()

    result = df.groupby("catalyst_grade", observed=True).agg(agg_cols)
    return result


def analyze_grades_vs_abs_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare absolute move magnitude across grades.
    The key question: do higher-graded catalysts produce BIGGER moves
    (regardless of direction)?
    """
    if "catalyst_grade" not in df.columns:
        return pd.DataFrame()

    records = []
    for grade in ["D", "C", "B", "A"]:
        sub = df[df["catalyst_grade"] == grade]
        if sub.empty:
            continue

        rec = {"grade": grade, "n_events": len(sub)}

        for col in ["pre_5d_return", "post_1d_return", "post_5d_return"]:
            if col in sub.columns:
                vals = sub[col].dropna()
                rec[f"{col}_abs_mean"] = round(vals.abs().mean() * 100, 2)
                rec[f"{col}_abs_median"] = round(vals.abs().median() * 100, 2)
                rec[f"{col}_pct_gt5"] = round((vals.abs() > 0.05).mean() * 100, 1)

        if "volume_ratio_event_day" in sub.columns:
            rec["avg_vol_ratio"] = round(
                sub["volume_ratio_event_day"].dropna().mean(), 2
            )

        records.append(rec)

    return pd.DataFrame(records)


def analyze_conditional_by_grade(
    events_df: pd.DataFrame,
    signal_window: int = 5,
    entry_window: int = 3,
    exit_window: int = 1,
    signal_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Run the conditional momentum strategy separately for each catalyst grade
    to see if grading improves strategy selection.
    """
    from trade_feasibility import compute_conditional_strategy, strategy_summary

    if "catalyst_grade" not in events_df.columns:
        return pd.DataFrame()

    records = []
    for grade in ["D", "C", "B", "A"]:
        sub = events_df[events_df["catalyst_grade"] == grade]
        if len(sub) < 5:
            continue

        strat = compute_conditional_strategy(
            sub,
            signal_window=signal_window,
            entry_window=entry_window,
            exit_window=exit_window,
            signal_threshold=signal_threshold,
            cost_bps=30.0,
        )
        if strat.empty:
            continue

        summ = strategy_summary(strat)
        summ["grade"] = grade
        summ["n_source_events"] = len(sub)
        records.append(summ)

    return pd.DataFrame(records)
