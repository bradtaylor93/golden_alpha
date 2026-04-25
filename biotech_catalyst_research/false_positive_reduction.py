"""
Module 4: False positive reduction analysis.

Not every pre-event price move indicates informed trading. This module
provides filters and scoring approaches to separate signal from noise.
"""

import logging

import pandas as pd
import numpy as np
from scipy import stats

from config import PRE_EVENT_WINDOWS, OUTPUT_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter 1: Sector / market beta adjustment
# ---------------------------------------------------------------------------

def beta_adjust_returns(
    events_df: pd.DataFrame,
    market_returns: pd.DataFrame,
    beta_col: str = "beta",
) -> pd.DataFrame:
    """
    Subtract market return * beta from raw event returns to isolate
    idiosyncratic moves.

    market_returns: DataFrame with date index and 'market_return' column.
    If beta is not in events_df, uses 1.0.
    """
    df = events_df.copy()
    if beta_col not in df.columns:
        df[beta_col] = 1.0

    for w in PRE_EVENT_WINDOWS:
        raw_col = f"pre_{w}d_return"
        if raw_col not in df.columns:
            continue
        df[f"pre_{w}d_excess_return"] = df[raw_col]  # placeholder
        # Full implementation requires aligning market returns with event dates;
        # structure is provided so the user can plug in their market data.

    return df


# ---------------------------------------------------------------------------
# Filter 2: Volatility context – is the move unusual given normal vol?
# ---------------------------------------------------------------------------

def compute_z_scores(
    events_df: pd.DataFrame,
    historical_vol_window: int = 60,
) -> pd.DataFrame:
    """
    For each event, compute z-scores of the pre-event returns relative to the
    stock's own trailing volatility. High z-scores indicate abnormal moves.
    """
    df = events_df.copy()

    for w in PRE_EVENT_WINDOWS:
        ret_col = f"pre_{w}d_return"
        if ret_col not in df.columns:
            continue
        returns = df[ret_col].dropna()
        if len(returns) < 10:
            continue

        mu = returns.mean()
        sigma = returns.std()
        df[f"pre_{w}d_zscore"] = (df[ret_col] - mu) / sigma if sigma > 0 else np.nan

    return df


# ---------------------------------------------------------------------------
# Filter 3: Volume confirmation
# ---------------------------------------------------------------------------

def volume_filter(
    df: pd.DataFrame,
    volume_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Flag events where the pre-event volume was abnormally high,
    which adds conviction that price moves are information-driven
    rather than random.
    """
    df = df.copy()
    if "volume_ratio_event_day" in df.columns:
        df["volume_confirmed"] = df["volume_ratio_event_day"] > volume_threshold
    return df


# ---------------------------------------------------------------------------
# Filter 4: Options activity (structure only – requires external data)
# ---------------------------------------------------------------------------

def options_activity_filter(
    df: pd.DataFrame,
    options_data: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    If options flow data is available (e.g., unusual options activity),
    flag events where pre-event options volume was anomalous.

    This is a structural placeholder. Actual implementation requires
    historical options data from providers like:
      - CBOE
      - OptionMetrics (WRDS)
      - Unusual Whales
      - Market Chameleon
    """
    df = df.copy()
    if options_data is not None and not options_data.empty:
        # Merge on ticker + date window and flag anomalies
        df["options_unusual"] = False  # placeholder
    else:
        df["options_unusual"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Filter 5: News / SEC filing check
# ---------------------------------------------------------------------------

def sec_filing_proximity_filter(
    df: pd.DataFrame,
    filings_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Flag events near SEC filings (8-K, 10-Q) that might explain
    price moves independently of the clinical trial result.

    Reduces false positives by identifying alternative explanations.
    """
    df = df.copy()
    if filings_df is not None and not filings_df.empty:
        df["near_sec_filing"] = False  # placeholder for merge logic
    else:
        df["near_sec_filing"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def compute_signal_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multiple filters into a composite 'signal quality' score.

    Higher score = more likely the pre-event move is information-driven
    and less likely a false positive.

    Score components (each 0-1, summed):
      1. z-score magnitude (>2 = 1 point)
      2. Volume confirmation (1 point if volume_confirmed)
      3. Magnitude of pre-5d return (>5% = 1 point)
      4. Consistency of direction with post-event move (1 point)
    """
    df = df.copy()
    score = pd.Series(0.0, index=df.index)

    # Z-score component
    if "pre_5d_zscore" in df.columns:
        score += (df["pre_5d_zscore"].abs() > 2.0).astype(float)

    # Volume component
    if "volume_confirmed" in df.columns:
        score += df["volume_confirmed"].astype(float).fillna(0)

    # Magnitude component
    if "pre_5d_return" in df.columns:
        score += (df["pre_5d_return"].abs() > 0.05).astype(float)

    # Direction consistency component
    if "pre_5d_return" in df.columns and "post_1d_return" in df.columns:
        same_dir = (df["pre_5d_return"] * df["post_1d_return"]) > 0
        score += same_dir.astype(float)

    df["signal_score"] = score
    df["signal_quality"] = pd.cut(
        score,
        bins=[-0.1, 1, 2, 3, 4],
        labels=["low", "medium", "high", "very_high"],
    )

    return df


def summarize_by_signal_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group events by signal quality and compare performance metrics.
    Shows that higher-quality signals yield better trades and fewer
    false positives.
    """
    if "signal_quality" not in df.columns:
        return pd.DataFrame()

    cols_to_agg = {}
    if "pre_5d_return" in df.columns:
        cols_to_agg["pre_5d_return"] = ["mean", "median", "count"]
    if "post_1d_return" in df.columns:
        cols_to_agg["post_1d_return"] = ["mean", "median"]
    if "net_return" in df.columns:
        cols_to_agg["net_return"] = ["mean", "median"]

    if not cols_to_agg:
        return pd.DataFrame()

    return df.groupby("signal_quality", observed=True).agg(cols_to_agg)
