"""
Module 2: Price move analysis around known catalyst dates.

Answers: "What % of pre-announced catalysts show abnormal price moves
before and after the event?"
"""

import logging

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    PRE_EVENT_WINDOWS,
    POST_EVENT_WINDOWS,
    VOLUME_SPIKE_THRESHOLD,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


def classify_moves(
    df: pd.DataFrame,
    threshold_pct: float = 5.0,
) -> pd.DataFrame:
    """
    Classify each event by the magnitude of pre- and post-event moves.

    Categories for each window:
        big_up    : return > +threshold
        small_up  : 0 < return <= +threshold
        flat      : return == 0 (unlikely with decimals)
        small_down: -threshold <= return < 0
        big_down  : return < -threshold
    """
    df = df.copy()

    def _classify(val, thr):
        if pd.isna(val):
            return np.nan
        if val > thr / 100:
            return "big_up"
        elif val > 0:
            return "small_up"
        elif val < -thr / 100:
            return "big_down"
        elif val < 0:
            return "small_down"
        return "flat"

    for w in PRE_EVENT_WINDOWS:
        col = f"pre_{w}d_return"
        if col in df.columns:
            df[f"pre_{w}d_class"] = df[col].apply(lambda v: _classify(v, threshold_pct))

    for w in POST_EVENT_WINDOWS:
        col = f"post_{w}d_return"
        if col in df.columns:
            df[f"post_{w}d_class"] = df[col].apply(lambda v: _classify(v, threshold_pct))

    return df


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics across all events.

    Returns a dict of DataFrames keyed by analysis type.
    """
    results = {}

    # --- Pre-event return distributions ---
    pre_cols = [c for c in df.columns if c.startswith("pre_") and c.endswith("_return")]
    if pre_cols:
        pre_stats = df[pre_cols].describe().T
        pre_stats["pct_positive"] = (df[pre_cols] > 0).mean() * 100
        pre_stats["pct_gt_5pct"] = (df[pre_cols] > 0.05).mean() * 100
        pre_stats["pct_gt_10pct"] = (df[pre_cols] > 0.10).mean() * 100
        pre_stats["pct_lt_neg5pct"] = (df[pre_cols] < -0.05).mean() * 100
        results["pre_event_stats"] = pre_stats

    # --- Post-event return distributions ---
    post_cols = [c for c in df.columns if c.startswith("post_") and c.endswith("_return")]
    if post_cols:
        post_stats = df[post_cols].describe().T
        post_stats["pct_positive"] = (df[post_cols] > 0).mean() * 100
        post_stats["pct_gt_5pct"] = (df[post_cols] > 0.05).mean() * 100
        post_stats["pct_gt_10pct"] = (df[post_cols] > 0.10).mean() * 100
        post_stats["pct_lt_neg5pct"] = (df[post_cols] < -0.05).mean() * 100
        results["post_event_stats"] = post_stats

    # --- Pre vs Post correlation ---
    if pre_cols and post_cols:
        corr_pairs = {}
        for pre_c in pre_cols:
            for post_c in post_cols:
                mask = df[[pre_c, post_c]].dropna().index
                if len(mask) > 10:
                    corr_pairs[f"{pre_c} vs {post_c}"] = df.loc[mask, pre_c].corr(
                        df.loc[mask, post_c]
                    )
        results["pre_post_correlation"] = pd.Series(corr_pairs)

    # --- Volume spike analysis ---
    if "volume_ratio_event_day" in df.columns:
        vol = df["volume_ratio_event_day"].dropna()
        results["volume_spike_stats"] = pd.Series({
            "mean_volume_ratio": vol.mean(),
            "median_volume_ratio": vol.median(),
            "pct_above_2x": (vol > VOLUME_SPIKE_THRESHOLD).mean() * 100,
            "pct_above_3x": (vol > 3.0).mean() * 100,
            "pct_above_5x": (vol > 5.0).mean() * 100,
        })

    # --- Leaked vs non-leaked classification ---
    # A rough heuristic: if the pre-event move (5d) is >5% in the same
    # direction as the post-event move (1d), it may indicate information
    # leakage or anticipation.
    if "pre_5d_return" in df.columns and "post_1d_return" in df.columns:
        mask = df[["pre_5d_return", "post_1d_return"]].dropna().index
        sub = df.loc[mask]
        same_dir = (sub["pre_5d_return"] * sub["post_1d_return"] > 0)
        pre_big = sub["pre_5d_return"].abs() > 0.05
        results["anticipation_stats"] = pd.Series({
            "n_events": len(sub),
            "pct_same_direction": same_dir.mean() * 100,
            "pct_pre_big_and_same_dir": (same_dir & pre_big).mean() * 100,
            "n_potential_leaked": (same_dir & pre_big).sum(),
        })

    return results


def plot_return_distributions(df: pd.DataFrame, save: bool = True) -> None:
    """Generate distribution plots for pre/post event returns."""
    return_cols = [c for c in df.columns if c.endswith("_return")]
    if not return_cols:
        logger.warning("No return columns to plot")
        return

    n = len(return_cols)
    fig, axes = plt.subplots(
        (n + 2) // 3, 3, figsize=(15, 4 * ((n + 2) // 3))
    )
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(return_cols):
        ax = axes[i]
        data = df[col].dropna() * 100
        ax.hist(data, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        ax.axvline(data.median(), color="orange", linestyle="-", alpha=0.7)
        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Return (%)")
        ax.set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "return_distributions.png"
        plt.savefig(path, dpi=150)
        logger.info("Saved return distributions to %s", path)
    plt.close()


def plot_pre_vs_post_scatter(
    df: pd.DataFrame,
    pre_window: int = 5,
    post_window: int = 1,
    save: bool = True,
) -> None:
    """Scatter plot of pre-event vs post-event returns."""
    pre_col = f"pre_{pre_window}d_return"
    post_col = f"post_{post_window}d_return"

    if pre_col not in df.columns or post_col not in df.columns:
        return

    sub = df[[pre_col, post_col]].dropna() * 100

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(sub[pre_col], sub[post_col], alpha=0.4, s=20, c="steelblue")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(f"Pre-event {pre_window}d return (%)")
    ax.set_ylabel(f"Post-event {post_window}d return (%)")
    ax.set_title(f"Pre ({pre_window}d) vs Post ({post_window}d) Returns")

    # Quadrant counts
    q1 = ((sub[pre_col] > 0) & (sub[post_col] > 0)).sum()
    q2 = ((sub[pre_col] < 0) & (sub[post_col] > 0)).sum()
    q3 = ((sub[pre_col] < 0) & (sub[post_col] < 0)).sum()
    q4 = ((sub[pre_col] > 0) & (sub[post_col] < 0)).sum()
    total = len(sub)

    ax.text(0.95, 0.95, f"Q1 (↑↑): {q1} ({q1/total*100:.0f}%)",
            transform=ax.transAxes, ha="right", fontsize=9)
    ax.text(0.05, 0.95, f"Q2 (↓↑): {q2} ({q2/total*100:.0f}%)",
            transform=ax.transAxes, ha="left", fontsize=9)
    ax.text(0.05, 0.05, f"Q3 (↓↓): {q3} ({q3/total*100:.0f}%)",
            transform=ax.transAxes, ha="left", fontsize=9)
    ax.text(0.95, 0.05, f"Q4 (↑↓): {q4} ({q4/total*100:.0f}%)",
            transform=ax.transAxes, ha="right", fontsize=9)

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / f"pre{pre_window}d_vs_post{post_window}d_scatter.png"
        plt.savefig(path, dpi=150)
        logger.info("Saved scatter plot to %s", path)
    plt.close()
