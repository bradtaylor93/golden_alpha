#!/usr/bin/env python3
"""
Comprehensive strategy parameter optimization.

Sweeps across all trading parameters to find optimal configuration:
  - Signal threshold (symmetric and asymmetric long/short)
  - Entry timing (how many days before event to enter)
  - Exit timing (how many days after event to exit)
  - Take-profit levels
  - Stop-loss levels
  - Minimum pre-event volume filter
"""

import logging
import sys
from itertools import product

import pandas as pd
import numpy as np

from config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_events() -> pd.DataFrame:
    """Load the scored events dataset."""
    path = OUTPUT_DIR / "scored_events.csv"
    df = pd.read_csv(path, parse_dates=["event_date", "nearest_trading_date"])
    logger.info("Loaded %d events", len(df))
    return df


def _compute_return(row, entry_col, exit_col, signal, tp, sl):
    """
    Compute trade return with optional take-profit and stop-loss.

    TP/SL are applied to the final exit return only (we don't have
    intraday data to simulate intra-period stops precisely).
    """
    entry_price = row[entry_col]
    exit_price = row[exit_col]

    if signal == "long":
        raw = exit_price / entry_price - 1
    else:
        raw = -(exit_price / entry_price - 1)

    if tp is not None and raw >= tp:
        return tp
    if sl is not None and raw <= -sl:
        return -sl

    return raw


def run_full_optimization(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sweep across all parameter combinations.

    Parameters swept:
      - long_threshold: [1%, 2%, 3%, 5%, 7%, 10%]
      - short_threshold: [1%, 2%, 3%, 5%, 7%, 10%]
      - entry_window: [1, 3, 5] days before event
      - exit_window: [1, 3, 5, 10] days after event
      - take_profit: [None, 10%, 15%, 20%, 30%]
      - stop_loss: [None, 5%, 10%, 15%, 20%]
    """
    long_thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    short_thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    entry_windows = [1, 3, 5]
    exit_windows = [1, 3, 5, 10]
    take_profits = [None, 0.10, 0.15, 0.20, 0.30]
    stop_losses = [None, 0.05, 0.10, 0.15, 0.20]

    signal_window = 5
    signal_col = f"pre_{signal_window}d_return"
    cost_bps = 30.0
    cost = cost_bps / 10_000

    if signal_col not in events_df.columns:
        logger.error("Signal column %s not found", signal_col)
        return pd.DataFrame()

    results = []
    logger.info("Sweeping parameter grid...")

    checked = 0
    for lt, st, ew, xw, tp, sl in product(
        long_thresholds, short_thresholds, entry_windows,
        exit_windows, take_profits, stop_losses,
    ):
        # Entry must be strictly inside signal window
        if ew >= signal_window:
            continue

        entry_col = f"pre_{ew}d_close"
        exit_col = f"post_{xw}d_close"

        if entry_col not in events_df.columns or exit_col not in events_df.columns:
            continue

        df = events_df.dropna(subset=[signal_col, entry_col, exit_col, "event_close"]).copy()

        # Classify signals with asymmetric thresholds
        signals = np.where(
            df[signal_col] > lt, "long",
            np.where(df[signal_col] < -st, "short", "skip")
        )
        df["signal"] = signals
        trades = df[df["signal"] != "skip"]

        if len(trades) < 15:
            continue

        # Compute returns with TP/SL
        net_returns = []
        for _, row in trades.iterrows():
            ret = _compute_return(row, entry_col, exit_col, row["signal"], tp, sl)
            net_returns.append(ret - cost)

        net = np.array(net_returns)
        n_long = (trades["signal"] == "long").sum()
        n_short = (trades["signal"] == "short").sum()

        long_mask = trades["signal"].values == "long"
        short_mask = trades["signal"].values == "short"

        mean_ret = net.mean()
        std_ret = net.std()
        if std_ret < 0.001:
            continue  # degenerate — all returns identical (e.g. all hit TP)
        sharpe = mean_ret / std_ret

        dates = pd.to_datetime(trades["event_date"])
        span = max((dates.max() - dates.min()).days / 365.25, 0.5)
        tpy = len(net) / span
        ann_sharpe = sharpe * np.sqrt(tpy)

        rec = {
            "long_thr": lt * 100,
            "short_thr": st * 100,
            "entry_d": ew,
            "exit_d": xw,
            "tp": f"{tp*100:.0f}%" if tp else "none",
            "sl": f"{sl*100:.0f}%" if sl else "none",
            "n_trades": len(net),
            "n_long": int(n_long),
            "n_short": int(n_short),
            "trades_yr": round(tpy, 1),
            "win_rate": round((net > 0).mean() * 100, 1),
            "mean_ret": round(mean_ret * 100, 2),
            "median_ret": round(np.median(net) * 100, 2),
            "std": round(std_ret * 100, 2),
            "sharpe_trade": round(sharpe, 3),
            "ann_sharpe": round(ann_sharpe, 2),
            "best": round(net.max() * 100, 1),
            "worst": round(net.min() * 100, 1),
        }

        # Long-only and short-only stats
        if n_long >= 5:
            long_net = net[long_mask]
            rec["long_wr"] = round((long_net > 0).mean() * 100, 1)
            rec["long_mean"] = round(long_net.mean() * 100, 2)
        if n_short >= 5:
            short_net = net[short_mask]
            rec["short_wr"] = round((short_net > 0).mean() * 100, 1)
            rec["short_mean"] = round(short_net.mean() * 100, 2)

        results.append(rec)
        checked += 1

    logger.info("Evaluated %d valid configs", checked)
    return pd.DataFrame(results)


def print_report(sweep: pd.DataFrame) -> None:
    """Print a clean optimization report."""
    print("\n" + "=" * 90)
    print("STRATEGY PARAMETER OPTIMIZATION REPORT")
    print("=" * 90)

    # --- Best by annualized Sharpe ---
    print("\n--- TOP 15 BY ANNUALIZED SHARPE (min 20 trades) ---")
    valid = sweep[sweep["n_trades"] >= 20]
    top = valid.nlargest(15, "ann_sharpe")
    cols = ["long_thr", "short_thr", "entry_d", "exit_d", "tp", "sl",
            "n_trades", "trades_yr", "win_rate", "mean_ret", "ann_sharpe",
            "worst"]
    print(top[cols].to_string(index=False))

    # --- Best by win rate (min 30 trades) ---
    print("\n--- TOP 10 BY WIN RATE (min 30 trades) ---")
    valid30 = sweep[sweep["n_trades"] >= 30]
    top_wr = valid30.nlargest(10, "win_rate")
    print(top_wr[cols].to_string(index=False))

    # --- Best by mean return (min 20 trades) ---
    print("\n--- TOP 10 BY MEAN RETURN (min 20 trades) ---")
    top_ret = valid.nlargest(10, "mean_ret")
    print(top_ret[cols].to_string(index=False))

    # --- Entry timing analysis ---
    print("\n--- ENTRY TIMING (aggregated across other params) ---")
    for ew in sorted(sweep["entry_d"].unique()):
        sub = sweep[(sweep["entry_d"] == ew) & (sweep["n_trades"] >= 20)]
        if sub.empty:
            continue
        print(f"  Entry {ew}d before: {len(sub)} configs, "
              f"median ann_sharpe={sub['ann_sharpe'].median():.2f}, "
              f"best={sub['ann_sharpe'].max():.2f}")

    # --- Exit timing analysis ---
    print("\n--- EXIT TIMING (aggregated across other params) ---")
    for xw in sorted(sweep["exit_d"].unique()):
        sub = sweep[(sweep["exit_d"] == xw) & (sweep["n_trades"] >= 20)]
        if sub.empty:
            continue
        print(f"  Exit {xw}d after:  {len(sub)} configs, "
              f"median ann_sharpe={sub['ann_sharpe'].median():.2f}, "
              f"best={sub['ann_sharpe'].max():.2f}")

    # --- Symmetric vs asymmetric thresholds ---
    print("\n--- SYMMETRIC vs ASYMMETRIC THRESHOLDS ---")
    sym = sweep[(sweep["long_thr"] == sweep["short_thr"]) & (sweep["n_trades"] >= 20)]
    asym = sweep[(sweep["long_thr"] != sweep["short_thr"]) & (sweep["n_trades"] >= 20)]
    if not sym.empty:
        print(f"  Symmetric:   {len(sym)} configs, "
              f"median sharpe={sym['ann_sharpe'].median():.2f}, "
              f"best={sym['ann_sharpe'].max():.2f}")
    if not asym.empty:
        print(f"  Asymmetric:  {len(asym)} configs, "
              f"median sharpe={asym['ann_sharpe'].median():.2f}, "
              f"best={asym['ann_sharpe'].max():.2f}")
        best_asym = asym.nlargest(5, "ann_sharpe")
        print(f"\n  Best asymmetric configs:")
        print(best_asym[cols].to_string(index=False))

    # --- Take-profit analysis ---
    print("\n--- TAKE-PROFIT IMPACT ---")
    for tp_val in sweep["tp"].unique():
        sub = sweep[(sweep["tp"] == tp_val) & (sweep["n_trades"] >= 20)]
        if sub.empty:
            continue
        print(f"  TP={tp_val:>5s}: {len(sub)} configs, "
              f"median sharpe={sub['ann_sharpe'].median():.2f}, "
              f"best={sub['ann_sharpe'].max():.2f}, "
              f"median worst={sub['worst'].median():.1f}%")

    # --- Stop-loss analysis ---
    print("\n--- STOP-LOSS IMPACT ---")
    for sl_val in sweep["sl"].unique():
        sub = sweep[(sweep["sl"] == sl_val) & (sweep["n_trades"] >= 20)]
        if sub.empty:
            continue
        print(f"  SL={sl_val:>5s}: {len(sub)} configs, "
              f"median sharpe={sub['ann_sharpe'].median():.2f}, "
              f"best={sub['ann_sharpe'].max():.2f}, "
              f"median worst={sub['worst'].median():.1f}%")

    # --- Best risk-controlled config ---
    print("\n--- BEST CONFIG WITH RISK CONTROLS (TP+SL both set, min 25 trades) ---")
    controlled = sweep[
        (sweep["tp"] != "none") & (sweep["sl"] != "none")
        & (sweep["n_trades"] >= 25)
    ]
    if not controlled.empty:
        best_ctrl = controlled.nlargest(5, "ann_sharpe")
        cols_ext = cols + ["long_wr", "long_mean", "short_wr", "short_mean"]
        available = [c for c in cols_ext if c in best_ctrl.columns]
        print(best_ctrl[available].to_string(index=False))

    print("\n" + "=" * 90)


if __name__ == "__main__":
    events = load_events()
    sweep = run_full_optimization(events)
    sweep.to_csv(OUTPUT_DIR / "full_optimization_sweep.csv", index=False)
    logger.info("Saved %d configs to %s", len(sweep),
                OUTPUT_DIR / "full_optimization_sweep.csv")
    print_report(sweep)
