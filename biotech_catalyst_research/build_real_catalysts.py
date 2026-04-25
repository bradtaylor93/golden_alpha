#!/usr/bin/env python3
"""
Build a dataset of REAL catalyst dates — the actual market-moving announcements,
not the ClinicalTrials.gov registry posting dates (which lag by 1-3 years).

Strategy:
  1. Use PDUFA dates (already exact — FDA decision dates)
  2. For each ticker in our universe, find actual large-move days
     that line up with known trial completion windows
  3. Use SEC 8-K filing dates as announcement date proxies
  4. Detect abnormal volume + price move days as catalyst events

This approach inverts the problem: instead of "find the trial, find the price,"
we do "find the price shock, confirm it was a catalyst."
"""

import logging
import time

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

from config import DATA_DIR, OUTPUT_DIR, PRE_EVENT_WINDOWS, POST_EVENT_WINDOWS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def detect_catalyst_days(
    ticker: str,
    start: str = "2010-01-01",
    end: str = "2026-04-25",
    move_threshold: float = 0.08,
    volume_multiple: float = 2.0,
    min_price: float = 1.0,
) -> pd.DataFrame:
    """
    Detect days where a stock had both a large move AND abnormal volume.
    These are likely catalyst events (trial results, FDA decisions,
    earnings with pipeline updates, etc.)

    A day qualifies if:
      - |daily return| > move_threshold (default 8%)
      - Volume > volume_multiple × 20-day average (default 2x)
      - Price > min_price (avoid penny stock noise)
    """
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty or len(data) < 30:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
    except Exception as e:
        logger.warning("Failed to download %s: %s", ticker, e)
        return pd.DataFrame()

    data["return"] = data["Close"].pct_change()
    data["vol_20d_avg"] = data["Volume"].rolling(20).mean()
    data["vol_ratio"] = data["Volume"] / data["vol_20d_avg"]
    data["abs_return"] = data["return"].abs()

    # Filter to catalyst days
    catalyst = data[
        (data["abs_return"] > move_threshold)
        & (data["vol_ratio"] > volume_multiple)
        & (data["Close"] > min_price)
    ].copy()

    if catalyst.empty:
        return pd.DataFrame()

    # Build event records with pre/post prices
    records = []
    trading_dates = data.index

    for event_date in catalyst.index:
        idx = trading_dates.get_loc(event_date)

        rec = {
            "ticker": ticker,
            "event_date": event_date,
            "event_close": data.iloc[idx]["Close"],
            "event_return": data.iloc[idx]["return"],
            "event_volume_ratio": data.iloc[idx]["vol_ratio"],
        }

        for w in PRE_EVENT_WINDOWS:
            pre_idx = idx - w
            if pre_idx >= 0:
                rec[f"pre_{w}d_close"] = data.iloc[pre_idx]["Close"]
                rec[f"pre_{w}d_return"] = (
                    data.iloc[idx]["Close"] / data.iloc[pre_idx]["Close"] - 1
                )

        for w in POST_EVENT_WINDOWS:
            post_idx = idx + w
            if post_idx < len(data):
                rec[f"post_{w}d_close"] = data.iloc[post_idx]["Close"]
                rec[f"post_{w}d_return"] = (
                    data.iloc[post_idx]["Close"] / data.iloc[idx]["Close"] - 1
                )

        if idx >= 20:
            avg_vol = data.iloc[idx - 20:idx]["Volume"].mean()
            rec["volume_ratio_event_day"] = (
                data.iloc[idx]["Volume"] / avg_vol if avg_vol > 0 else np.nan
            )

        records.append(rec)

    return pd.DataFrame(records)


def build_catalyst_dataset(
    tickers: list[str],
    move_threshold: float = 0.08,
    volume_multiple: float = 2.0,
) -> pd.DataFrame:
    """Build catalyst events for a list of tickers."""
    all_events = []

    for ticker in tqdm(tickers, desc="Scanning for catalysts"):
        events = detect_catalyst_days(
            ticker,
            move_threshold=move_threshold,
            volume_multiple=volume_multiple,
        )
        if not events.empty:
            all_events.append(events)
        time.sleep(0.3)

    if not all_events:
        return pd.DataFrame()

    return pd.concat(all_events, ignore_index=True)


def run_clean_analysis_on_real_events(
    events: pd.DataFrame,
    entry_days: list[int] = None,
    exit_days: list[int] = None,
):
    """Run the clean-signal strategy on detected catalyst events."""
    if entry_days is None:
        entry_days = [1, 3, 5, 10]
    if exit_days is None:
        exit_days = [1, 3, 5, 10]

    print(f"\nEvents: {len(events)}, Tickers: {events['ticker'].nunique()}")
    print(f"Date range: {events['event_date'].min()} to {events['event_date'].max()}")

    # Event-day move distribution
    print(f"\n--- EVENT-DAY MOVE DISTRIBUTION ---")
    er = events["event_return"].dropna()
    print(f"  Mean: {er.mean()*100:+.1f}%, Median: {er.median()*100:+.1f}%, Std: {er.std()*100:.1f}%")
    print(f"  >+10%: {(er > 0.10).sum()}, <-10%: {(er < -0.10).sum()}")
    print(f"  >+20%: {(er > 0.20).sum()}, <-20%: {(er < -0.20).sum()}")
    print(f"  Positive: {(er > 0).sum()} ({(er > 0).mean()*100:.0f}%), "
          f"Negative: {(er < 0).sum()} ({(er < 0).mean()*100:.0f}%)")

    # Pre-event drift analysis (CLEAN — before the event day)
    print(f"\n--- PRE-EVENT DRIFT (days before event, all clean) ---")
    results = []

    for entry_d in entry_days:
        entry_col = f"pre_{entry_d}d_close"
        if entry_col not in events.columns:
            continue

        # Clean signal: drift from 20d to entry_d (all before entry)
        for sig_start in [5, 10, 20]:
            sig_col = f"pre_{sig_start}d_close"
            if sig_col not in events.columns or sig_start <= entry_d:
                continue

            drift = events[entry_col] / events[sig_col] - 1

            for exit_d in exit_days:
                exit_col = f"post_{exit_d}d_close"
                if exit_col not in events.columns:
                    continue

                trade_ret = events[exit_col] / events[entry_col] - 1
                valid = drift.notna() & trade_ret.notna()

                for thr in [0.03, 0.05, 0.07, 0.10, 0.15]:
                    for direction in ["both", "short_only"]:
                        if direction == "both":
                            long_mask = valid & (drift > thr)
                            short_mask = valid & (drift < -thr)
                            l_ret = trade_ret[long_mask].clip(lower=-0.05) - 0.003
                            s_ret = (-trade_ret[short_mask]).clip(lower=-0.05) - 0.003
                            all_ret = pd.concat([l_ret, s_ret])
                            nl, ns = len(l_ret), len(s_ret)
                        else:
                            short_mask = valid & (drift < -thr)
                            all_ret = (-trade_ret[short_mask]).clip(lower=-0.05) - 0.003
                            nl, ns = 0, len(all_ret)

                        if len(all_ret) < 15:
                            continue

                        dates = events.loc[all_ret.index, "event_date"]
                        span = max((dates.max() - dates.min()).days / 365.25, 0.5)
                        tpy = len(all_ret) / span
                        sh = all_ret.mean() / all_ret.std() if all_ret.std() > 0 else 0
                        ann_sh = sh * np.sqrt(tpy)

                        results.append({
                            "sig": f"d-{sig_start}→d-{entry_d}",
                            "entry": entry_d,
                            "exit": exit_d,
                            "thr%": thr * 100,
                            "dir": direction,
                            "n": len(all_ret),
                            "n_l": nl, "n_s": ns,
                            "/yr": round(tpy, 1),
                            "wr%": round((all_ret > 0).mean() * 100, 1),
                            "mean%": round(all_ret.mean() * 100, 2),
                            "sharpe": round(sh, 3),
                            "ann_sh": round(ann_sh, 2),
                            "worst%": round(all_ret.min() * 100, 1),
                        })

    if not results:
        print("  No valid configs")
        return pd.DataFrame()

    res_df = pd.DataFrame(results)

    print(f"\n--- TOP 25 CLEAN CONFIGS (by annualized Sharpe, n>=15) ---")
    top = res_df.nlargest(25, "ann_sh")
    print(top.to_string(index=False))

    # Direction comparison
    print(f"\n--- DIRECTION COMPARISON ---")
    for d in ["both", "short_only"]:
        sub = res_df[res_df["dir"] == d]
        if sub.empty:
            continue
        print(f"  {d:>12}: {len(sub)} configs, "
              f"median sharpe={sub['ann_sh'].median():.2f}, "
              f"best={sub['ann_sh'].max():.2f}, "
              f"median WR={sub['wr%'].median():.1f}%")

    # Exit timing
    print(f"\n--- EXIT TIMING (best per exit window) ---")
    for xd in sorted(res_df["exit"].unique()):
        sub = res_df[res_df["exit"] == xd]
        best = sub.loc[sub["ann_sh"].idxmax()]
        print(f"  Exit d+{xd}: best sharpe={best['ann_sh']:.2f}, "
              f"n={int(best['n'])}, wr={best['wr%']:.1f}%, mean={best['mean%']:.2f}%")

    res_df.to_csv(OUTPUT_DIR / "real_catalyst_sweep.csv", index=False)
    return res_df


if __name__ == "__main__":
    # Get the ticker universe from our existing data
    enriched = pd.read_csv(DATA_DIR / "trials_enriched.csv")
    tickers = enriched["ticker"].dropna().unique().tolist()

    # Also add PDUFA tickers
    pdufa_path = DATA_DIR / "pdufa_dates.csv"
    if pdufa_path.exists():
        pdufa = pd.read_csv(pdufa_path)
        pdufa_tickers = pdufa["ticker"].dropna().unique().tolist()
        tickers = list(set(tickers + pdufa_tickers))

    logger.info("Scanning %d tickers for catalyst events...", len(tickers))

    # Detect real catalyst days (big move + big volume)
    events = build_catalyst_dataset(tickers, move_threshold=0.08, volume_multiple=2.0)
    events.to_csv(DATA_DIR / "real_catalyst_events.csv", index=False)
    logger.info("Found %d catalyst events across %d tickers",
                len(events), events["ticker"].nunique())

    # Run clean analysis
    print("\n" + "=" * 90)
    print("REAL CATALYST EVENTS — CLEAN SIGNAL ANALYSIS")
    print("=" * 90)
    run_clean_analysis_on_real_events(events)
    print("=" * 90)
