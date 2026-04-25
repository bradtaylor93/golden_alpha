#!/usr/bin/env python3
"""
Main orchestration script for the biotech catalyst research project.

Run this to execute the full pipeline:
    python run_analysis.py

Or import individual modules for interactive analysis.
"""

import logging
import json
import sys
from pathlib import Path

import pandas as pd

from config import DATA_DIR, OUTPUT_DIR
from data_sources import (
    fetch_trials_with_results,
    enrich_trials_with_tickers,
    build_event_price_dataset,
    load_pdufa_dates,
)
from price_analysis import (
    classify_moves,
    compute_summary_stats,
    plot_return_distributions,
    plot_pre_vs_post_scatter,
)
from trade_feasibility import (
    assess_liquidity,
    compute_strategy_returns,
    strategy_summary,
)
from false_positive_reduction import (
    compute_z_scores,
    volume_filter,
    compute_signal_score,
    summarize_by_signal_quality,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_full_pipeline(
    max_trials: int = 500,
    use_cached: bool = True,
) -> dict:
    """
    Execute the full research pipeline and return all results.

    Steps:
        1. Fetch or load clinical trial data
        2. Map sponsors to tickers
        3. Fetch price data around events
        4. Analyze pre/post price moves
        5. Assess trade feasibility
        6. Score signal quality for false positive reduction
    """
    results = {}

    # ---------------------------------------------------------------
    # Step 1: Get trial data
    # ---------------------------------------------------------------
    trials_cache = DATA_DIR / "trials_raw.csv"

    if use_cached and trials_cache.exists():
        logger.info("Loading cached trial data from %s", trials_cache)
        trials = pd.read_csv(trials_cache, parse_dates=[
            "primary_completion_date", "completion_date",
            "results_first_post_date", "study_first_post_date",
        ])
    else:
        logger.info("Fetching trial data from ClinicalTrials.gov ...")
        trials = fetch_trials_with_results(max_trials=max_trials)
        if not trials.empty:
            trials.to_csv(trials_cache, index=False)
            logger.info("Cached %d trials to %s", len(trials), trials_cache)

    if trials.empty:
        logger.error("No trial data available. Exiting.")
        return results

    results["n_trials_raw"] = len(trials)
    logger.info("Raw trials: %d", len(trials))

    # ---------------------------------------------------------------
    # Step 2: Map to tickers
    # ---------------------------------------------------------------
    trials = enrich_trials_with_tickers(trials)
    matched = trials.dropna(subset=["ticker"])
    results["n_trials_with_ticker"] = len(matched)
    logger.info("Trials with ticker match: %d / %d", len(matched), len(trials))

    # Save the enriched dataset
    matched.to_csv(DATA_DIR / "trials_enriched.csv", index=False)

    # ---------------------------------------------------------------
    # Step 3: Build event-price dataset
    # ---------------------------------------------------------------
    price_cache = DATA_DIR / "event_prices.csv"

    if use_cached and price_cache.exists():
        logger.info("Loading cached price data from %s", price_cache)
        event_prices = pd.read_csv(price_cache, parse_dates=[
            "event_date", "nearest_trading_date",
        ])
    else:
        logger.info("Fetching price data for %d events ...", len(matched))
        event_prices = build_event_price_dataset(matched)
        if not event_prices.empty:
            event_prices.to_csv(price_cache, index=False)

    if event_prices.empty:
        logger.error("No price data available. Exiting.")
        return results

    results["n_events_with_prices"] = len(event_prices)
    logger.info("Events with price data: %d", len(event_prices))

    # ---------------------------------------------------------------
    # Step 4: Analyze price moves
    # ---------------------------------------------------------------
    logger.info("Analyzing price moves ...")
    classified = classify_moves(event_prices, threshold_pct=5.0)
    stats = compute_summary_stats(classified)
    results["price_analysis"] = {
        k: v.to_string() if isinstance(v, (pd.DataFrame, pd.Series)) else v
        for k, v in stats.items()
    }

    plot_return_distributions(classified)
    plot_pre_vs_post_scatter(classified)

    # ---------------------------------------------------------------
    # Step 5: Trade feasibility
    # ---------------------------------------------------------------
    logger.info("Assessing trade feasibility ...")
    liquidity = assess_liquidity(event_prices)
    results["liquidity_summary"] = {
        "n_tickers": len(liquidity),
        "n_tradable": int(liquidity["tradable"].sum()) if "tradable" in liquidity.columns else 0,
        "pct_tradable": round(
            liquidity["tradable"].mean() * 100, 1
        ) if "tradable" in liquidity.columns else 0,
    }

    strat = compute_strategy_returns(classified, entry_window=5, exit_window=1)
    if not strat.empty:
        results["strategy_5d_1d"] = strategy_summary(strat)

    strat_10_3 = compute_strategy_returns(classified, entry_window=10, exit_window=3)
    if not strat_10_3.empty:
        results["strategy_10d_3d"] = strategy_summary(strat_10_3)

    # ---------------------------------------------------------------
    # Step 6: False positive reduction
    # ---------------------------------------------------------------
    logger.info("Computing false positive reduction scores ...")
    scored = compute_z_scores(classified)
    scored = volume_filter(scored)
    scored = compute_signal_score(scored)

    quality_summary = summarize_by_signal_quality(scored)
    if not quality_summary.empty:
        results["signal_quality_breakdown"] = quality_summary.to_string()

    # Save scored dataset
    scored.to_csv(OUTPUT_DIR / "scored_events.csv", index=False)

    # ---------------------------------------------------------------
    # Output summary
    # ---------------------------------------------------------------
    summary_path = OUTPUT_DIR / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Full summary saved to %s", summary_path)

    print_summary(results)
    return results


def print_summary(results: dict) -> None:
    """Pretty-print key findings."""
    print("\n" + "=" * 70)
    print("BIOTECH CATALYST RESEARCH – SUMMARY")
    print("=" * 70)

    print(f"\n  Raw trials fetched:           {results.get('n_trials_raw', 'N/A')}")
    print(f"  Trials with ticker match:     {results.get('n_trials_with_ticker', 'N/A')}")
    print(f"  Events with price data:       {results.get('n_events_with_prices', 'N/A')}")

    liq = results.get("liquidity_summary", {})
    print(f"\n  Tradable tickers:             {liq.get('n_tradable', 'N/A')} / {liq.get('n_tickers', 'N/A')}"
          f" ({liq.get('pct_tradable', 'N/A')}%)")

    for key in ["strategy_5d_1d", "strategy_10d_3d"]:
        strat = results.get(key, {})
        if strat:
            print(f"\n  Strategy ({key}):")
            print(f"    Trades:      {strat.get('n_trades', 'N/A')}")
            print(f"    Win rate:    {strat.get('win_rate_pct', 'N/A')}%")
            print(f"    Mean return: {strat.get('mean_return_pct', 'N/A')}%")
            print(f"    Sharpe/trade:{strat.get('sharpe_per_trade', 'N/A')}")

    pa = results.get("price_analysis", {})
    antic = pa.get("anticipation_stats")
    if antic and isinstance(antic, dict):
        print(f"\n  Pre-event anticipation:")
        print(f"    Events analyzed:                  {antic.get('n_events', 'N/A')}")
        print(f"    Same direction (pre/post):        {antic.get('pct_same_direction', 'N/A')}%")
        print(f"    Big pre-move + same dir:          {antic.get('pct_pre_big_and_same_dir', 'N/A')}%")
    elif antic:
        print(f"\n  Pre-event anticipation:")
        print(f"    {antic}")

    print("\n" + "=" * 70)
    print(f"  Full results: {OUTPUT_DIR / 'analysis_summary.json'}")
    print(f"  Scored events: {OUTPUT_DIR / 'scored_events.csv'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    max_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    run_full_pipeline(max_trials=max_trials)
