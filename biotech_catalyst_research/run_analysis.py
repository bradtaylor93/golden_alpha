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
import yfinance as yf

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
    compute_conditional_strategy,
    conditional_strategy_sweep,
    compute_options_straddle_strategy,
    straddle_strategy_summary,
    straddle_strategy_sweep,
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
    max_trials: int = 1000,
    use_cached: bool = True,
    max_market_cap: float = None,
) -> dict:
    """
    Execute the full research pipeline and return all results.

    Steps:
        1. Fetch or load clinical trial data (Phase 2 + Phase 3)
        2. Map sponsors to tickers
        3. Filter by market cap if max_market_cap is set
        4. Fetch price data around events
        5. Analyze pre/post price moves
        6. Assess trade feasibility
        7. Score signal quality for false positive reduction
    """
    from config import MAX_MARKET_CAP
    if max_market_cap is None:
        max_market_cap = MAX_MARKET_CAP

    results = {}

    # ---------------------------------------------------------------
    # Step 1: Get trial data (Phase 2 and Phase 3)
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
        trials_p3 = fetch_trials_with_results(
            phase="PHASE3", max_trials=max_trials,
        )
        trials_p2 = fetch_trials_with_results(
            phase="PHASE2", max_trials=max_trials,
        )
        trials = pd.concat([trials_p3, trials_p2], ignore_index=True)
        trials = trials.drop_duplicates(subset=["nct_id"])
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

    # ---------------------------------------------------------------
    # Step 2b: Filter by market cap ceiling
    # ---------------------------------------------------------------
    if max_market_cap and max_market_cap < float("inf"):
        logger.info("Filtering to tickers with market cap < $%.0fB ...",
                     max_market_cap / 1e9)
        mktcap_cache = DATA_DIR / "ticker_mktcaps.csv"
        if use_cached and mktcap_cache.exists():
            mktcap_df = pd.read_csv(mktcap_cache)
        else:
            records = []
            for ticker in matched["ticker"].unique():
                try:
                    info = yf.Ticker(ticker).info
                    mc = info.get("marketCap")
                    records.append({"ticker": ticker, "market_cap": mc})
                except Exception:
                    records.append({"ticker": ticker, "market_cap": None})
            mktcap_df = pd.DataFrame(records)
            mktcap_df.to_csv(mktcap_cache, index=False)

        small_tickers = mktcap_df[
            mktcap_df["market_cap"].notna()
            & (mktcap_df["market_cap"] <= max_market_cap)
        ]["ticker"].tolist()
        large_excluded = mktcap_df[
            mktcap_df["market_cap"].notna()
            & (mktcap_df["market_cap"] > max_market_cap)
        ]
        logger.info("Tickers under $%.0fB cap: %d / %d",
                     max_market_cap / 1e9, len(small_tickers),
                     len(mktcap_df))
        if not large_excluded.empty:
            logger.info("Excluded large-cap: %s",
                        ", ".join(large_excluded["ticker"].tolist()))

        matched = matched[matched["ticker"].isin(small_tickers)]
        results["n_trials_under_cap"] = len(matched)
        results["tickers_under_cap"] = small_tickers
        results["tickers_excluded_large"] = large_excluded["ticker"].tolist() if not large_excluded.empty else []

    # Save the enriched dataset
    matched.to_csv(DATA_DIR / "trials_enriched.csv", index=False)

    # ---------------------------------------------------------------
    # Step 2c: Load and merge PDUFA dates
    # ---------------------------------------------------------------
    pdufa_path = DATA_DIR / "pdufa_dates.csv"
    if pdufa_path.exists():
        logger.info("Loading PDUFA dates from %s", pdufa_path)
        pdufa = load_pdufa_dates(str(pdufa_path))
        if not pdufa.empty:
            # Filter to tickers we care about (under cap, if applicable)
            if "tickers_under_cap" in results:
                pdufa = pdufa[pdufa["ticker"].isin(results["tickers_under_cap"])]

            # Remove future/pending entries
            pdufa = pdufa[pdufa["outcome"] != "pending"]

            # Convert to same event format as trial data
            pdufa_events = pdufa.rename(columns={
                "pdufa_date": "results_first_post_date",
            })[["ticker", "drug_name", "results_first_post_date"]].copy()
            pdufa_events["nct_id"] = "PDUFA_" + pdufa_events["drug_name"].str.replace(" ", "_")
            pdufa_events["source"] = "pdufa"
            matched["source"] = "clinicaltrials"

            combined = pd.concat([matched, pdufa_events], ignore_index=True)
            results["n_pdufa_events"] = len(pdufa_events)
            results["n_combined_events"] = len(combined)
            logger.info("Added %d PDUFA events, total: %d",
                         len(pdufa_events), len(combined))
        else:
            combined = matched.copy()
            combined["source"] = "clinicaltrials"
    else:
        combined = matched.copy()
        combined["source"] = "clinicaltrials"

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
        logger.info("Fetching price data for %d events ...", len(combined))
        event_prices = build_event_price_dataset(combined)
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

    # --- 5a: Naive long strategy (baseline) ---
    strat = compute_strategy_returns(classified, entry_window=5, exit_window=1)
    if not strat.empty:
        results["strategy_naive_long_5d_1d"] = strategy_summary(strat)

    # --- 5b: Conditional momentum strategy ---
    logger.info("Running conditional momentum strategy sweep ...")
    cond_sweep = conditional_strategy_sweep(classified, cost_bps=30.0)
    if not cond_sweep.empty:
        cond_sweep.to_csv(OUTPUT_DIR / "conditional_strategy_sweep.csv", index=False)
        best_cond = cond_sweep.loc[cond_sweep["sharpe_per_trade"].idxmax()]
        results["conditional_strategy_best"] = best_cond.to_dict()

        best_params = compute_conditional_strategy(
            classified,
            signal_window=int(best_cond["signal_window"]),
            entry_window=int(best_cond["entry_window"]),
            exit_window=int(best_cond["exit_window"]),
            signal_threshold=best_cond["threshold_pct"] / 100,
            cost_bps=30.0,
        )
        if not best_params.empty:
            results["conditional_strategy_detail"] = strategy_summary(best_params)
            best_params.to_csv(OUTPUT_DIR / "conditional_trades.csv", index=False)

        # Also show the top 5 configurations
        top5 = cond_sweep.nlargest(5, "sharpe_per_trade")[
            ["signal_window", "entry_window", "exit_window", "threshold_pct",
             "n_trades", "win_rate_pct", "mean_return_pct", "sharpe_per_trade",
             "n_long", "n_short"]
        ]
        results["conditional_top5"] = top5.to_string(index=False)

    # --- 5c: Options straddle strategy ---
    logger.info("Running options straddle strategy sweep ...")
    straddle_sweep = straddle_strategy_sweep(classified)
    if not straddle_sweep.empty:
        straddle_sweep.to_csv(OUTPUT_DIR / "straddle_strategy_sweep.csv", index=False)
        best_straddle = straddle_sweep.loc[straddle_sweep["sharpe_per_trade"].idxmax()]
        results["straddle_strategy_best"] = best_straddle.to_dict()

        best_straddle_detail = compute_options_straddle_strategy(
            classified,
            entry_window=int(best_straddle["entry_window"]),
            exit_window=int(best_straddle["exit_window"]),
            implied_vol_annual=best_straddle["implied_vol"],
            vol_crush_pct=best_straddle["vol_crush"],
        )
        if not best_straddle_detail.empty:
            results["straddle_strategy_detail"] = straddle_strategy_summary(best_straddle_detail)

        top5_straddle = straddle_sweep.nlargest(5, "sharpe_per_trade")[
            ["entry_window", "exit_window", "implied_vol", "vol_crush",
             "n_trades", "win_rate_pct", "mean_pnl_pct", "sharpe_per_trade",
             "avg_straddle_cost_pct", "avg_realized_move_pct", "pct_exceeded_breakeven"]
        ]
        results["straddle_top5"] = top5_straddle.to_string(index=False)

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
    if "n_trials_under_cap" in results:
        print(f"  Trials under market cap:      {results['n_trials_under_cap']}")
        print(f"  Tickers under cap:            {results.get('tickers_under_cap', [])}")
        excluded = results.get("tickers_excluded_large", [])
        if excluded:
            print(f"  Excluded (large cap):         {excluded}")
    print(f"  Events with price data:       {results.get('n_events_with_prices', 'N/A')}")

    if "n_pdufa_events" in results:
        print(f"  PDUFA events added:           {results['n_pdufa_events']}")
        print(f"  Combined events:              {results.get('n_combined_events', 'N/A')}")

    liq = results.get("liquidity_summary", {})
    print(f"\n  Tradable tickers:             {liq.get('n_tradable', 'N/A')} / {liq.get('n_tickers', 'N/A')}"
          f" ({liq.get('pct_tradable', 'N/A')}%)")

    # Naive baseline
    naive = results.get("strategy_naive_long_5d_1d", {})
    if naive:
        print(f"\n  STRATEGY 1: Naive Long (buy 5d before, sell 1d after)")
        print(f"    Trades:      {naive.get('n_trades', 'N/A')}")
        print(f"    Win rate:    {naive.get('win_rate_pct', 'N/A')}%")
        print(f"    Mean return: {naive.get('mean_return_pct', 'N/A')}%")
        print(f"    Sharpe/trade:{naive.get('sharpe_per_trade', 'N/A')}")

    # Conditional momentum
    cond = results.get("conditional_strategy_detail", {})
    cond_best = results.get("conditional_strategy_best", {})
    if cond:
        print(f"\n  STRATEGY 2: Conditional Momentum (best config)")
        print(f"    Signal: {cond_best.get('signal_window', '?')}d drift, "
              f"threshold: {cond_best.get('threshold_pct', '?')}%")
        print(f"    Entry: {cond_best.get('entry_window', '?')}d before, "
              f"Exit: {cond_best.get('exit_window', '?')}d after")
        print(f"    Trades:      {cond.get('n_trades', 'N/A')} "
              f"(L:{cond_best.get('n_long','?')} / S:{cond_best.get('n_short','?')})")
        print(f"    Win rate:    {cond.get('win_rate_pct', 'N/A')}%")
        print(f"    Mean return: {cond.get('mean_return_pct', 'N/A')}%")
        print(f"    Sharpe/trade:{cond.get('sharpe_per_trade', 'N/A')}")

    top5_cond = results.get("conditional_top5")
    if top5_cond:
        print(f"\n    Top 5 conditional configs:")
        for line in str(top5_cond).split("\n"):
            print(f"      {line}")

    # Options straddle
    straddle = results.get("straddle_strategy_detail", {})
    straddle_best = results.get("straddle_strategy_best", {})
    if straddle:
        print(f"\n  STRATEGY 3: Options Straddle (best config)")
        print(f"    Entry: {straddle_best.get('entry_window', '?')}d before, "
              f"Exit: {straddle_best.get('exit_window', '?')}d after")
        print(f"    IV: {straddle_best.get('implied_vol', '?')}, "
              f"Vol crush: {straddle_best.get('vol_crush', '?')}")
        print(f"    Trades:            {straddle.get('n_trades', 'N/A')}")
        print(f"    Win rate:          {straddle.get('win_rate_pct', 'N/A')}%")
        print(f"    Mean P&L:          {straddle.get('mean_pnl_pct', 'N/A')}%")
        print(f"    Sharpe/trade:      {straddle.get('sharpe_per_trade', 'N/A')}")
        print(f"    Avg straddle cost: {straddle.get('avg_straddle_cost_pct', 'N/A')}%")
        print(f"    Avg realized move: {straddle.get('avg_realized_move_pct', 'N/A')}%")
        print(f"    Exceeded breakeven:{straddle.get('pct_exceeded_breakeven', 'N/A')}%")

    top5_straddle = results.get("straddle_top5")
    if top5_straddle:
        print(f"\n    Top 5 straddle configs:")
        for line in str(top5_straddle).split("\n"):
            print(f"      {line}")

    print("\n" + "=" * 70)
    print(f"  Full results: {OUTPUT_DIR / 'analysis_summary.json'}")
    print(f"  Scored events: {OUTPUT_DIR / 'scored_events.csv'}")
    print(f"  Strategy sweeps: {OUTPUT_DIR / 'conditional_strategy_sweep.csv'}")
    print(f"                   {OUTPUT_DIR / 'straddle_strategy_sweep.csv'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    max_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run_full_pipeline(max_trials=max_trials)
