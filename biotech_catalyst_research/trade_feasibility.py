"""
Module 3: Trade feasibility assessment.

Evaluates whether one could practically trade around known catalyst dates by
examining liquidity, spread costs, borrow availability, and options markets.

Includes:
  - Liquidity assessment
  - Execution cost modeling
  - Naive directional strategy
  - Conditional momentum strategy (long/short based on pre-event drift)
  - Options straddle/strangle strategy
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf

from config import MIN_MARKET_CAP, MIN_ADV_DOLLARS, OUTPUT_DIR

logger = logging.getLogger(__name__)


def assess_liquidity(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    For each unique ticker in the dataset, pull current market data and assess
    tradability metrics:
      - market cap
      - average daily volume (20d)
      - average daily dollar volume
      - whether options are listed
      - whether the stock is shortable (proxy: institutional ownership)
    """
    tickers = df[ticker_col].dropna().unique()
    records = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            logger.warning("Could not fetch info for %s", ticker)
            continue

        mkt_cap = info.get("marketCap", np.nan)
        avg_vol = info.get("averageVolume", np.nan)
        price = info.get("regularMarketPrice") or info.get("previousClose", np.nan)
        adv_dollars = avg_vol * price if not (np.isnan(avg_vol) or np.isnan(price)) else np.nan

        records.append({
            "ticker": ticker,
            "market_cap": mkt_cap,
            "avg_daily_volume": avg_vol,
            "current_price": price,
            "avg_daily_dollar_volume": adv_dollars,
            "meets_mktcap_threshold": mkt_cap >= MIN_MARKET_CAP if not np.isnan(mkt_cap) else False,
            "meets_adv_threshold": adv_dollars >= MIN_ADV_DOLLARS if not np.isnan(adv_dollars) else False,
            "tradable": (
                (mkt_cap >= MIN_MARKET_CAP if not np.isnan(mkt_cap) else False)
                and
                (adv_dollars >= MIN_ADV_DOLLARS if not np.isnan(adv_dollars) else False)
            ),
        })

    return pd.DataFrame(records)


def estimate_execution_costs(
    avg_daily_dollar_volume: float,
    position_size_dollars: float = 100_000,
) -> dict:
    """
    Rough estimate of market impact and execution costs using a
    simplified square-root model.

    Market impact ≈ sigma * sqrt(Q / ADV)
    where sigma is daily volatility (assumed ~3% for biotech)
    and Q/ADV is participation rate.
    """
    if avg_daily_dollar_volume <= 0 or np.isnan(avg_daily_dollar_volume):
        return {"participation_rate": np.nan, "estimated_impact_bps": np.nan}

    sigma_daily = 0.03
    participation_rate = position_size_dollars / avg_daily_dollar_volume

    impact = sigma_daily * np.sqrt(participation_rate)
    impact_bps = impact * 10_000

    return {
        "participation_rate": participation_rate,
        "estimated_impact_bps": round(impact_bps, 1),
        "estimated_impact_pct": round(impact * 100, 3),
    }


def compute_strategy_returns(
    events_df: pd.DataFrame,
    entry_window: int = 5,
    exit_window: int = 1,
    direction: str = "long",
    cost_bps: float = 20.0,
) -> pd.DataFrame:
    """
    Simulate a simple strategy:
      - Enter `entry_window` trading days before the event
      - Exit `exit_window` trading days after the event
      - Apply round-trip cost

    direction: 'long' or 'short'
    """
    pre_col = f"pre_{entry_window}d_close"
    post_col = f"post_{exit_window}d_close"

    if pre_col not in events_df.columns or post_col not in events_df.columns:
        logger.warning("Required columns %s, %s not found", pre_col, post_col)
        return pd.DataFrame()

    df = events_df.dropna(subset=[pre_col, post_col]).copy()

    gross_return = df[post_col] / df[pre_col] - 1
    if direction == "short":
        gross_return = -gross_return

    cost = cost_bps / 10_000
    df["gross_return"] = gross_return
    df["net_return"] = gross_return - cost
    df["profitable"] = df["net_return"] > 0

    return df


def strategy_summary(df: pd.DataFrame) -> dict:
    """Summarize strategy performance including annualized Sharpe."""
    if df.empty or "net_return" not in df.columns:
        return {}

    net = df["net_return"].dropna()
    sharpe_per_trade = round(net.mean() / net.std(), 3) if net.std() > 0 else np.nan

    # Estimate annualized Sharpe: Sharpe_annual = Sharpe_trade * sqrt(trades/year)
    if "event_date" in df.columns:
        dates = pd.to_datetime(df["event_date"])
        span_years = max((dates.max() - dates.min()).days / 365.25, 0.5)
        trades_per_year = len(net) / span_years
    else:
        trades_per_year = 12.0  # conservative default

    annualized_sharpe = (
        sharpe_per_trade * np.sqrt(trades_per_year)
        if not np.isnan(sharpe_per_trade) else np.nan
    )

    return {
        "n_trades": len(net),
        "mean_return_pct": round(net.mean() * 100, 3),
        "median_return_pct": round(net.median() * 100, 3),
        "std_pct": round(net.std() * 100, 3),
        "sharpe_per_trade": sharpe_per_trade,
        "annualized_sharpe": round(annualized_sharpe, 2) if not np.isnan(annualized_sharpe) else np.nan,
        "trades_per_year": round(trades_per_year, 1),
        "win_rate_pct": round((net > 0).mean() * 100, 1),
        "best_pct": round(net.max() * 100, 2),
        "worst_pct": round(net.min() * 100, 2),
        "total_return_pct": round((1 + net).prod() - 1, 4) * 100,
    }


# ---------------------------------------------------------------------------
# Conditional momentum strategy
# ---------------------------------------------------------------------------

def compute_conditional_strategy(
    events_df: pd.DataFrame,
    signal_window: int = 5,
    entry_window: int = 3,
    exit_window: int = 1,
    signal_threshold: float = 0.02,
    cost_bps: float = 30.0,
) -> pd.DataFrame:
    """
    Conditional strategy: go long when pre-event drift is positive,
    go short when pre-event drift is negative.

    Logic:
      1. Observe the return over `signal_window` days before entry
      2. If return > +signal_threshold: go LONG
      3. If return < -signal_threshold: go SHORT
      4. Otherwise: no trade (skip)
      5. Enter `entry_window` days before the event, exit `exit_window` after

    This exploits the positive pre/post correlation found in small-cap
    biotech catalysts — if the stock is already drifting in a direction
    before the catalyst, it tends to continue.

    Higher cost_bps vs naive strategy accounts for shorting costs.
    """
    signal_col = f"pre_{signal_window}d_return"
    entry_col = f"pre_{entry_window}d_close"
    exit_col = f"post_{exit_window}d_close"

    required = [signal_col, entry_col, exit_col]
    missing = [c for c in required if c not in events_df.columns]
    if missing:
        logger.warning("Missing columns for conditional strategy: %s", missing)
        return pd.DataFrame()

    df = events_df.dropna(subset=required).copy()

    df["signal"] = np.where(
        df[signal_col] > signal_threshold, "long",
        np.where(df[signal_col] < -signal_threshold, "short", "skip")
    )

    trades = df[df["signal"] != "skip"].copy()
    if trades.empty:
        return trades

    raw_return = trades[exit_col] / trades[entry_col] - 1
    trades["gross_return"] = np.where(
        trades["signal"] == "long", raw_return, -raw_return
    )

    cost = cost_bps / 10_000
    trades["net_return"] = trades["gross_return"] - cost
    trades["profitable"] = trades["net_return"] > 0

    return trades


def conditional_strategy_sweep(
    events_df: pd.DataFrame,
    signal_windows: list[int] = None,
    entry_windows: list[int] = None,
    exit_windows: list[int] = None,
    thresholds: list[float] = None,
    cost_bps: float = 30.0,
) -> pd.DataFrame:
    """
    Sweep across parameter combinations and return a summary for each.
    Useful for finding which configuration works best.
    """
    if signal_windows is None:
        signal_windows = [3, 5, 10]
    if entry_windows is None:
        entry_windows = [1, 3, 5]
    if exit_windows is None:
        exit_windows = [1, 3, 5]
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.03, 0.05]

    results = []
    for sw in signal_windows:
        for ew in entry_windows:
            if ew >= sw:
                continue
            for xw in exit_windows:
                for thr in thresholds:
                    strat = compute_conditional_strategy(
                        events_df,
                        signal_window=sw,
                        entry_window=ew,
                        exit_window=xw,
                        signal_threshold=thr,
                        cost_bps=cost_bps,
                    )
                    if strat.empty:
                        continue
                    summ = strategy_summary(strat)
                    if not summ:
                        continue
                    summ["signal_window"] = sw
                    summ["entry_window"] = ew
                    summ["exit_window"] = xw
                    summ["threshold_pct"] = thr * 100
                    n_long = (strat["signal"] == "long").sum()
                    n_short = (strat["signal"] == "short").sum()
                    summ["n_long"] = n_long
                    summ["n_short"] = n_short

                    long_sub = strat[strat["signal"] == "long"]
                    short_sub = strat[strat["signal"] == "short"]
                    if len(long_sub) > 0:
                        summ["long_win_rate"] = round(
                            (long_sub["net_return"] > 0).mean() * 100, 1
                        )
                        summ["long_mean_return"] = round(
                            long_sub["net_return"].mean() * 100, 3
                        )
                    if len(short_sub) > 0:
                        summ["short_win_rate"] = round(
                            (short_sub["net_return"] > 0).mean() * 100, 1
                        )
                        summ["short_mean_return"] = round(
                            short_sub["net_return"].mean() * 100, 3
                        )

                    results.append(summ)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Options straddle/strangle strategy
# ---------------------------------------------------------------------------

def compute_options_straddle_strategy(
    events_df: pd.DataFrame,
    entry_window: int = 5,
    exit_window: int = 1,
    implied_vol_annual: float = 0.80,
    vol_crush_pct: float = 0.30,
    cost_bps: float = 50.0,
) -> pd.DataFrame:
    """
    Simulate an ATM straddle purchased before a catalyst event.

    This is a simplified model since we don't have historical options data.
    We estimate P&L using the Black-Scholes-inspired logic:

    Straddle cost ≈ 2 * S * sigma * sqrt(T)
      where T = holding period in years
      sigma = implied vol (elevated pre-catalyst)

    Straddle P&L:
      - Intrinsic value at exit = |S_exit - S_entry|  (move captured)
      - Time decay = straddle_cost * (1 - sqrt(T_remaining/T_original))
      - Vol crush = reduction in straddle value from IV dropping post-event

    The straddle profits when the realized move exceeds the cost
    (implied breakeven). Pre-catalyst IV is typically 1.5-3x normal in
    small-cap biotech, so we model elevated IV.

    Parameters:
        implied_vol_annual: annualized IV at entry (80% typical for small biotech pre-catalyst)
        vol_crush_pct: fraction of IV that evaporates after the event (30% typical)
        cost_bps: additional execution cost per leg
    """
    entry_col = f"pre_{entry_window}d_close"
    exit_col = f"post_{exit_window}d_close"
    event_close_col = "event_close"

    required = [entry_col, exit_col, event_close_col]
    missing = [c for c in required if c not in events_df.columns]
    if missing:
        logger.warning("Missing columns for straddle strategy: %s", missing)
        return pd.DataFrame()

    df = events_df.dropna(subset=required).copy()
    if df.empty:
        return df

    holding_days = entry_window + exit_window
    T = holding_days / 252.0
    sqrt_T = np.sqrt(T)

    S = df[entry_col]
    S_exit = df[exit_col]
    S_event = df[event_close_col]

    straddle_cost = 2.0 * S * implied_vol_annual * sqrt_T
    straddle_cost_pct = straddle_cost / S

    # Realized move from entry to exit (absolute)
    realized_move = (S_exit - S).abs()

    # Also track the event-day move since vol crush happens there
    event_day_move = (S_event - S).abs()

    # Intrinsic at exit
    intrinsic_at_exit = realized_move

    # Vol crush impact: after the event, the remaining time value
    # of the straddle drops because IV normalizes.
    # Pre-event: straddle has time value based on elevated IV
    # Post-event: IV drops by vol_crush_pct, remaining time value shrinks
    T_remaining = exit_window / 252.0
    time_value_at_entry = straddle_cost
    time_value_remaining_no_crush = 2.0 * S * implied_vol_annual * np.sqrt(T_remaining)
    time_value_remaining_crushed = 2.0 * S * implied_vol_annual * (1 - vol_crush_pct) * np.sqrt(T_remaining)

    # Net P&L = intrinsic_at_exit - straddle_cost_paid + remaining_time_value
    # But since we're closing, the remaining time value is what we sell for
    straddle_exit_value = intrinsic_at_exit + time_value_remaining_crushed
    gross_pnl = straddle_exit_value - straddle_cost

    cost = cost_bps / 10_000 * S * 2  # cost per share for both legs
    net_pnl = gross_pnl - cost

    df["straddle_cost_pct"] = straddle_cost_pct * 100
    df["realized_move_pct"] = (realized_move / S) * 100
    df["event_day_move_pct"] = (event_day_move / S) * 100
    df["straddle_gross_pnl_pct"] = (gross_pnl / straddle_cost) * 100
    df["straddle_net_pnl_pct"] = (net_pnl / straddle_cost) * 100
    df["breakeven_move_pct"] = straddle_cost_pct * 100
    df["exceeded_breakeven"] = realized_move > straddle_cost
    df["net_return"] = net_pnl / straddle_cost
    df["profitable"] = df["net_return"] > 0

    return df


def straddle_strategy_summary(df: pd.DataFrame) -> dict:
    """Summarize straddle strategy performance."""
    if df.empty or "net_return" not in df.columns:
        return {}

    net = df["net_return"].dropna()
    return {
        "n_trades": len(net),
        "mean_pnl_pct": round(net.mean() * 100, 2),
        "median_pnl_pct": round(net.median() * 100, 2),
        "std_pct": round(net.std() * 100, 2),
        "sharpe_per_trade": round(net.mean() / net.std(), 3) if net.std() > 0 else np.nan,
        "win_rate_pct": round((net > 0).mean() * 100, 1),
        "best_pct": round(net.max() * 100, 2),
        "worst_pct": round(net.min() * 100, 2),
        "avg_straddle_cost_pct": round(df["straddle_cost_pct"].mean(), 2),
        "avg_realized_move_pct": round(df["realized_move_pct"].mean(), 2),
        "pct_exceeded_breakeven": round(df["exceeded_breakeven"].mean() * 100, 1),
    }


def straddle_strategy_sweep(
    events_df: pd.DataFrame,
    entry_windows: list[int] = None,
    exit_windows: list[int] = None,
    iv_levels: list[float] = None,
    vol_crush_levels: list[float] = None,
) -> pd.DataFrame:
    """Sweep straddle parameters to find optimal configuration."""
    if entry_windows is None:
        entry_windows = [3, 5, 10]
    if exit_windows is None:
        exit_windows = [1, 3, 5]
    if iv_levels is None:
        iv_levels = [0.60, 0.80, 1.00, 1.20]
    if vol_crush_levels is None:
        vol_crush_levels = [0.20, 0.30, 0.40, 0.50]

    results = []
    for ew in entry_windows:
        for xw in exit_windows:
            for iv in iv_levels:
                for vc in vol_crush_levels:
                    strat = compute_options_straddle_strategy(
                        events_df,
                        entry_window=ew,
                        exit_window=xw,
                        implied_vol_annual=iv,
                        vol_crush_pct=vc,
                    )
                    if strat.empty:
                        continue
                    summ = straddle_strategy_summary(strat)
                    if not summ:
                        continue
                    summ["entry_window"] = ew
                    summ["exit_window"] = xw
                    summ["implied_vol"] = iv
                    summ["vol_crush"] = vc
                    results.append(summ)

    return pd.DataFrame(results)
