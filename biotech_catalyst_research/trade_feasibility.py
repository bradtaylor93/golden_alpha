"""
Module 3: Trade feasibility assessment.

Evaluates whether one could practically trade around known catalyst dates by
examining liquidity, spread costs, borrow availability, and options markets.
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
    """Summarize strategy performance."""
    if df.empty or "net_return" not in df.columns:
        return {}

    net = df["net_return"].dropna()
    return {
        "n_trades": len(net),
        "mean_return_pct": round(net.mean() * 100, 3),
        "median_return_pct": round(net.median() * 100, 3),
        "std_pct": round(net.std() * 100, 3),
        "sharpe_per_trade": round(net.mean() / net.std(), 3) if net.std() > 0 else np.nan,
        "win_rate_pct": round((net > 0).mean() * 100, 1),
        "best_pct": round(net.max() * 100, 2),
        "worst_pct": round(net.min() * 100, 2),
        "total_return_pct": round((1 + net).prod() - 1, 4) * 100,
        "avg_holding_period_comment": "Entry to exit window as configured",
    }
