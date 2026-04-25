"""
Module 1: Data collection from public sources.

Key insight: many biotech catalysts are KNOWN IN ADVANCE because:
  - ClinicalTrials.gov lists expected primary completion dates
  - FDA PDUFA dates are published ~10 months ahead of decision
  - Companies announce timelines in earnings calls / press releases
  - Conference presentations are scheduled weeks/months ahead

This module collects:
  1. Clinical trial completion/result dates from ClinicalTrials.gov
  2. PDUFA action dates (FDA drug approval decisions)
  3. Historical stock prices around those dates
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from tqdm import tqdm

from config import (
    CT_GOV_BASE_URL,
    DATA_DIR,
    PRICE_LOOKBACK_DAYS,
    PRICE_LOOKFORWARD_DAYS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. ClinicalTrials.gov – trials with results from publicly-traded sponsors
# ---------------------------------------------------------------------------

def fetch_trials_with_results(
    sponsor_type: str = "INDUSTRY",
    phase: str = "PHASE3",
    max_trials: int = 1000,
    page_size: int = 100,
) -> pd.DataFrame:
    """
    Query ClinicalTrials.gov v2 API for completed trials that have posted
    results, filtering by sponsor type and phase.

    The API returns structured JSON including:
      - protocolSection.identificationModule  (NCT ID, title, org)
      - protocolSection.statusModule          (dates, status)
      - protocolSection.sponsorCollaboratorsModule
      - resultsSection                        (primary outcome results)
    """
    records = []
    next_page_token = None
    fetched = 0

    with tqdm(total=max_trials, desc="Fetching trials") as pbar:
        while fetched < max_trials:
            query_parts = [
                f"AREA[Phase]{phase}",
                f"AREA[LeadSponsorClass]{sponsor_type}",
                "AREA[ResultsFirstPostDate]RANGE[MIN,MAX]",
            ]
            params = {
                "query.term": " AND ".join(query_parts),
                "filter.overallStatus": "COMPLETED",
                "pageSize": min(page_size, max_trials - fetched),
                "fields": (
                    "NCTId,BriefTitle,OrgFullName,Phase,"
                    "CompletionDate,ResultsFirstPostDate,"
                    "PrimaryCompletionDate,StudyFirstPostDate,"
                    "OverallStatus,LeadSponsorName,Condition"
                ),
                "format": "json",
            }
            if next_page_token:
                params["pageToken"] = next_page_token

            try:
                resp = requests.get(
                    f"{CT_GOV_BASE_URL}/studies",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                logger.warning("ClinicalTrials.gov request failed: %s", e)
                break

            studies = data.get("studies", [])
            if not studies:
                break

            for study in studies:
                proto = study.get("protocolSection", {})
                ident = proto.get("identificationModule", {})
                status = proto.get("statusModule", {})
                sponsor = proto.get("sponsorCollaboratorsModule", {})
                conditions = proto.get("conditionsModule", {})

                lead = sponsor.get("leadSponsor", {})
                completion = status.get("completionDateStruct", {})
                results_date = status.get("resultsFirstPostDateStruct", {})
                primary_completion = status.get("primaryCompletionDateStruct", {})

                records.append({
                    "nct_id": ident.get("nctId"),
                    "title": ident.get("briefTitle"),
                    "sponsor": lead.get("name"),
                    "phase": ",".join(proto.get("designModule", {}).get("phases", [])),
                    "conditions": ",".join(conditions.get("conditions", [])),
                    "overall_status": status.get("overallStatus"),
                    "primary_completion_date": primary_completion.get("date"),
                    "completion_date": completion.get("date"),
                    "results_first_post_date": results_date.get("date"),
                    "study_first_post_date": status.get("studyFirstPostDateStruct", {}).get("date"),
                })

            fetched += len(studies)
            pbar.update(len(studies))

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

            time.sleep(0.3)

    df = pd.DataFrame(records)
    if not df.empty:
        for col in ["primary_completion_date", "completion_date",
                     "results_first_post_date", "study_first_post_date"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 2. PDUFA dates – historical FDA action dates
# ---------------------------------------------------------------------------

# Maintained datasets of PDUFA dates exist from:
#   - BioPharmCatalyst (commercial, most complete)
#   - FDA website (partial, current year)
#   - RTTNews PDUFA calendar
#   - Biopharm Insight
#
# For a free approach we provide a manual/CSV-based loader plus a
# demonstration scraper for the FDA site.

def load_pdufa_dates(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load PDUFA action dates from a CSV file.

    Expected columns:
        ticker, drug_name, indication, pdufa_date, outcome, announced_date

    If no CSV provided, returns an empty DataFrame with the correct schema
    that the user can populate.
    """
    cols = ["ticker", "drug_name", "indication", "pdufa_date",
            "outcome", "announced_date"]
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["pdufa_date", "announced_date"])
    else:
        template_path = DATA_DIR / "pdufa_dates_template.csv"
        if not template_path.exists():
            pd.DataFrame(columns=cols).to_csv(template_path, index=False)
            logger.info("Created PDUFA template at %s – populate with data", template_path)
        df = pd.DataFrame(columns=cols)
    return df


# ---------------------------------------------------------------------------
# 3. Sponsor → ticker mapping
# ---------------------------------------------------------------------------

# ClinicalTrials.gov uses company legal names, not tickers.
# We provide a helper that attempts a rough match via yfinance search,
# plus a manual override dictionary.

_MANUAL_TICKER_MAP = {
    "Pfizer": "PFE",
    "Merck Sharp & Dohme LLC": "MRK",
    "Merck Sharp & Dohme Corp.": "MRK",
    "Novartis": "NVS",
    "Novartis Pharmaceuticals": "NVS",
    "Roche": "RHHBY",
    "Hoffmann-La Roche": "RHHBY",
    "AstraZeneca": "AZN",
    "Johnson & Johnson": "JNJ",
    "Bristol-Myers Squibb": "BMY",
    "Eli Lilly and Company": "LLY",
    "AbbVie": "ABBV",
    "Amgen": "AMGN",
    "Gilead Sciences": "GILD",
    "Regeneron Pharmaceuticals": "REGN",
    "Vertex Pharmaceuticals": "VRTX",
    "Biogen": "BIIB",
    "Moderna": "MRNA",
    "BioNTech": "BNTX",
    "Sanofi": "SNY",
    "GSK": "GSK",
    "GlaxoSmithKline": "GSK",
    "Takeda": "TAK",
    "Novo Nordisk": "NVO",
    "Bayer": "BAYRY",
}


def map_sponsor_to_ticker(sponsor_name: str) -> Optional[str]:
    """Best-effort mapping from sponsor name to US-listed ticker."""
    if not sponsor_name:
        return None
    for key, ticker in _MANUAL_TICKER_MAP.items():
        if key.lower() in sponsor_name.lower():
            return ticker
    return None


def enrich_trials_with_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'ticker' column to the trials DataFrame."""
    df = df.copy()
    df["ticker"] = df["sponsor"].apply(map_sponsor_to_ticker)
    return df


# ---------------------------------------------------------------------------
# 4. Price data retrieval
# ---------------------------------------------------------------------------

def fetch_price_data(
    ticker: str,
    event_date: datetime,
    lookback: int = PRICE_LOOKBACK_DAYS,
    lookforward: int = PRICE_LOOKFORWARD_DAYS,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV data around an event date.
    Uses calendar-day buffer to ensure enough trading days.
    """
    start = event_date - timedelta(days=int(lookback * 1.8))
    end = event_date + timedelta(days=int(lookforward * 1.8))

    try:
        data = yf.download(
            ticker, start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            return None
        # Flatten MultiIndex columns if present (yfinance >=0.2.31)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        logger.warning("Failed to fetch price data for %s: %s", ticker, e)
        return None


def build_event_price_dataset(
    events: pd.DataFrame,
    date_col: str = "results_first_post_date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    For each event row, fetch price data and compute returns over
    multiple windows before and after the event date.
    Fetches prices per-event to handle events spanning different years.
    """
    from config import PRE_EVENT_WINDOWS, POST_EVENT_WINDOWS

    results = []
    valid = events.dropna(subset=[date_col, ticker_col])
    price_cache = {}

    for ticker in tqdm(valid[ticker_col].unique(), desc="Fetching prices"):
        group = valid[valid[ticker_col] == ticker]

        for _, row in group.iterrows():
            event_dt = pd.Timestamp(row[date_col])
            cache_key = (ticker, event_dt.strftime("%Y-%m"))

            if cache_key not in price_cache:
                price_df = fetch_price_data(
                    ticker, event_dt,
                    lookback=max(PRE_EVENT_WINDOWS) + 5,
                    lookforward=max(POST_EVENT_WINDOWS) + 5,
                )
                price_cache[cache_key] = price_df

            price_df = price_cache[cache_key]
            if price_df is None or price_df.empty:
                continue

            trading_dates = price_df.index
            on_or_after = trading_dates[trading_dates >= event_dt]
            if on_or_after.empty:
                continue
            event_td = on_or_after[0]
            event_idx = trading_dates.get_loc(event_td)

            rec = {
                "nct_id": row.get("nct_id"),
                "ticker": ticker,
                "event_date": event_dt,
                "nearest_trading_date": event_td,
                "event_close": price_df.iloc[event_idx]["Close"],
            }

            for w in PRE_EVENT_WINDOWS:
                pre_idx = event_idx - w
                if pre_idx >= 0:
                    pre_close = price_df.iloc[pre_idx]["Close"]
                    rec[f"pre_{w}d_return"] = (
                        rec["event_close"] / pre_close - 1
                    )
                    rec[f"pre_{w}d_close"] = pre_close

            for w in POST_EVENT_WINDOWS:
                post_idx = event_idx + w
                if post_idx < len(price_df):
                    post_close = price_df.iloc[post_idx]["Close"]
                    rec[f"post_{w}d_return"] = (
                        post_close / rec["event_close"] - 1
                    )
                    rec[f"post_{w}d_close"] = post_close

            if event_idx >= 20:
                avg_vol = price_df.iloc[event_idx - 20:event_idx]["Volume"].mean()
                event_vol = price_df.iloc[event_idx]["Volume"]
                rec["volume_ratio_event_day"] = (
                    event_vol / avg_vol if avg_vol > 0 else np.nan
                )

            results.append(rec)

        time.sleep(0.5)

    return pd.DataFrame(results)
