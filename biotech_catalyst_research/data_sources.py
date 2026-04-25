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

_HIGH_IMPACT_CONDITIONS = {
    "oncology": ["cancer", "tumor", "carcinoma", "melanoma", "lymphoma",
                 "leukemia", "myeloma", "sarcoma", "glioblastoma", "nsclc",
                 "neoplasm", "malignant", "metastat"],
    "rare_disease": ["orphan", "rare", "duchenne", "dmd", "sma", "cystic fibrosis",
                     "huntington", "als", "amyotrophic", "gaucher", "fabry",
                     "hemophilia", "thalassemia", "pku", "phenylketonuria"],
    "neurology": ["alzheimer", "parkinson", "multiple sclerosis", "epilepsy",
                  "migraine", "depression", "schizophrenia", "bipolar",
                  "anxiety", "adhd", "autism", "neuropath"],
    "immunology": ["lupus", "rheumatoid", "crohn", "colitis", "psoriasis",
                   "atopic dermatitis", "asthma", "eczema", "autoimmune"],
    "infectious": ["hiv", "hepatitis", "covid", "influenza", "tuberculosis",
                   "malaria", "infection"],
    "cardiovascular": ["heart failure", "hypertension", "atrial fibrillation",
                       "coronary", "stroke", "thrombosis", "pulmonary arterial"],
    "metabolic": ["diabetes", "obesity", "nash", "nafld", "cholesterol",
                  "lipid", "metabolic"],
}


def _categorize_condition(conditions: list[str]) -> str:
    """Categorize trial conditions into therapeutic areas."""
    text = " ".join(conditions).lower()
    matches = []
    for category, keywords in _HIGH_IMPACT_CONDITIONS.items():
        if any(kw in text for kw in keywords):
            matches.append(category)
    return ",".join(matches) if matches else "other"


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
                design = proto.get("designModule", {})
                design_info = design.get("designInfo", {})
                masking_info = design_info.get("maskingInfo", {})
                enrollment = design.get("enrollmentInfo", {})
                arms_mod = proto.get("armsInterventionsModule", {})
                outcomes = proto.get("outcomesModule", {})

                lead = sponsor.get("leadSponsor", {})
                completion = status.get("completionDateStruct", {})
                results_date = status.get("resultsFirstPostDateStruct", {})
                primary_completion = status.get("primaryCompletionDateStruct", {})

                # Extract intervention types and names
                interventions = arms_mod.get("interventions", [])
                intervention_types = [i.get("type", "") for i in interventions]
                intervention_names = [i.get("name", "") for i in interventions]

                # Count arms and arm types
                arm_groups = arms_mod.get("armGroups", [])
                arm_types = [a.get("type", "") for a in arm_groups]

                # Primary outcomes
                primary_outcomes = outcomes.get("primaryOutcomes", [])

                condition_list = conditions.get("conditions", [])

                records.append({
                    "nct_id": ident.get("nctId"),
                    "title": ident.get("briefTitle"),
                    "sponsor": lead.get("name"),
                    "phase": ",".join(design.get("phases", [])),
                    "conditions": ",".join(condition_list),
                    "overall_status": status.get("overallStatus"),
                    "primary_completion_date": primary_completion.get("date"),
                    "completion_date": completion.get("date"),
                    "results_first_post_date": results_date.get("date"),
                    "study_first_post_date": status.get("studyFirstPostDateStruct", {}).get("date"),
                    # Catalyst grading fields
                    "enrollment": enrollment.get("count"),
                    "allocation": design_info.get("allocation"),
                    "intervention_model": design_info.get("interventionModel"),
                    "primary_purpose": design_info.get("primaryPurpose"),
                    "masking": masking_info.get("masking"),
                    "n_arms": len(arm_groups),
                    "arm_types": ",".join(arm_types),
                    "intervention_types": ",".join(intervention_types),
                    "intervention_names": "|".join(intervention_names),
                    "n_primary_outcomes": len(primary_outcomes),
                    "is_fda_regulated": proto.get("oversightModule", {}).get("isFDARegulatedDrug"),
                    "condition_category": _categorize_condition(condition_list),
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
    # --- Large cap (>$20B, kept for completeness but filtered later) ---
    "Pfizer": "PFE",
    "Merck Sharp & Dohme": "MRK",
    "Novartis": "NVS",
    "Roche": "RHHBY",
    "Hoffmann-La Roche": "RHHBY",
    "Genentech": "RHHBY",
    "AstraZeneca": "AZN",
    "Johnson & Johnson": "JNJ",
    "Janssen": "JNJ",
    "Bristol-Myers Squibb": "BMY",
    "Eli Lilly": "LLY",
    "AbbVie": "ABBV",
    "Amgen": "AMGN",
    "Gilead Sciences": "GILD",
    "Regeneron": "REGN",
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
    "Daiichi Sankyo": "DSNKY",
    "Astellas": "ALPMY",
    "Boehringer Ingelheim": "0#BING",  # private, no ticker
    "Otsuka Pharmaceutical": "OTSKF",
    "Teva": "TEVA",
    "CSL Behring": "CSLLY",
    # --- Mid cap ($2B–$20B) ---
    "BioMarin Pharmaceutical": "BMRN",
    "BioMarin": "BMRN",
    "Incyte": "INCY",
    "Ionis Pharmaceuticals": "IONS",
    "Alnylam Pharmaceuticals": "ALNY",
    "BeiGene": "BGNE",
    "Jazz Pharmaceuticals": "JAZZ",
    "Neurocrine": "NBIX",
    "Insmed": "INSM",
    "Ascendis Pharma": "ASND",
    "Alkermes": "ALKS",
    "UCB Pharma": "UCBJY",
    "UCB Biopharma": "UCBJY",
    "UCB Japan": "UCBJY",
    "Ipsen": "IPSEY",
    "Eisai": "ESALY",
    "Shionogi": "SGIOY",
    "Celgene": "CELG",
    "ACADIA Pharmaceuticals": "ACAD",
    "Aurinia Pharmaceuticals": "AUPH",
    "Lexicon Pharmaceuticals": "LXRX",
    "Rhythm Pharmaceuticals": "RYTM",
    "Supernus Pharmaceuticals": "SUPN",
    "Lantheus": "LNTH",
    "Amphastar Pharmaceuticals": "AMPH",
    "Samsung Bioepis": "207940.KS",
    "Swedish Orphan Biovitrum": "BIOVF",
    "Allergan": "AGN",
    "Celltrion": "068270.KS",
    "Vanda Pharmaceuticals": "VNDA",
    "Spectrum Pharmaceuticals": "SPPI",
    "The Medicines Company": "MDCO",
    # --- Small cap (<$2B) – the most interesting for catalyst trading ---
    "Rigel Pharmaceuticals": "RIGL",
    "BioXcel Therapeutics": "BTAI",
    "Omeros": "OMER",
    "Tonix Pharmaceuticals": "TNXP",
    "Annovis Bio": "ANVS",
    "Phathom Pharmaceuticals": "PHAT",
    "Scynexis": "SCYX",
    "Orexigen Therapeutics": "OREX",
    "Motif Bio": "MTFB",
    "Novan": "NOVN",
    "Glaukos": "GKOS",
    "Protalix": "PLX",
    "Melinta Therapeutics": "MLNT",
    "OPKO": "OPK",
    "Cumberland Pharmaceuticals": "CPIX",
    "Integra LifeSciences": "IART",
    "Boston Scientific": "BSX",
    "Dermira": "DERM",
    "XenoPort": "XNPT",
    "Alder Biopharmaceuticals": "ALDR",
    "Otonomy": "OTIC",
    "Human Genome Sciences": "HGSI",
    "Valneva": "VALN",
    "Forest Laboratories": "FRX",
    "Kythera Biopharmaceuticals": "KYTH",
    "Dr. Reddy's": "RDY",
    "Sun Pharma": "SUNPHARMA.NS",
    "Actelion": "ALIOY",
    "ImmuPharma": "IMM.L",
    "Bausch": "BHC",
    "SANUWAVE": "SNWV",
    "Timber Pharmaceuticals": "TMBR",
    "VIVUS": "VVUS",
    "Vyne Therapeutics": "VYNE",
    "RVL Pharmaceuticals": "RVLP",
    "Biocon": "BIOCON.NS",
    "Abbott": "ABT",
    "Sprout Pharmaceuticals": "SPRX",
    "Bioverativ": "BIVV",
    "Shire": "SHPG",
    "Baxalta": "BXLT",
    "EMD Serono": "MKGAY",
    "Merck KGaA": "MKGAY",
    "GE Healthcare": "GEHC",
    "Organon": "OGN",
    "LEO Pharma": "0#LEO",  # private
    "Colgate Palmolive": "CL",
    "Grifols": "GRFS",
    "Fresenius Kabi": "FSNUY",
    "Alcon": "ALC",
    "ViiV Healthcare": "GSK",
    "LianBio": "LIAN",
    "HanAll BioPharma": "009420.KS",
    "Kyowa Kirin": "KYKOF",
    "Sumitomo Pharma": "SMPNY",
}


def map_sponsor_to_ticker(sponsor_name: str) -> Optional[str]:
    """Best-effort mapping from sponsor name to US-listed ticker."""
    if not sponsor_name:
        return None
    for key, ticker in _MANUAL_TICKER_MAP.items():
        if key.lower() in sponsor_name.lower():
            if ticker.startswith("0#"):
                return None  # private company marker
            return ticker
    return None


_yf_lookup_cache: dict[str, Optional[str]] = {}


def _yfinance_ticker_lookup(sponsor_name: str) -> Optional[str]:
    """Attempt to find a ticker via yfinance search for unmatched sponsors."""
    if sponsor_name in _yf_lookup_cache:
        return _yf_lookup_cache[sponsor_name]

    clean = sponsor_name.split(",")[0].split("(")[0].strip()
    clean = clean.replace("Inc.", "").replace("LLC", "").replace("Ltd.", "").strip()

    try:
        results = yf.Search(clean, max_results=3)
        quotes = getattr(results, "quotes", [])
        if not quotes:
            _yf_lookup_cache[sponsor_name] = None
            return None

        for q in quotes:
            exchange = q.get("exchange", "")
            symbol = q.get("symbol", "")
            name = q.get("shortname", "") or q.get("longname", "")
            if exchange in ("NMS", "NYQ", "NGM", "NCM", "ASE", "PCX"):
                _yf_lookup_cache[sponsor_name] = symbol
                return symbol

        _yf_lookup_cache[sponsor_name] = None
        return None
    except Exception:
        _yf_lookup_cache[sponsor_name] = None
        return None


def enrich_trials_with_tickers(
    df: pd.DataFrame,
    use_yfinance_lookup: bool = True,
) -> pd.DataFrame:
    """Add a 'ticker' column to the trials DataFrame.
    First tries the manual map, then optionally falls back to yfinance search."""
    df = df.copy()
    df["ticker"] = df["sponsor"].apply(map_sponsor_to_ticker)

    if use_yfinance_lookup:
        unmatched_mask = df["ticker"].isna()
        unmatched_sponsors = df.loc[unmatched_mask, "sponsor"].dropna().unique()
        logger.info("Attempting yfinance lookup for %d unmatched sponsors...", len(unmatched_sponsors))

        for sponsor in tqdm(unmatched_sponsors, desc="Ticker lookup"):
            ticker = _yfinance_ticker_lookup(sponsor)
            if ticker:
                df.loc[df["sponsor"] == sponsor, "ticker"] = ticker
                logger.info("  %s -> %s", sponsor, ticker)
            time.sleep(0.2)

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
