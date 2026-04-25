"""
Central configuration for the biotech catalyst research project.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ClinicalTrials.gov v2 API
CT_GOV_BASE_URL = "https://clinicaltrials.gov/api/v2"

# FDA PDUFA calendar (scraped or manual)
FDA_CALENDAR_URL = "https://www.fda.gov/drugs/nda-and-bla-approvals"

# Price data
PRICE_LOOKBACK_DAYS = 30
PRICE_LOOKFORWARD_DAYS = 10

# Event windows for analysis
PRE_EVENT_WINDOWS = [1, 3, 5, 10, 20]   # trading days before
POST_EVENT_WINDOWS = [1, 3, 5, 10, 20]  # trading days after

# Volume spike threshold (multiple of 20-day average)
VOLUME_SPIKE_THRESHOLD = 2.0

# Minimum market cap for tradability (USD)
MIN_MARKET_CAP = 100_000_000

# Maximum market cap – focus on small/mid cap where catalysts move the stock
MAX_MARKET_CAP = 20_000_000_000

# Minimum average daily dollar volume for tradability
MIN_ADV_DOLLARS = 1_000_000
