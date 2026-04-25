# Biotech Catalyst Research

Research toolkit for analyzing stock price behavior around **pre-announced therapeutic company catalysts** (clinical trial results, FDA decisions).

## Core Question

> Can we systematically identify upcoming biotech catalysts that are known in advance, measure whether prices move *before* the announcement, assess whether this is tradeable, and reduce false positives?

---

## Feasibility Assessment

### Goal 1: Build a dataset of known-in-advance catalyst dates

**Verdict: Feasible, with caveats.**

Many biotech catalysts are genuinely known in advance:

| Source | What's Known | Lead Time | Access |
|--------|-------------|-----------|--------|
| ClinicalTrials.gov | Trial primary completion dates, result posting dates | Months–years | Free API (v2) |
| FDA PDUFA dates | Exact FDA decision dates for NDAs/BLAs | ~10 months | Public / free |
| Company guidance | "Data readout in H2 2025" | Weeks–quarters | Earnings calls, press releases |
| Conference schedules | Oral/poster presentations at ASCO, AACR, etc. | Weeks | Public schedules |
| SEC filings | 8-K filings, prospectus filings near milestones | Days | EDGAR / free |

**Key limitations:**
- ClinicalTrials.gov dates are *estimates* and frequently slip
- The *exact* day of a press release is rarely known >1 week ahead
- PDUFA dates are the most precise (exact calendar date, months ahead)
- Mapping sponsor names → stock tickers requires curation
- Small/micro-cap biotechs (where moves are largest) often have minimal ClinicalTrials.gov compliance

### Goal 2: Measure pre- and post-announcement price moves

**Verdict: Fully feasible.**

We compute:
- Returns over 1, 3, 5, 10, 20 trading days *before* the event
- Returns over 1, 3, 5, 10, 20 trading days *after* the event
- Volume ratios (event-day volume vs 20-day average)
- Correlation between pre-event drift and post-event reaction
- Classification of "anticipation" patterns (same-direction pre/post moves)

### Goal 3: Assess trade feasibility

**Verdict: Feasible for large/mid-cap, challenging for small-cap.**

Key factors assessed:
- **Liquidity**: Average daily dollar volume ≥ $1M threshold
- **Market impact**: Square-root impact model for execution cost estimates
- **Options availability**: Listed options enable defined-risk positions
- **Borrow availability**: Shorting small-cap biotech is often expensive/impossible
- **Holding period**: PDUFA trades have known entry/exit; trial readouts less precise

### Goal 4: Reduce false positives

**Verdict: Feasible with multi-factor scoring.**

Not every pre-event price move is information-driven. Our filters:

1. **Beta adjustment** – Strip out market/sector moves
2. **Z-score filter** – Is the move abnormal given the stock's own volatility?
3. **Volume confirmation** – Abnormal volume adds conviction
4. **Options flow** – Unusual call buying (requires external data)
5. **SEC filing check** – Alternative explanations (8-K, shelf registration)
6. **Composite score** – Combine filters into a quality tier (low/medium/high/very_high)

---

## Project Structure

```
biotech_catalyst_research/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── config.py                  # Central configuration
├── data_sources.py            # Data collection (ClinicalTrials.gov, prices)
├── price_analysis.py          # Return distribution and correlation analysis
├── trade_feasibility.py       # Liquidity, costs, strategy simulation
├── false_positive_reduction.py # Signal scoring and filtering
├── run_analysis.py            # Main pipeline orchestration
├── data/                      # Cached data files (git-ignored)
│   └── pdufa_dates_template.csv
└── output/                    # Analysis outputs (git-ignored)
    ├── analysis_summary.json
    ├── scored_events.csv
    └── *.png                  # Charts
```

## Setup

```bash
cd biotech_catalyst_research
pip install -r requirements.txt
```

## Usage

### Full pipeline

```bash
python run_analysis.py            # default: 500 trials
python run_analysis.py 1000       # fetch up to 1000 trials
```

### Interactive / modular

```python
from data_sources import fetch_trials_with_results, enrich_trials_with_tickers, build_event_price_dataset
from price_analysis import classify_moves, compute_summary_stats
from trade_feasibility import compute_strategy_returns, strategy_summary
from false_positive_reduction import compute_signal_score, summarize_by_signal_quality

# Step 1: Get trials
trials = fetch_trials_with_results(max_trials=200)
trials = enrich_trials_with_tickers(trials)

# Step 2: Get prices
events = build_event_price_dataset(trials)

# Step 3: Analyze
classified = classify_moves(events)
stats = compute_summary_stats(classified)

# Step 4: Trade feasibility
strat = compute_strategy_returns(classified, entry_window=5, exit_window=1)
print(strategy_summary(strat))

# Step 5: Score quality
scored = compute_signal_score(classified)
print(summarize_by_signal_quality(scored))
```

### PDUFA dates

For the most precise catalysts (FDA decision dates), populate `data/pdufa_dates_template.csv`:

```csv
ticker,drug_name,indication,pdufa_date,outcome,announced_date
SGEN,tisotumab vedotin,cervical cancer,2024-04-17,approved,2024-04-17
```

Then load with:

```python
from data_sources import load_pdufa_dates
pdufa = load_pdufa_dates("data/pdufa_dates.csv")
```

## Data Sources & Limitations

| Source | Cost | Coverage | Precision |
|--------|------|----------|-----------|
| ClinicalTrials.gov API v2 | Free | Comprehensive for registered trials | Completion dates often imprecise |
| FDA PDUFA calendar | Free | US NDAs/BLAs only | Exact date, very reliable |
| yfinance (Yahoo Finance) | Free | US equities, good history | Adjusted prices, no intraday |
| BioPharmCatalyst | ~$50/mo | Best catalyst database | High precision, historical |
| Unusual Whales / options data | $30-100/mo | Options flow | Real-time unusual activity |
| WRDS/OptionMetrics | Academic | Full historical options | Institutional-grade |

For serious research, **BioPharmCatalyst** is the single best source for historical catalyst dates with ticker mapping already done. The free approach in this project uses ClinicalTrials.gov + manual curation.
