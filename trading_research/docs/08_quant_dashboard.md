# 08 Quant Dashboard

The dashboard is designed for a quant to answer:

- What happened at each node in the flow?
- Which folds are healthy vs degraded?
- Which assets contribute most to risk or alpha?
- Where does performance drift over time?
- What should we try next?

Launch:

```bash
PYTHONPATH=/workspace ~/.local/bin/streamlit run trading_research/apps/quant_dashboard.py
```

or with explicit run root:

```bash
PYTHONPATH=/workspace ~/.local/bin/streamlit run trading_research/apps/quant_dashboard.py -- --runs-root trading_research/examples/_output
```

## Views

### 1) Run Overview

- Run metadata and manifest
- Node and artifact counts
- Artifact-type distribution
- Fold coverage
- Mean fold train/test error and deviance gap

### 2) Workflow Flow View

- Layered node listing:
  - ingest
  - targets/folds
  - features
  - models
  - diagnostics
  - projections/selections
- Full lineage table from artifact index

### 3) Fold Deep Dive

- Fold-level metrics:
  - train_metric
  - test_metric
  - train_test_deviance_gap
  - residual_std_by_asset
  - coefficient_stability
  - prediction_disagreement
- Fold comparison charts
- Per-fold artifact listing

### 4) Time-Series Analysis

- Prediction and residual behavior over time
- Rolling residual standard deviation
- Fold boundary overlays

### 5) Asset Analysis

- Performance by asset:
  - MAE
  - MSE
  - residual std
  - directional hit-rate
- Asset contribution plots

### 6) Diagnostics and Drift

- Diagnostic node outputs over folds
- Drift metric trend panels
- Stability indicators

### 7) Quant Agent Handoff

Machine-readable summary includes:

- run metadata
- top/bottom assets
- worst folds
- drift and deviance signals
- candidate next actions

The handoff is copy/paste ready for downstream quant-agent prompting.

## Data source integration note

The dashboard is source-agnostic. If run artifacts come from Polygon-ingested
data (via `PolygonMarketDataVendor`), all fold/asset/time views remain identical.

