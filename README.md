# Trading Research Engine

`trading_research` is a typed, artifact-first engine for systematic alpha research.

It implements the requested two-layer architecture:

- Layer A: concise user-facing recipe API
- Layer B: explicit typed execution graph with persistent fold-scoped artifacts

## Highlights

- Real market data adapter for Polygon (`trading_research/data/vendors/polygon.py`)
- Model expansion including LOESS and Kernel Ridge regression
- Quant dashboard for run-level, fold-level, time-series, and asset-level deep dives
- Agent handoff summary generation for downstream "what-next" recommendation agents

## Start here

- Overview: `trading_research/docs/00_overview.md`
- Architecture: `trading_research/docs/01_architecture.md`
- Examples: `trading_research/examples/`

## Quick start

```bash
PYTHONPATH=/workspace python3 trading_research/examples/01_single_model_run.py
PYTHONPATH=/workspace python3 trading_research/examples/07_advanced_cross_target_meta_run.py
PYTHONPATH=/workspace python3 trading_research/examples/10_spy_horizon_error_meta_recipe.py
PYTHONPATH=/workspace python3 trading_research/examples/11_champion_edge_recipe.py
PYTHONPATH=/workspace python3 trading_research/examples/12_no_leakage_sanity_report.py
PYTHONPATH=/workspace python3 trading_research/examples/13_global_learning_asset_expansion_test.py
PYTHONPATH=/workspace python3 trading_research/examples/14_asset_specific_vol_feature_ablation.py
PYTHONPATH=/workspace python3 trading_research/examples/15_regime_aware_moe_gating.py
```

## Polygon vendor usage

```python
from trading_research.data.vendors import PolygonMarketDataVendor, PolygonVendorConfig
vendor = PolygonMarketDataVendor(PolygonVendorConfig(api_key="YOUR_POLYGON_KEY"))
bars = vendor.fetch_bars(
    assets=["AAPL", "MSFT"],
    start="2022-01-01",
    end="2023-01-01",
)
```

## Yahoo vendor usage

```python
from trading_research.data.vendors import YahooFinanceVendor

vendor = YahooFinanceVendor()
bars = vendor.fetch_bars(
    assets=["SPY"],
    period="2y",
    interval="1h",
)
```

## Quant dashboard

```bash
PYTHONPATH=/workspace python3 -m streamlit run trading_research/apps/quant_dashboard.py
```

## Live trading (Alpaca)

This repo now includes a production-oriented live runner for the current
`smart_breadth_quality_3x` logic with Alpaca.

Files:

- Broker adapter: `trading_research/live/alpaca.py`
- Strategy signal/weights: `trading_research/live/smart_breadth_live.py`
- Executable runner: `trading_research/examples/45_live_smart_breadth_alpaca.py`

Environment variables:

- `ALPACA_API_KEY_ID` (required for broker connectivity)
- `ALPACA_API_SECRET_KEY` (required for broker connectivity)
- `ALPACA_BASE_URL` (optional; defaults to paper endpoint)

Install dependency:

```bash
python3 -m pip install requests
```

Dry-run (recommended first, no orders):

```bash
PYTHONPATH=/workspace python3 trading_research/examples/45_live_smart_breadth_alpaca.py --mode dry-run
```

Apply orders in Alpaca paper account:

```bash
PYTHONPATH=/workspace python3 trading_research/examples/45_live_smart_breadth_alpaca.py --mode apply --paper
```

Run in live brokerage account (dangerous):

```bash
PYTHONPATH=/workspace python3 trading_research/examples/45_live_smart_breadth_alpaca.py --mode apply
```

Notes:

- The runner computes targets from the same universe and portfolio logic,
  then submits delta notional orders.
- It includes safety checks for max notional and order minimums.
- Keep paper mode until you confirm behavior and broker fills.
