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
