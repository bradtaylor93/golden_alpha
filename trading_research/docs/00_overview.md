# Trading Research Engine Overview

This system is a **two-layer design**:

1. **Recipe API (user-facing)** for concise experiment declarations.
2. **Typed execution graph** for explicit, auditable artifact execution.

```
Recipe -> Compiler -> WorkflowGraph -> Runner -> Run Directory + Artifact Index
```

Core guarantees:

- Artifact-first persistence
- Fold-aware scope metadata
- Explicit diagnostics and projection nodes
- Reproducible runs with graph/config snapshots and lineage index

## Minimal usage

```python
graph = SingleModelRecipe(dataset_name="u1").compile()
run_id = WorkflowRunner(runs_dir=Path("runs"), data_catalog=catalog).run(
    graph,
    run_config={"exp": "quick"},
)
```

## Real data + advanced models

- Real vendor integration is available via `PolygonMarketDataVendor`.
- Additional non-linear regressors are available:
  - `algorithm="loess"`
  - `algorithm="kernel_ridge"`

See:

- `trading_research/examples/08_polygon_vendor_ingest.py`
- `trading_research/examples/09_advanced_models_run.py`

## Quant dashboard

Launch the dashboard:

```bash
python3 -m streamlit run trading_research/apps/quant_dashboard.py
```

The dashboard provides:

- run flow map and lineage edges
- fold-level diagnostics and performance views
- time and asset-level deep dives
- recommendations aimed at “what to do next”
- a machine-readable handoff payload for downstream quant agents
