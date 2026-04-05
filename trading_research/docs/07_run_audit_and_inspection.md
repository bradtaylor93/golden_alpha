# Run Audit and Inspection

Each run persists:

- `config/run_config.json`
- `config/workflow_graph.json`
- `manifest.json`
- `artifact_index.parquet`
- fold-scoped artifact files

Use:

```python
from trading_research import load_run

inspector = load_run("run_20260405T120000Z_abcd1234", runs_root="trading_research/examples/runs")
print(inspector.list_nodes())
print(inspector.list_artifacts()[:5])
print(inspector.show_lineage(node_name="meta_model"))
preds = inspector.load_predictions("meta_model")
```

Inspection endpoints:

- `load_run(run_id)`
- `list_nodes()`
- `list_artifacts()`
- `load_artifact(node_name=..., outer_fold_id=..., artifact_type=..., split_role=...)`
- `show_lineage(node_name=..., outer_fold_id=...)`
- `load_predictions(node_name=..., outer_fold_id=...)`
- `load_inner_validation(node_name=..., outer_fold_id=...)`
- `load_training_snapshot(node_name=..., outer_fold_id=...)`

## Quant dashboard

The dashboard gives a deep, fold-aware, end-to-end view for human quants and quant agents.

Run:

```bash
python3 -m streamlit run trading_research/apps/quant_dashboard.py -- --runs-root runs
```

Views include:

- **Run Overview**: run metadata, node count, artifact count, graph topology.
- **Flow / Graph View**: node dependency graph and per-node lineage preview.
- **Model & Fold Diagnostics**:
  - fold-level train/test metrics
  - train-test deviance gap
  - coefficient stability
  - prediction disagreement
- **Performance over time**:
  - cumulative test PnL proxy from prediction * target
  - rolling correlation and rolling error
- **Performance by asset**:
  - MAE, RMSE, residual std, directional hit-rate per asset
  - fold-by-asset stability matrix
- **Deep Dive**:
  - node-specific artifact explorers
  - raw fold artifact tables
  - training snapshots when available
- **Next-best-actions panel**:
  - machine-readable recommendation JSON
  - intended for direct handoff to a quant agent.
