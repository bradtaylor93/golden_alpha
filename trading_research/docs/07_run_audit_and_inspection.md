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
