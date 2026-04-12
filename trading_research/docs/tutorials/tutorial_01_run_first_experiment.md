# Tutorial 01: Run your first experiment

1. Generate local mock bars and run the simplest recipe:

```bash
python3 -m trading_research.examples.01_single_model_run
```

2. Inspect the created run folder:

```text
runs/<run_id>/
  config/run_config.json
  config/workflow_graph.json
  manifest.json
  artifact_index.parquet
  global/
```

3. Use the inspector API:

```python
from trading_research.workflow.inspectors import load_run
inspector = load_run("<run_id>")
print(inspector.list_nodes())
print(inspector.list_artifacts())
```
