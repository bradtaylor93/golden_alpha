# Tutorial 06: Inspect a complex run

```python
from trading_research.workflow.inspectors import load_run

inspector = load_run("run_id_here", runs_root="runs")
print(inspector.list_nodes())
print(inspector.list_artifacts())
print(inspector.show_lineage(node_name="meta_model"))
```

