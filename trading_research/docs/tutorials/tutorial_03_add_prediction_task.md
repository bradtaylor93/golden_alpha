# Tutorial 03: Add a prediction task

Define a task by implementing `PredictionTask` fields and registering it.

```python
from trading_research.targets.registry import PredictionTask

def my_target(df, horizon: int):
    # Example: future return over horizon
    return df.groupby("asset")["close"].shift(-horizon) / df["close"] - 1.0

task = PredictionTask(
    name="my_forward_return_10",
    task_type="regression",
    horizon=10,
    build_target_fn=my_target,
    metric_names=("mae", "rmse"),
    embargo=1,
)
```

Add it to your `TaskRegistry` before running the graph.
