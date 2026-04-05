# Tutorial 04: Add a model

1. Implement estimator fit/predict.
2. Register it in `ModelRegistry`.
3. Reference it in `ModelSpec`.

```python
from trading_research.models.registry import ModelSpec

spec = ModelSpec(
    name="my_linear_model",
    algorithm="ridge",
    params={"alpha": 0.5},
)
```

Attach `spec` to a `ModelNode` in your recipe.
