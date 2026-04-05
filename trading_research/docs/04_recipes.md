# Recipe Layer

Recipes are the primary user API. A recipe compiles into a typed `WorkflowGraph`.

## Built-in recipes

- `SingleModelRecipe`
- `FeatureComparisonRecipe`
- `NestedTuningRecipe`
- `StackedModelRecipe`
- `DiagnosticAwareMetaRecipe`
- `SelectorRecipe`

## Why recipes matter

Researchers should not be forced to author low-level execution graphs for common workflows.
Recipes provide concise, reproducible templates while preserving full graph auditability.

## Typical usage

```python
from pathlib import Path

from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.mock import MockMarketDataVendor
from trading_research.models.registry import ModelSpec
from trading_research.recipes.single_model import SingleModelRecipe
from trading_research.workflow.runner import WorkflowRunner

catalog = DataCatalog(Path("data"))
bars = MockMarketDataVendor().fetch_bars(["AAPL"])
catalog.persist_processed_bars("us_equities", bars)

recipe = SingleModelRecipe(
    universe_name="us_equities",
    task_name="forward_return_20",
    model_spec=ModelSpec(name="ridge", algorithm="ridge", params={"alpha": 1.0}),
)

run_id = WorkflowRunner(Path("runs"), catalog).run(recipe.compile(), {"recipe": "single_model"})
print(run_id)
```
