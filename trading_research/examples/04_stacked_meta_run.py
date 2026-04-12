"""Example 4: stacked meta run."""

from __future__ import annotations

from pathlib import Path

from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.mock import MockMarketDataVendor
from trading_research.models.registry import ModelSpec
from trading_research.recipes.stacking import StackedModelRecipe
from trading_research.workflow.runner import WorkflowRunner


def main() -> None:
    root = Path(".")
    catalog = DataCatalog(root / "local_data")
    bars = MockMarketDataVendor().fetch_bars(["AAPL", "MSFT"])
    catalog.persist_processed_bars("default_universe", bars, metadata={"source": "mock"})

    recipe = StackedModelRecipe(
        universe=("AAPL", "MSFT"),
        task_name="forward_return_20",
        base_model_spec=ModelSpec(name="base_ridge", algorithm="ridge", params={"alpha": 1.0}),
        meta_model_spec=ModelSpec(name="meta_ridge", algorithm="ridge", params={"alpha": 0.7}),
    )
    graph = recipe.compile()
    run_id = WorkflowRunner(runs_dir=root / "runs", data_catalog=catalog).run(
        graph,
        run_config={"example": "04_stacked_meta_run"},
    )
    print(run_id)


if __name__ == "__main__":
    main()
