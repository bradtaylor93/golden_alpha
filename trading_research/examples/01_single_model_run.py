"""Example 1: simplest single model run."""

from __future__ import annotations

from pathlib import Path

from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.mock import MockMarketDataVendor
from trading_research.models.registry import ModelSpec
from trading_research.recipes.single_model import SingleModelRecipe
from trading_research.workflow.runner import WorkflowRunner


def main() -> None:
    root = Path("trading_research/examples/_output/01_single")
    catalog = DataCatalog(root / "data")
    bars = MockMarketDataVendor().fetch_bars(["AAPL"])
    catalog.persist_processed_bars("default_universe", bars, metadata={"assets": ["AAPL"]})

    recipe = SingleModelRecipe(
        dataset_name="default_universe",
        universe=("AAPL",),
        task_name="forward_return_20",
        model_spec=ModelSpec(name="ridge_single", algorithm="ridge", params={"alpha": 1.0}),
    )
    graph = recipe.compile()
    run_id = WorkflowRunner(runs_dir=root / "runs", data_catalog=catalog).run(
        graph,
        run_config={"example": "01_single_model_run"},
    )
    print(run_id)


if __name__ == "__main__":
    main()
