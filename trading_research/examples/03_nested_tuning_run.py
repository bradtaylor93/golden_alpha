"""Example 3: nested tuning run."""

from __future__ import annotations

from pathlib import Path

from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.mock import MockMarketDataVendor
from trading_research.recipes import NestedTuningRecipe
from trading_research.workflow.compiler import RecipeCompiler
from trading_research.workflow.runner import WorkflowRunner


def main() -> None:
    root = Path("local_state")
    catalog = DataCatalog(root / "data")
    vendor = MockMarketDataVendor()
    bars = vendor.fetch_bars(["AAPL", "MSFT", "NVDA"])
    catalog.persist_processed_bars("us_tech", bars, metadata={"assets": ["AAPL", "MSFT", "NVDA"]})

    recipe = NestedTuningRecipe(
        dataset_name="us_tech",
        universe=("AAPL", "MSFT", "NVDA"),
        task_name="forward_return_20",
    )
    graph = RecipeCompiler().compile(recipe)
    runner = WorkflowRunner(runs_dir=root / "runs", data_catalog=catalog)
    run_id = runner.run(graph, run_config={"example": "03_nested_tuning_run"})
    print(run_id)


if __name__ == "__main__":
    main()
