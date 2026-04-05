"""Example 2: feature comparison run."""

from __future__ import annotations

from pathlib import Path

from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.mock import MockMarketDataVendor
from trading_research.recipes.comparison import FeatureComparisonRecipe
from trading_research.workflow.compiler import RecipeCompiler
from trading_research.workflow.runner import WorkflowRunner


def main() -> None:
    root = Path(".")
    catalog = DataCatalog(root / "local_data")
    vendor = MockMarketDataVendor()
    bars = vendor.fetch_bars(["SPY"])
    catalog.persist_processed_bars("default_universe", bars, metadata={"assets": ["SPY"]})

    recipe = FeatureComparisonRecipe(
        dataset_name="default_universe",
        universe=("SPY",),
    )
    graph = RecipeCompiler().compile(recipe)

    runner = WorkflowRunner(runs_dir=root / "runs", data_catalog=catalog)
    run_id = runner.run(graph, run_config={"example": "02_feature_comparison_run"})
    print(run_id)


if __name__ == "__main__":
    main()
