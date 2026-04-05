"""Example 9: run LOESS and Kernel Ridge candidate models."""

from __future__ import annotations

from pathlib import Path

from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.mock import MockMarketDataVendor
from trading_research.models.registry import ModelSpec
from trading_research.recipes.selector import SelectorRecipe
from trading_research.workflow.runner import WorkflowRunner


def main() -> None:
    root = Path("trading_research/examples/_output/09_advanced_models")
    catalog = DataCatalog(root / "data")
    bars = MockMarketDataVendor().fetch_bars(["AAPL", "MSFT", "NVDA", "QQQ"])
    catalog.persist_processed_bars("default_universe", bars)

    recipe = SelectorRecipe(
        dataset_name="default_universe",
        universe=("AAPL", "MSFT", "NVDA", "QQQ"),
        task_name="forward_return_20",
        candidates=(
            ModelSpec(name="ridge_candidate", algorithm="ridge", params={"alpha": 1.0}),
            ModelSpec(
                name="loess_candidate",
                algorithm="loess",
                params={"span": 0.25, "ridge_alpha": 1e-3, "max_reference_rows": 1500},
            ),
            ModelSpec(
                name="kernel_ridge_candidate",
                algorithm="kernel_ridge",
                params={"alpha": 0.5, "gamma": 0.5, "max_reference_rows": 1500},
            ),
        ),
        strategy="weighted_blend",
    )
    run_id = WorkflowRunner(runs_dir=root / "runs", data_catalog=catalog).run(
        recipe.compile(),
        run_config={"example": "09_advanced_models_run"},
    )
    print(run_id)


if __name__ == "__main__":
    main()
