"""Example 6: selector and blend strategy run."""

from __future__ import annotations

from pathlib import Path

from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.mock import MockMarketDataVendor
from trading_research.models import ModelSpec
from trading_research.recipes import SelectorRecipe
from trading_research.workflow.inspectors import load_run
from trading_research.workflow.runner import WorkflowRunner


def main() -> None:
    root = Path("trading_research/examples/_output/06_selector")
    catalog = DataCatalog(root / "data")
    bars = MockMarketDataVendor().fetch_bars(["AAPL", "MSFT", "GOOG"])
    catalog.persist_processed_bars("default_universe", bars)

    recipe = SelectorRecipe(
        dataset_name="default_universe",
        universe=("AAPL", "MSFT", "GOOG"),
        task_name="forward_return_20",
        candidates=(
            ModelSpec(name="candidate_a", algorithm="ridge", params={"alpha": 0.1}),
            ModelSpec(name="candidate_b", algorithm="ridge", params={"alpha": 1.0}),
            ModelSpec(name="candidate_c", algorithm="ridge", params={"alpha": 5.0}),
        ),
        strategy="weighted_blend",
    )
    graph = recipe.compile()
    run_id = WorkflowRunner(runs_dir=root / "runs", data_catalog=catalog).run(
        graph,
        run_config={"example": "06_selector"},
    )
    print("run_id:", run_id)
    inspector = load_run(run_id, root / "runs")
    print("selection:")
    print(inspector.load_artifact("selector", artifact_type="SelectionArtifact").head())


if __name__ == "__main__":
    main()
