"""Example 5: diagnostic-aware meta model run."""

from __future__ import annotations

from pathlib import Path

from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.mock import MockMarketDataVendor
from trading_research.recipes.diagnostics_meta import DiagnosticAwareMetaRecipe
from trading_research.workflow.runner import WorkflowRunner


def main() -> None:
    root = Path("trading_research/examples/_output/05_diagnostic_meta")
    catalog = DataCatalog(root / "data")
    bars = MockMarketDataVendor().fetch_bars(["AAPL", "MSFT", "NVDA"])
    catalog.persist_processed_bars("default_universe", bars, metadata={"assets": 3})

    recipe = DiagnosticAwareMetaRecipe(dataset_name="default_universe")
    graph = recipe.compile()
    run_id = WorkflowRunner(runs_dir=root / "runs", data_catalog=catalog).run(
        graph,
        run_config={"example": "05_diagnostic_aware_meta_run"},
    )
    print(run_id)


if __name__ == "__main__":
    main()
