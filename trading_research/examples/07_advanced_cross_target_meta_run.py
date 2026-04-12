"""Advanced required workflow:
- ret20 and vol20 base models
- diagnostics + projection
- downstream model on different target with mixed-task inputs
"""

from __future__ import annotations

from pathlib import Path

from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.mock import MockMarketDataVendor
from trading_research.models.registry import ModelSpec
from trading_research.recipes.diagnostics_meta import DiagnosticAwareMetaRecipe
from trading_research.workflow.runner import WorkflowRunner


def main() -> None:
    out_root = Path("runs/advanced_cross_target")
    data_root = out_root / "data"
    catalog = DataCatalog(data_root)
    vendor = MockMarketDataVendor()
    bars = vendor.fetch_bars(["AAPL", "MSFT", "NVDA", "QQQ"])
    catalog.persist_processed_bars("advanced_universe", bars)

    recipe = DiagnosticAwareMetaRecipe(
        dataset_name="advanced_universe",
        universe=("AAPL", "MSFT", "NVDA", "QQQ"),
        final_task_name="forward_return_40",
        final_model_spec=ModelSpec(name="meta_ret40", algorithm="ridge", params={"alpha": 0.7}),
    )
    graph = recipe.compile()
    runner = WorkflowRunner(runs_dir=out_root / "engine_runs", data_catalog=catalog)
    run_id = runner.run(graph, run_config={"example": "07_advanced_cross_target_meta_run"})
    print(f"completed run: {run_id}")
    print(f"run dir: {(out_root / 'engine_runs' / run_id).resolve()}")


if __name__ == "__main__":
    main()
