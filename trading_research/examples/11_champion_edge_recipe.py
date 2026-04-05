"""Run ambitious champion edge recipe on SPY Yahoo data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.yahoo import YahooMarketDataVendor
from trading_research.recipes.champion_edge import ChampionEdgeRecipe
from trading_research.workflow.inspectors import load_run
from trading_research.workflow.runner import WorkflowRunner


def _oos_metrics(pred: pd.DataFrame) -> dict[str, float]:
    test = pred[pred["split_role"] == "test"].copy()
    if test.empty:
        return {"mae": float("nan"), "rmse": float("nan")}
    err = test["target"] - test["prediction"]
    return {
        "mae": float(err.abs().mean()),
        "rmse": float((err.pow(2).mean()) ** 0.5),
    }


def main() -> None:
    root = Path("trading_research/examples/_output/11_champion_edge")
    root.mkdir(parents=True, exist_ok=True)
    catalog = DataCatalog(root / "data")

    bars = YahooMarketDataVendor().fetch_bars(
        assets=["SPY"],
        period="2y",
        interval="1h",
    )
    dataset_name = "spy_2y_1h"
    catalog.persist_processed_bars(
        dataset_name,
        bars,
        metadata={"source": "yahoo", "period": "2y", "interval": "1h"},
    )

    recipe = ChampionEdgeRecipe(dataset_name=dataset_name, universe=("SPY",))
    runner = WorkflowRunner(runs_dir=root / "runs", data_catalog=catalog)
    run_id = runner.run(
        recipe.compile(),
        run_config={
            "recipe": "ChampionEdgeRecipe",
            "description": "Ambitious multi-model, diagnostics-aware meta alpha pipeline.",
        },
    )

    inspector = load_run(run_id, root / "runs")
    candidate_nodes = [
        "ret_persistence_baseline",
        "ret_loess_base",
        "ret_kernel_rich",
        "ret_ridge_stable",
        "champion_meta_return",
    ]
    rows: list[dict[str, float | str]] = []
    for node in candidate_nodes:
        frame = inspector.load_predictions(node)
        rows.append({"node_name": node, **_oos_metrics(frame)})
    perf = pd.DataFrame(rows).sort_values("rmse", ascending=True).reset_index(drop=True)

    selector = inspector.load_artifact("champion_selector", artifact_type="SelectionArtifact")

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    perf.to_csv(reports / "candidate_oos_performance.csv", index=False)
    selector.to_csv(reports / "selector_decision.csv", index=False)

    print("run_id:", run_id)
    print("reports:", reports.resolve())
    print("best candidate by RMSE:", perf.iloc[0]["node_name"] if not perf.empty else "n/a")


if __name__ == "__main__":
    main()

