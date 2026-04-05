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


def _gate_meta_predictions(
    base: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    max_pred_abs_err: float = 0.012,
    max_pred_disagreement: float = 0.010,
) -> pd.DataFrame:
    """Fallback from meta to base when uncertainty proxies are elevated."""
    left = meta.copy()
    right = base[["timestamp", "asset", "split_role", "prediction"]].rename(
        columns={"prediction": "base_prediction"}
    )
    merged = left.merge(right, on=["timestamp", "asset", "split_role"], how="left")
    risk_cols = [c for c in merged.columns if c.startswith("pred_err_") or c.startswith("pred_disagreement_")]
    if not risk_cols:
        return meta
    risk_err_cols = [c for c in risk_cols if c.startswith("pred_err_")]
    risk_dis_cols = [c for c in risk_cols if c.startswith("pred_disagreement_")]
    pred_abs_err = (
        merged[risk_err_cols].mean(axis=1) if risk_err_cols else pd.Series(0.0, index=merged.index)
    )
    pred_dis = (
        merged[risk_dis_cols].mean(axis=1) if risk_dis_cols else pd.Series(0.0, index=merged.index)
    )
    high_risk = (pred_abs_err > max_pred_abs_err) | (pred_dis > max_pred_disagreement)
    merged["prediction"] = merged["prediction"].where(~high_risk, merged["base_prediction"])
    return merged[left.columns]


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

    # Meta-gated variant: fallback to stable ridge in high-uncertainty conditions.
    meta_pred = inspector.load_predictions("champion_meta_return")
    ridge_pred = inspector.load_predictions("ret_ridge_stable")
    gated_meta = _gate_meta_predictions(ridge_pred, meta_pred)
    rows.append({"node_name": "champion_meta_gated", **_oos_metrics(gated_meta)})

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

