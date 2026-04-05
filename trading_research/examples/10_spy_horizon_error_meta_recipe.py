"""Run SPY 2-year Yahoo 1h nested horizon + error-meta recipe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_research.analysis.dashboard_data import load_run_analytics
from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.yahoo import YahooMarketDataVendor
from trading_research.recipes.horizon_error_meta import HorizonErrorMetaRecipe
from trading_research.workflow.inspectors import load_run
from trading_research.workflow.runner import WorkflowRunner


def main() -> None:
    root = Path("trading_research/examples/_output/10_spy_horizon_error_meta")
    catalog = DataCatalog(root / "data")
    bars = YahooMarketDataVendor().fetch_bars(
        assets=["SPY"],
        period="2y",
        interval="1h",
    )
    catalog.persist_processed_bars("spy_2y_1h", bars, metadata={"source": "yahoo", "period": "2y", "interval": "1h"})
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    recipe = HorizonErrorMetaRecipe(dataset_name="spy_2y_1h", universe=("SPY",), run_name="spy_horizon_error_meta")
    graph = recipe.compile()
    run_id = WorkflowRunner(runs_dir=root / "runs", data_catalog=catalog).run(
        graph,
        run_config={
            "recipe": "HorizonErrorMetaRecipe",
            "target_family": ["return", "realized_volatility"],
            "horizons": [4, 10, 20, 40],
            "asset_scope": ["SPY"],
            "model_family_candidates": ["loess", "kernel_ridge"],
            "ci_method": "pred ± 1.96 * predicted_abs_error",
            "validation_recommendation": {
                "outer": {"split_type": "expanding", "min_train_size": 24 * 120, "test_size": 24 * 20},
                "inner": {"split_type": "expanding", "min_train_size": 24 * 60, "test_size": 24 * 10},
            },
        },
    )

    inspector = load_run(run_id, root / "runs")
    preds = []
    diagnostics = []
    for node in inspector.list_nodes():
        if node.startswith("ret_") or node.startswith("vol_") or node.startswith("baseline_"):
            try:
                frame = inspector.load_predictions(node)
                frame = frame.copy()
                frame["node_name"] = node
                preds.append(frame)
            except Exception:
                pass
            try:
                diag = inspector.load_artifact(node, artifact_type="DiagnosticArtifact")
                diag = diag.copy()
                diag["node_name"] = node
                diagnostics.append(diag)
            except Exception:
                pass

    pred_df = pd.concat(preds, ignore_index=True, sort=False) if preds else pd.DataFrame()
    diag_df = pd.concat(diagnostics, ignore_index=True, sort=False) if diagnostics else pd.DataFrame()

    if not pred_df.empty and {"target", "prediction"}.issubset(pred_df.columns):
        pred_df["abs_error"] = (pred_df["target"] - pred_df["prediction"]).abs()
        pred_df["predicted_abs_error_proxy"] = pred_df.groupby("node_name", observed=True)["abs_error"].transform(
            lambda s: s.rolling(window=24, min_periods=1).mean()
        )
        pred_df["ci_lower_95"] = pred_df["prediction"] - 1.96 * pred_df["predicted_abs_error_proxy"]
        pred_df["ci_upper_95"] = pred_df["prediction"] + 1.96 * pred_df["predicted_abs_error_proxy"]
        pred_df["ci_covered"] = (
            (pred_df["target"] >= pred_df["ci_lower_95"]) & (pred_df["target"] <= pred_df["ci_upper_95"])
        ).astype(float)

        fold_perf = (
            pred_df[pred_df["split_role"] == "test"]
            .groupby(["node_name", "outer_fold_id"], as_index=False)
            .agg(
                mae=("abs_error", "mean"),
                rmse=("abs_error", lambda s: float((s**2).mean() ** 0.5)),
                ci95_coverage=("ci_covered", "mean"),
            )
        )
        overall_perf = (
            pred_df[pred_df["split_role"] == "test"]
            .groupby(["node_name"], as_index=False)
            .agg(
                mae=("abs_error", "mean"),
                rmse=("abs_error", lambda s: float((s**2).mean() ** 0.5)),
                ci95_coverage=("ci_covered", "mean"),
            )
        )
    else:
        fold_perf = pd.DataFrame()
        overall_perf = pd.DataFrame()

    if not fold_perf.empty:
        fold_perf.to_csv(reports_dir / "fold_oos_performance.csv", index=False)
    if not overall_perf.empty:
        overall_perf.to_csv(reports_dir / "overall_oos_performance.csv", index=False)
    if not diag_df.empty:
        diag_df.to_csv(reports_dir / "diagnostics_by_fold.csv", index=False)

    analytics = load_run_analytics(run_id, root / "runs")
    (reports_dir / "quant_agent_handoff.json").write_text(
        pd.Series(analytics.handoff_payload).to_json(indent=2),
        encoding="utf-8",
    )
    print("run_id:", run_id)
    print("reports:", reports_dir.resolve())


if __name__ == "__main__":
    main()

