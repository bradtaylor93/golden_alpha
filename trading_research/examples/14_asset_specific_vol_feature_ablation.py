"""Asset-specific ablation: does predicted volatility improve return models?"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_research.analysis.sanity_checks import run_sanity_checks
from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.yahoo import YahooMarketDataVendor
from trading_research.models.registry import ModelSpec, TrainingRecipe
from trading_research.validation.splits import ValidationSpec
from trading_research.workflow.graph import WorkflowGraph
from trading_research.workflow.inspectors import load_run
from trading_research.workflow.nodes import (
    DataNode,
    FeatureNode,
    FoldPlanNode,
    ModelInputs,
    ModelNode,
    TargetNode,
)
from trading_research.workflow.runner import WorkflowRunner


def _build_graph(dataset_name: str) -> WorkflowGraph:
    graph = WorkflowGraph(name="asset_specific_vol_feature_ablation")
    graph.add_node(DataNode(name="bars", universe=("SPY",), dataset_name=dataset_name))
    graph.add_node(
        FoldPlanNode(
            name="folds_outer",
            depends_on=("bars",),
            validation=ValidationSpec(
                split_type="expanding",
                min_train_size=24 * 90,
                test_size=24 * 15,
                embargo=4,
                rolling_train_size=None,
                mode="pooled",
            ),
            level="outer",
        )
    )
    graph.add_node(TargetNode(name="target_return", depends_on=("bars",), task_name="forward_return_20"))
    graph.add_node(
        TargetNode(
            name="target_volatility",
            depends_on=("bars",),
            task_name="forward_realized_volatility_20",
        )
    )
    graph.add_node(
        FeatureNode(
            name="features_base",
            depends_on=("bars",),
            family_name="baseline",
            params={"lookbacks": [1, 2, 4, 10, 20, 40, 80, 120]},
        )
    )
    graph.add_node(
        FeatureNode(
            name="features_long",
            depends_on=("bars",),
            family_name="research_pack",
            params={
                "lookbacks": [2, 4, 10, 20, 40, 80, 120, 160],
                "short_window": 10,
                "medium_window": 40,
                "long_window": 160,
            },
        )
    )

    graph.add_node(
        ModelNode(
            name="vol_context",
            depends_on=("features_long", "target_volatility", "folds_outer"),
            model_spec=ModelSpec("vol_context", "kernel_ridge", {"alpha": 0.4, "gamma": 0.12}),
            training_recipe=TrainingRecipe(save_training_snapshot=True),
            inputs=ModelInputs(
                feature_tables=["features_long"],
                target_tables=["target_volatility"],
                fold_plans=["folds_outer"],
                standardize_features=True,
                clip_quantiles=(0.01, 0.99),
                max_features=28,
            ),
        )
    )

    # Normal (single-layer) return models.
    graph.add_node(
        ModelNode(
            name="ret_normal_no_vol",
            depends_on=("features_base", "features_long", "target_return", "folds_outer"),
            model_spec=ModelSpec("ret_normal_no_vol", "ridge", {"alpha": 1.3}),
            training_recipe=TrainingRecipe(save_training_snapshot=True),
            inputs=ModelInputs(
                feature_tables=["features_base", "features_long"],
                target_tables=["target_return"],
                fold_plans=["folds_outer"],
                standardize_features=True,
                clip_quantiles=(0.01, 0.99),
                max_features=32,
                feature_exclude_regex=[".*dollar_volume.*"],
            ),
        )
    )
    graph.add_node(
        ModelNode(
            name="ret_normal_with_vol",
            depends_on=("features_base", "features_long", "vol_context", "target_return", "folds_outer"),
            model_spec=ModelSpec("ret_normal_with_vol", "ridge", {"alpha": 1.3}),
            training_recipe=TrainingRecipe(save_training_snapshot=True),
            inputs=ModelInputs(
                feature_tables=["features_base", "features_long"],
                prediction_tables=["vol_context"],
                target_tables=["target_return"],
                fold_plans=["folds_outer"],
                standardize_features=True,
                clip_quantiles=(0.01, 0.99),
                max_features=36,
                feature_exclude_regex=[".*dollar_volume.*"],
            ),
        )
    )

    # Meta branch.
    graph.add_node(
        ModelNode(
            name="ret_base_kernel",
            depends_on=("features_long", "target_return", "folds_outer"),
            model_spec=ModelSpec("ret_base_kernel", "kernel_ridge", {"alpha": 0.35, "gamma": 0.14}),
            training_recipe=TrainingRecipe(save_training_snapshot=True),
            inputs=ModelInputs(
                feature_tables=["features_long"],
                target_tables=["target_return"],
                fold_plans=["folds_outer"],
                standardize_features=True,
                clip_quantiles=(0.01, 0.99),
                max_features=28,
            ),
        )
    )
    graph.add_node(
        ModelNode(
            name="ret_base_loess",
            depends_on=("features_base", "target_return", "folds_outer"),
            model_spec=ModelSpec("ret_base_loess", "loess", {"frac": 0.2, "ridge": 1e-4}),
            training_recipe=TrainingRecipe(save_training_snapshot=True),
            inputs=ModelInputs(
                feature_tables=["features_base"],
                target_tables=["target_return"],
                fold_plans=["folds_outer"],
                standardize_features=True,
                clip_quantiles=(0.01, 0.99),
                max_features=20,
            ),
        )
    )
    graph.add_node(
        ModelNode(
            name="ret_meta_no_vol",
            depends_on=("features_long", "ret_base_kernel", "ret_base_loess", "target_return", "folds_outer"),
            model_spec=ModelSpec("ret_meta_no_vol", "ridge", {"alpha": 2.2}),
            training_recipe=TrainingRecipe(save_training_snapshot=True),
            inputs=ModelInputs(
                feature_tables=["features_long"],
                prediction_tables=["ret_base_kernel", "ret_base_loess"],
                target_tables=["target_return"],
                fold_plans=["folds_outer"],
                standardize_features=True,
                clip_quantiles=(0.01, 0.99),
                max_features=22,
                feature_include_regex=["^(ret_|vol_|mom_|pred_).*"],
                feature_exclude_regex=[".*dollar_volume.*"],
            ),
        )
    )
    graph.add_node(
        ModelNode(
            name="ret_meta_with_vol",
            depends_on=(
                "features_long",
                "ret_base_kernel",
                "ret_base_loess",
                "vol_context",
                "target_return",
                "folds_outer",
            ),
            model_spec=ModelSpec("ret_meta_with_vol", "ridge", {"alpha": 2.2}),
            training_recipe=TrainingRecipe(save_training_snapshot=True),
            inputs=ModelInputs(
                feature_tables=["features_long"],
                prediction_tables=["ret_base_kernel", "ret_base_loess", "vol_context"],
                target_tables=["target_return"],
                fold_plans=["folds_outer"],
                standardize_features=True,
                clip_quantiles=(0.01, 0.99),
                max_features=24,
                feature_include_regex=["^(ret_|vol_|mom_|pred_).*"],
                feature_exclude_regex=[".*dollar_volume.*"],
            ),
        )
    )
    return graph


def _metrics(pred: pd.DataFrame) -> dict[str, float]:
    test = pred[pred["split_role"] == "test"].copy()
    if test.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson_corr": float("nan"), "n_test": 0.0}
    test["target"] = pd.to_numeric(test["target"], errors="coerce")
    test["prediction"] = pd.to_numeric(test["prediction"], errors="coerce")
    test = test.dropna(subset=["target", "prediction"])
    if test.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson_corr": float("nan"), "n_test": 0.0}
    err = test["target"] - test["prediction"]
    return {
        "mae": float(err.abs().mean()),
        "rmse": float((err.pow(2).mean()) ** 0.5),
        "pearson_corr": float(test["target"].corr(test["prediction"])),
        "n_test": float(len(test)),
    }


def main() -> None:
    root = Path("trading_research/examples/_output/14_asset_specific_vol_feature_ablation")
    reports_dir = root / "reports"
    runs_dir = root / "runs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    catalog = DataCatalog(root / "data")

    bars = YahooMarketDataVendor().fetch_bars(["SPY"], period="2y", interval="1h")
    dataset_name = "spy_2y_1h"
    catalog.persist_processed_bars(
        dataset_name,
        bars,
        metadata={"source": "yahoo", "interval": "1h", "period": "2y", "asset": "SPY"},
    )

    runner = WorkflowRunner(runs_dir=runs_dir, data_catalog=catalog)
    run_id = runner.run(_build_graph(dataset_name), run_config={"experiment": "predicted_vol_feature_ablation"})
    inspector = load_run(run_id, runs_dir)

    candidates = [
        "ret_normal_no_vol",
        "ret_normal_with_vol",
        "ret_meta_no_vol",
        "ret_meta_with_vol",
    ]
    rows: list[dict[str, float | str]] = []
    for node in candidates:
        rows.append({"node_name": node, **_metrics(inspector.load_predictions(node))})
    perf = pd.DataFrame(rows).sort_values("rmse", ascending=True).reset_index(drop=True)

    def _rmse(node: str) -> float:
        row = perf[perf["node_name"] == node]
        return float(row.iloc[0]["rmse"]) if not row.empty else float("nan")

    summary = {
        "run_id": run_id,
        "normal_no_vol_rmse": _rmse("ret_normal_no_vol"),
        "normal_with_vol_rmse": _rmse("ret_normal_with_vol"),
        "normal_delta_with_minus_without": _rmse("ret_normal_with_vol") - _rmse("ret_normal_no_vol"),
        "meta_no_vol_rmse": _rmse("ret_meta_no_vol"),
        "meta_with_vol_rmse": _rmse("ret_meta_with_vol"),
        "meta_delta_with_minus_without": _rmse("ret_meta_with_vol") - _rmse("ret_meta_no_vol"),
    }

    sanity = run_sanity_checks(run_id=run_id, runs_root=runs_dir)
    failed = sanity[~sanity["passed"]]
    summary["sanity_failed_checks"] = int(len(failed))
    summary["sanity_critical_failures"] = int(
        len(failed[failed["severity"].astype(str).str.lower() == "critical"])
    )

    perf.to_csv(reports_dir / "candidate_oos_performance.csv", index=False)
    sanity.to_csv(reports_dir / "no_leakage_assertions.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("run_id:", run_id)
    print("reports:", reports_dir.resolve())
    print(
        "normal delta (with_vol - no_vol):",
        summary["normal_delta_with_minus_without"],
        "meta delta (with_vol - no_vol):",
        summary["meta_delta_with_minus_without"],
    )
    print(
        "sanity_failed_checks:",
        summary["sanity_failed_checks"],
        "critical:",
        summary["sanity_critical_failures"],
    )


if __name__ == "__main__":
    main()

