"""Nested walk-forward return/vol horizon recipe with error meta-modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from trading_research.models.registry import ModelSpec, TrainingRecipe
from trading_research.recipes.base import Recipe
from trading_research.validation.splits import ValidationSpec
from trading_research.workflow.graph import WorkflowGraph
from trading_research.workflow.inspectors import load_run
from trading_research.workflow.nodes import (
    DataNode,
    DerivedTargetInputs,
    DerivedTargetNode,
    DiagnosticInputs,
    DiagnosticNode,
    FeatureNode,
    FoldPlanNode,
    ModelInputs,
    ModelNode,
    ProjectionInputs,
    ProjectionNode,
    TargetNode,
)
from trading_research.workflow.runner import WorkflowRunner


@dataclass(frozen=True)
class HorizonErrorMetaRecipe(Recipe):
    """Return/vol recipe across multiple horizons with error-model nodes.

    This recipe creates:
    - return and volatility target models for each horizon
    - naive baseline model nodes for uplift comparison
    - error meta-model nodes trained on projected diagnostics + base features
    """

    dataset_name: str = "spy_1h_2y"
    universe: tuple[str, ...] = ("SPY",)
    task_name: str = "forward_return_20"
    run_name: str = "horizon_error_meta"
    horizons: tuple[int, ...] = (4, 10, 20, 40)
    outer_validation_spec: ValidationSpec = ValidationSpec(
        split_type="expanding",
        min_train_size=24 * 120,  # ~6 months of hourly bars
        test_size=24 * 20,  # ~1 month
        embargo=4,
        mode="pooled",
    )
    inner_validation_spec: ValidationSpec = ValidationSpec(
        split_type="expanding",
        min_train_size=24 * 60,  # ~3 months
        test_size=24 * 10,  # ~2 weeks
        embargo=4,
        mode="pooled",
    )
    low_s_model: ModelSpec = field(
        default_factory=lambda: ModelSpec("low_s_loess", "loess", {"frac": 0.2, "ridge": 1e-4})
    )
    rich_model: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            "rich_kernel_ridge", "kernel_ridge", {"alpha": 0.4, "gamma": 0.2}
        )
    )

    def compile(self) -> WorkflowGraph:
        graph = WorkflowGraph(name=self.run_name)
        graph.add_node(DataNode(name="bars", universe=self.universe, dataset_name=self.dataset_name))
        graph.add_node(
            FoldPlanNode(
                name="folds_outer",
                depends_on=("bars",),
                validation=self.outer_validation_spec,
                level="outer",
            )
        )
        graph.add_node(
            FoldPlanNode(
                name="folds_inner",
                depends_on=("bars",),
                validation=self.inner_validation_spec,
                level="inner",
                build_inner=True,
                inner_folds=6,
            )
        )
        graph.add_node(
            FeatureNode(
                name="features_base",
                depends_on=("bars",),
                family_name="baseline",
                params={"lookbacks": [1, 2, 4, 10, 20, 40]},
            )
        )
        graph.add_node(
            FeatureNode(
                name="features_rich",
                depends_on=("bars",),
                family_name="research_pack",
                params={"short_window": 6, "medium_window": 20, "long_window": 60},
            )
        )

        for horizon in self.horizons:
            ret_task = f"forward_return_{horizon}"
            vol_task = f"forward_realized_volatility_{horizon}"

            graph.add_node(
                TargetNode(
                    name=f"target_ret_{horizon}",
                    depends_on=("bars",),
                    task_name=ret_task,
                )
            )
            graph.add_node(
                TargetNode(
                    name=f"target_vol_{horizon}",
                    depends_on=("bars",),
                    task_name=vol_task,
                )
            )

            # Naive persistence baselines using low_s model on compact features.
            graph.add_node(
                ModelNode(
                    name=f"baseline_ret_{horizon}",
                    depends_on=("features_base", f"target_ret_{horizon}", "folds_outer"),
                    model_spec=ModelSpec(
                        name=f"baseline_ret_{horizon}",
                        algorithm="ridge",
                        params={"alpha": 1e-6},
                    ),
                    training_recipe=TrainingRecipe(save_training_snapshot=True),
                    inputs=ModelInputs(
                        feature_tables=["features_base"],
                        target_tables=[f"target_ret_{horizon}"],
                        fold_plans=["folds_outer"],
                    ),
                )
            )
            graph.add_node(
                ModelNode(
                    name=f"baseline_vol_{horizon}",
                    depends_on=("features_base", f"target_vol_{horizon}", "folds_outer"),
                    model_spec=ModelSpec(
                        name=f"baseline_vol_{horizon}",
                        algorithm="ridge",
                        params={"alpha": 1e-6},
                    ),
                    training_recipe=TrainingRecipe(save_training_snapshot=True),
                    inputs=ModelInputs(
                        feature_tables=["features_base"],
                        target_tables=[f"target_vol_{horizon}"],
                        fold_plans=["folds_outer"],
                    ),
                )
            )

            # Candidate models for returns (low_s vs rich).
            graph.add_node(
                ModelNode(
                    name=f"ret_low_s_{horizon}",
                    depends_on=("features_base", f"target_ret_{horizon}", "folds_outer", "folds_inner"),
                    model_spec=ModelSpec(
                        name=f"ret_low_s_{horizon}",
                        algorithm=self.low_s_model.algorithm,
                        params=self.low_s_model.params,
                    ),
                    training_recipe=TrainingRecipe(
                        selection_validation=self.inner_validation_spec,
                        save_training_snapshot=True,
                    ),
                    inputs=ModelInputs(
                        feature_tables=["features_base"],
                        target_tables=[f"target_ret_{horizon}"],
                        fold_plans=["folds_outer"],
                    ),
                )
            )
            graph.add_node(
                ModelNode(
                    name=f"ret_rich_{horizon}",
                    depends_on=("features_rich", f"target_ret_{horizon}", "folds_outer", "folds_inner"),
                    model_spec=ModelSpec(
                        name=f"ret_rich_{horizon}",
                        algorithm=self.rich_model.algorithm,
                        params=self.rich_model.params,
                    ),
                    training_recipe=TrainingRecipe(
                        selection_validation=self.inner_validation_spec,
                        save_training_snapshot=True,
                    ),
                    inputs=ModelInputs(
                        feature_tables=["features_rich"],
                        target_tables=[f"target_ret_{horizon}"],
                        fold_plans=["folds_outer"],
                    ),
                )
            )
            graph.add_node(
                DiagnosticNode(
                    name=f"diag_ret_rich_{horizon}",
                    depends_on=(f"ret_rich_{horizon}",),
                    diagnostic_types=["train_test_deviance_gap", "feature_drift", "residual_std_by_asset"],
                    labels={"task_family": "return", "horizon": horizon, "role": "base_diagnostic"},
                    inputs=DiagnosticInputs(prediction_tables=[f"ret_rich_{horizon}"]),
                )
            )
            graph.add_node(
                ProjectionNode(
                    name=f"proj_ret_rich_{horizon}",
                    depends_on=("bars", f"diag_ret_rich_{horizon}"),
                    labels={"task_family": "return", "horizon": horizon, "role": "diagnostic_projection"},
                    inputs=ProjectionInputs(
                        bars=["bars"],
                        fold_plans=["folds_outer"],
                        diagnostic_tables=[f"diag_ret_rich_{horizon}"],
                        lag=1,
                        mode="prev_fold_to_test",
                    ),
                )
            )
            graph.add_node(
                DerivedTargetNode(
                    name=f"derived_abs_error_ret_rich_{horizon}",
                    depends_on=(f"ret_rich_{horizon}",),
                    task_name=f"derived_abs_error_ret_{horizon}",
                    inputs=DerivedTargetInputs(prediction_tables=[f"ret_rich_{horizon}"]),
                    target_kind="abs_error",
                    source_prediction_node=f"ret_rich_{horizon}",
                    labels={"task_family": "return", "horizon": horizon, "role": "derived_target"},
                )
            )

            # Candidate models for volatility.
            graph.add_node(
                ModelNode(
                    name=f"vol_low_s_{horizon}",
                    depends_on=("features_base", f"target_vol_{horizon}", "folds_outer", "folds_inner"),
                    model_spec=ModelSpec(
                        name=f"vol_low_s_{horizon}",
                        algorithm=self.low_s_model.algorithm,
                        params=self.low_s_model.params,
                    ),
                    training_recipe=TrainingRecipe(
                        selection_validation=self.inner_validation_spec,
                        save_training_snapshot=True,
                    ),
                    inputs=ModelInputs(
                        feature_tables=["features_base"],
                        target_tables=[f"target_vol_{horizon}"],
                        fold_plans=["folds_outer"],
                    ),
                )
            )
            graph.add_node(
                ModelNode(
                    name=f"vol_rich_{horizon}",
                    depends_on=("features_rich", f"target_vol_{horizon}", "folds_outer", "folds_inner"),
                    model_spec=ModelSpec(
                        name=f"vol_rich_{horizon}",
                        algorithm=self.rich_model.algorithm,
                        params=self.rich_model.params,
                    ),
                    training_recipe=TrainingRecipe(
                        selection_validation=self.inner_validation_spec,
                        save_training_snapshot=True,
                    ),
                    inputs=ModelInputs(
                        feature_tables=["features_rich"],
                        target_tables=[f"target_vol_{horizon}"],
                        fold_plans=["folds_outer"],
                    ),
                )
            )

            # Error meta-models (predict absolute error proxy via same targets for now;
            # evaluation script remaps to absolute error label from base model outputs).
            graph.add_node(
                ModelNode(
                    name=f"err_meta_ret_{horizon}",
                    depends_on=(
                        "features_rich",
                        f"proj_ret_rich_{horizon}",
                        f"ret_low_s_{horizon}",
                        f"ret_rich_{horizon}",
                        f"derived_abs_error_ret_rich_{horizon}",
                        "folds_outer",
                    ),
                    model_spec=ModelSpec(
                        name=f"err_meta_ret_{horizon}",
                        algorithm="kernel_ridge",
                        params={"alpha": 0.6, "gamma": 0.15},
                    ),
                    training_recipe=TrainingRecipe(save_training_snapshot=True),
                    inputs=ModelInputs(
                        feature_tables=["features_rich", f"proj_ret_rich_{horizon}"],
                        prediction_tables=[f"ret_low_s_{horizon}", f"ret_rich_{horizon}"],
                        target_tables=[f"derived_abs_error_ret_rich_{horizon}"],
                        fold_plans=["folds_outer"],
                    ),
                )
            )

        return graph


@dataclass(frozen=True)
class HorizonErrorMetaRunResult:
    run_id: str
    summary: pd.DataFrame
    fold_performance: pd.DataFrame
    ci_performance: pd.DataFrame
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame


def _safe_float(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def _eval_prediction_table(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "coverage95": float("nan")}
    err = df["target"] - df["prediction"]
    abs_err = err.abs()
    mae = float(abs_err.mean())
    rmse = float((err**2).mean() ** 0.5)
    # Simple empirical 95% CI under Gaussian residual assumption.
    sigma = float(err.std(ddof=0))
    lower = df["prediction"] - 1.96 * sigma
    upper = df["prediction"] + 1.96 * sigma
    coverage = float(((df["target"] >= lower) & (df["target"] <= upper)).mean())
    return {"mae": mae, "rmse": rmse, "coverage95": coverage}


def _infer_horizon_from_node(node_name: str) -> int | None:
    parts = node_name.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return None


def _artifact_horizon_and_family(ref) -> tuple[int | None, str]:
    """Prefer structured artifact metadata over node-name parsing."""
    metadata = ref.metadata if isinstance(getattr(ref, "metadata", None), dict) else {}
    labels = metadata.get("node_labels", {}) if isinstance(metadata.get("node_labels", {}), dict) else {}
    horizon = labels.get("horizon")
    if isinstance(horizon, bool):
        horizon = None
    if horizon is not None:
        try:
            horizon = int(horizon)
        except Exception:
            horizon = None
    task_family = labels.get("task_family")
    if not isinstance(task_family, str):
        task_family = ""
    if horizon is None:
        horizon = _infer_horizon_from_node(ref.node_name)
    if not task_family:
        task_family = "return" if "_ret_" in ref.node_name or "ret_" in ref.node_name else "volatility"
    return horizon, task_family


def run_horizon_error_meta(
    *,
    recipe: HorizonErrorMetaRecipe,
    catalog,
    runs_root: Path | str,
) -> HorizonErrorMetaRunResult:
    """Compile, execute, and evaluate the horizon error meta recipe."""
    runner = WorkflowRunner(runs_dir=runs_root, data_catalog=catalog)
    run_id = runner.run(recipe.compile(), run_config={"recipe": "horizon_error_meta"})
    inspector = load_run(run_id, runs_root)

    pred_refs = inspector.artifact_store.index.list_artifacts(artifact_type="PredictionArtifact")
    diag_refs = inspector.artifact_store.index.list_artifacts(artifact_type="DiagnosticArtifact")

    pred_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    ci_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for ref in pred_refs:
        frame = inspector.artifact_store.load(ref.artifact_id)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["node_name"] = ref.node_name
        horizon, task = _artifact_horizon_and_family(ref)
        frame["horizon"] = horizon
        frame["task_family"] = task
        pred_frames.append(frame)

        test_df = frame[frame["split_role"] == "test"].copy()
        overall = _eval_prediction_table(test_df)
        summary_rows.append(
            {
                "node_name": ref.node_name,
                "task_family": task,
                "horizon": horizon,
                **overall,
            }
        )

        if "outer_fold_id" in test_df.columns:
            for fold_id, fold_df in test_df.groupby("outer_fold_id", observed=True):
                stats = _eval_prediction_table(fold_df)
                fold_rows.append(
                    {
                        "node_name": ref.node_name,
                        "task_family": task,
                        "horizon": horizon,
                        "outer_fold_id": int(fold_id),
                        **stats,
                    }
                )

        # CI quality for return-focused nodes only.
        if task == "return" and not test_df.empty:
            err = test_df["target"] - test_df["prediction"]
            abs_err = err.abs()
            # Proxy error model output: rolling absolute error mean.
            pred_abs_err = abs_err.rolling(window=20, min_periods=5).mean().bfill()
            lower = test_df["prediction"] - 1.96 * pred_abs_err
            upper = test_df["prediction"] + 1.96 * pred_abs_err
            coverage = float(((test_df["target"] >= lower) & (test_df["target"] <= upper)).mean())
            avg_width = float((upper - lower).mean())
            ci_rows.append(
                {
                    "node_name": ref.node_name,
                    "horizon": horizon,
                    "coverage95": coverage,
                    "avg_interval_width": avg_width,
                }
            )

    diagnostics = (
        pd.concat(
            [inspector.artifact_store.load(ref.artifact_id).assign(node_name=ref.node_name) for ref in diag_refs],
            ignore_index=True,
            sort=False,
        )
        if diag_refs
        else pd.DataFrame()
    )

    return HorizonErrorMetaRunResult(
        run_id=run_id,
        summary=pd.DataFrame(summary_rows),
        fold_performance=pd.DataFrame(fold_rows),
        ci_performance=pd.DataFrame(ci_rows),
        predictions=pd.concat(pred_frames, ignore_index=True, sort=False) if pred_frames else pd.DataFrame(),
        diagnostics=diagnostics,
    )


def write_recipe_outputs(result: HorizonErrorMetaRunResult, out_dir: Path | str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(out / "summary.csv", index=False)
    result.fold_performance.to_csv(out / "fold_performance.csv", index=False)
    result.ci_performance.to_csv(out / "ci_performance.csv", index=False)
    result.predictions.to_parquet(out / "predictions.parquet", index=False)
    result.diagnostics.to_parquet(out / "diagnostics.parquet", index=False)

