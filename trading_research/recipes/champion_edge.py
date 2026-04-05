"""Ambitious multi-signal champion recipe for return prediction edge."""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_research.models.registry import ModelSpec, TrainingRecipe
from trading_research.recipes.base import Recipe
from trading_research.validation.splits import ValidationSpec
from trading_research.workflow.graph import WorkflowGraph
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
    SelectionInputs,
    SelectionNode,
    TargetNode,
)


@dataclass(frozen=True)
class ChampionEdgeRecipe(Recipe):
    """High-ambition recipe using model diversity + risk/error meta-signals.

    Design intent:
    - combine smooth local (loess), nonlinear global (kernel ridge), and stable linear (ridge) views
    - add risk context via volatility and directional probability models
    - generate explicit diagnostics and causally project them into row-level features
    - model absolute error, signed residual, and model disagreement as secondary targets
    - train a final meta model that ingests base predictions + risk + projected diagnostics
    """

    run_name: str = "champion_edge_recipe"
    return_horizon: int = 20
    volatility_horizon: int = 20
    direction_horizon: int = 20
    validation_spec: ValidationSpec = ValidationSpec(
        split_type="expanding",
        min_train_size=24 * 90,
        test_size=24 * 15,
        embargo=4,
        rolling_train_size=None,
        mode="pooled",
    )
    loess_model: ModelSpec = field(
        default_factory=lambda: ModelSpec("ret_loess_base", "loess", {"frac": 0.2, "ridge": 1e-4})
    )
    kernel_model: ModelSpec = field(
        default_factory=lambda: ModelSpec("ret_kernel_rich", "kernel_ridge", {"alpha": 0.35, "gamma": 0.2})
    )
    ridge_model: ModelSpec = field(
        default_factory=lambda: ModelSpec("ret_ridge_stable", "ridge", {"alpha": 1.25})
    )
    meta_model: ModelSpec = field(
        default_factory=lambda: ModelSpec("champion_meta_return", "kernel_ridge", {"alpha": 0.5, "gamma": 0.12})
    )

    def compile(self) -> WorkflowGraph:
        graph = WorkflowGraph(name=self.run_name)
        graph.add_node(DataNode(name="bars", universe=self.universe, dataset_name=self.dataset_name))
        graph.add_node(
            FoldPlanNode(
                name="folds_outer",
                depends_on=("bars",),
                validation=self.validation_spec,
                level="outer",
                labels={"role": "cv"},
            )
        )

        target_return_name = f"forward_return_{self.return_horizon}"
        target_vol_name = f"forward_realized_volatility_{self.volatility_horizon}"
        target_dir_name = f"up_down_{self.direction_horizon}"

        graph.add_node(
            TargetNode(
                name="target_return",
                depends_on=("bars",),
                task_name=target_return_name,
                labels={"task_family": "return", "horizon": self.return_horizon},
            )
        )
        graph.add_node(
            TargetNode(
                name="target_volatility",
                depends_on=("bars",),
                task_name=target_vol_name,
                labels={"task_family": "volatility", "horizon": self.volatility_horizon},
            )
        )
        graph.add_node(
            TargetNode(
                name="target_direction",
                depends_on=("bars",),
                task_name=target_dir_name,
                labels={"task_family": "classification", "horizon": self.direction_horizon},
            )
        )

        graph.add_node(
            FeatureNode(
                name="features_base",
                depends_on=("bars",),
                family_name="baseline",
                params={"lookbacks": [1, 2, 4, 10, 20, 40]},
                labels={"role": "features", "pack": "base"},
            )
        )
        graph.add_node(
            FeatureNode(
                name="features_rich",
                depends_on=("bars",),
                family_name="research_pack",
                params={"short_window": 6, "medium_window": 20, "long_window": 60},
                labels={"role": "features", "pack": "rich"},
            )
        )
        graph.add_node(
            FeatureNode(
                name="features_regime",
                depends_on=("bars",),
                family_name="regime",
                params={"window": 24},
                labels={"role": "features", "pack": "regime"},
            )
        )

        # Base alpha candidates (different inductive biases).
        graph.add_node(
            ModelNode(
                name="ret_persistence_baseline",
                depends_on=("features_base", "target_return", "folds_outer"),
                model_spec=ModelSpec("ret_persistence_baseline", "ridge", {"alpha": 1e-6}),
                training_recipe=TrainingRecipe(save_training_snapshot=True),
                labels={"task_family": "return", "role": "baseline"},
                inputs=ModelInputs(
                    feature_tables=["features_base"],
                    target_tables=["target_return"],
                    fold_plans=["folds_outer"],
                ),
            )
        )
        graph.add_node(
            ModelNode(
                name="ret_loess_base",
                depends_on=("features_base", "features_regime", "target_return", "folds_outer"),
                model_spec=self.loess_model,
                training_recipe=TrainingRecipe(save_training_snapshot=True),
                labels={"task_family": "return", "role": "base_alpha"},
                inputs=ModelInputs(
                    feature_tables=["features_base", "features_regime"],
                    target_tables=["target_return"],
                    fold_plans=["folds_outer"],
                ),
            )
        )
        graph.add_node(
            ModelNode(
                name="ret_kernel_rich",
                depends_on=("features_rich", "features_regime", "target_return", "folds_outer"),
                model_spec=self.kernel_model,
                training_recipe=TrainingRecipe(save_training_snapshot=True),
                labels={"task_family": "return", "role": "base_alpha"},
                inputs=ModelInputs(
                    feature_tables=["features_rich", "features_regime"],
                    target_tables=["target_return"],
                    fold_plans=["folds_outer"],
                ),
            )
        )
        graph.add_node(
            ModelNode(
                name="ret_ridge_stable",
                depends_on=("features_base", "features_regime", "target_return", "folds_outer"),
                model_spec=self.ridge_model,
                training_recipe=TrainingRecipe(save_training_snapshot=True),
                labels={"task_family": "return", "role": "base_alpha"},
                inputs=ModelInputs(
                    feature_tables=["features_base", "features_regime"],
                    target_tables=["target_return"],
                    fold_plans=["folds_outer"],
                ),
            )
        )

        # Risk and direction context models.
        graph.add_node(
            ModelNode(
                name="vol_kernel_risk",
                depends_on=("features_rich", "features_regime", "target_volatility", "folds_outer"),
                model_spec=ModelSpec("vol_kernel_risk", "kernel_ridge", {"alpha": 0.45, "gamma": 0.18}),
                training_recipe=TrainingRecipe(save_training_snapshot=True),
                labels={"task_family": "volatility", "role": "risk_context"},
                inputs=ModelInputs(
                    feature_tables=["features_rich", "features_regime"],
                    target_tables=["target_volatility"],
                    fold_plans=["folds_outer"],
                ),
            )
        )
        graph.add_node(
            ModelNode(
                name="dir_logistic_prob",
                depends_on=("features_rich", "features_regime", "target_direction", "folds_outer"),
                model_spec=ModelSpec(
                    "dir_logistic_prob",
                    "logistic",
                    {"learning_rate": 0.04, "steps": 450, "l2": 8e-4},
                ),
                training_recipe=TrainingRecipe(save_training_snapshot=True),
                labels={"task_family": "classification", "role": "risk_context"},
                inputs=ModelInputs(
                    feature_tables=["features_rich", "features_regime"],
                    target_tables=["target_direction"],
                    fold_plans=["folds_outer"],
                ),
            )
        )

        # Explicit reusable diagnostics.
        graph.add_node(
            DiagnosticNode(
                name="diag_ret_loess",
                depends_on=("ret_loess_base",),
                diagnostic_types=["train_test_deviance_gap", "feature_drift", "residual_std_by_asset"],
                labels={"task_family": "return", "role": "diagnostic"},
                inputs=DiagnosticInputs(
                    prediction_tables=["ret_loess_base"],
                    feature_tables=["features_base"],
                ),
            )
        )
        graph.add_node(
            DiagnosticNode(
                name="diag_ret_kernel",
                depends_on=("ret_kernel_rich",),
                diagnostic_types=["train_test_deviance_gap", "feature_drift", "residual_std_by_asset"],
                labels={"task_family": "return", "role": "diagnostic"},
                inputs=DiagnosticInputs(
                    prediction_tables=["ret_kernel_rich"],
                    feature_tables=["features_rich"],
                ),
            )
        )
        graph.add_node(
            DiagnosticNode(
                name="diag_ret_ridge",
                depends_on=("ret_ridge_stable",),
                diagnostic_types=["train_test_deviance_gap", "feature_drift", "residual_std_by_asset"],
                labels={"task_family": "return", "role": "diagnostic"},
                inputs=DiagnosticInputs(
                    prediction_tables=["ret_ridge_stable"],
                    feature_tables=["features_base"],
                ),
            )
        )
        graph.add_node(
            DiagnosticNode(
                name="diag_dir_prob",
                depends_on=("dir_logistic_prob",),
                diagnostic_types=["train_test_deviance_gap", "calibration_gap"],
                labels={"task_family": "classification", "role": "diagnostic"},
                inputs=DiagnosticInputs(
                    prediction_tables=["dir_logistic_prob"],
                    feature_tables=["features_rich"],
                ),
            )
        )

        graph.add_node(
            ProjectionNode(
                name="proj_model_health",
                depends_on=(
                    "bars",
                    "folds_outer",
                    "diag_ret_loess",
                    "diag_ret_kernel",
                    "diag_ret_ridge",
                    "diag_dir_prob",
                ),
                labels={"role": "diagnostic_projection"},
                inputs=ProjectionInputs(
                    bars=["bars"],
                    fold_plans=["folds_outer"],
                    diagnostic_tables=[
                        "diag_ret_loess",
                        "diag_ret_kernel",
                        "diag_ret_ridge",
                        "diag_dir_prob",
                    ],
                    lag=1,
                    mode="prev_fold_to_test",
                ),
            )
        )

        # Derived targets for error and robustness modeling.
        graph.add_node(
            DerivedTargetNode(
                name="target_abs_err_kernel",
                depends_on=("ret_kernel_rich",),
                task_name="derived_abs_error_kernel",
                target_kind="abs_error",
                source_prediction_node="ret_kernel_rich",
                labels={"role": "derived_target", "target_kind": "abs_error"},
                inputs=DerivedTargetInputs(prediction_tables=["ret_kernel_rich"]),
            )
        )
        graph.add_node(
            DerivedTargetNode(
                name="target_abs_err_loess",
                depends_on=("ret_loess_base",),
                task_name="derived_abs_error_loess",
                target_kind="abs_error",
                source_prediction_node="ret_loess_base",
                labels={"role": "derived_target", "target_kind": "abs_error"},
                inputs=DerivedTargetInputs(prediction_tables=["ret_loess_base"]),
            )
        )
        graph.add_node(
            DerivedTargetNode(
                name="target_signed_resid_kernel",
                depends_on=("ret_kernel_rich",),
                task_name="derived_signed_residual_kernel",
                target_kind="signed_residual",
                source_prediction_node="ret_kernel_rich",
                labels={"role": "derived_target", "target_kind": "signed_residual"},
                inputs=DerivedTargetInputs(prediction_tables=["ret_kernel_rich"]),
            )
        )
        graph.add_node(
            DerivedTargetNode(
                name="target_disagreement_return",
                depends_on=("ret_loess_base", "ret_kernel_rich", "ret_ridge_stable"),
                task_name="derived_disagreement_return",
                target_kind="disagreement",
                labels={"role": "derived_target", "target_kind": "disagreement"},
                inputs=DerivedTargetInputs(
                    prediction_tables=[
                        "ret_loess_base",
                        "ret_kernel_rich",
                        "ret_ridge_stable",
                    ]
                ),
            )
        )
        graph.add_node(
            DerivedTargetNode(
                name="target_calibration_gap_dir",
                depends_on=("dir_logistic_prob", "diag_dir_prob"),
                task_name="derived_calibration_gap_dir",
                target_kind="calibration_gap",
                source_prediction_node="dir_logistic_prob",
                labels={"role": "derived_target", "target_kind": "calibration_gap"},
                inputs=DerivedTargetInputs(
                    prediction_tables=["dir_logistic_prob"],
                    diagnostic_tables=["diag_dir_prob"],
                ),
            )
        )

        # Error and disagreement forecasters.
        graph.add_node(
            ModelNode(
                name="err_kernel_abs",
                depends_on=("features_rich", "proj_model_health", "target_abs_err_kernel", "folds_outer"),
                model_spec=ModelSpec("err_kernel_abs", "kernel_ridge", {"alpha": 0.55, "gamma": 0.1}),
                training_recipe=TrainingRecipe(save_training_snapshot=True),
                labels={"role": "error_model"},
                inputs=ModelInputs(
                    feature_tables=["features_rich", "proj_model_health"],
                    target_tables=["target_abs_err_kernel"],
                    fold_plans=["folds_outer"],
                    prediction_tables=["ret_kernel_rich", "ret_loess_base", "ret_ridge_stable"],
                ),
            )
        )
        graph.add_node(
            ModelNode(
                name="err_loess_abs",
                depends_on=("features_rich", "proj_model_health", "target_abs_err_loess", "folds_outer"),
                model_spec=ModelSpec("err_loess_abs", "ridge", {"alpha": 0.9}),
                training_recipe=TrainingRecipe(save_training_snapshot=True),
                labels={"role": "error_model"},
                inputs=ModelInputs(
                    feature_tables=["features_rich", "proj_model_health"],
                    target_tables=["target_abs_err_loess"],
                    fold_plans=["folds_outer"],
                    prediction_tables=["ret_kernel_rich", "ret_loess_base", "ret_ridge_stable"],
                ),
            )
        )
        graph.add_node(
            ModelNode(
                name="signed_resid_kernel_model",
                depends_on=("features_rich", "proj_model_health", "target_signed_resid_kernel", "folds_outer"),
                model_spec=ModelSpec("signed_resid_kernel_model", "ridge", {"alpha": 1.5}),
                training_recipe=TrainingRecipe(save_training_snapshot=True),
                labels={"role": "error_model"},
                inputs=ModelInputs(
                    feature_tables=["features_rich", "proj_model_health"],
                    target_tables=["target_signed_resid_kernel"],
                    fold_plans=["folds_outer"],
                    prediction_tables=["ret_kernel_rich", "ret_loess_base", "ret_ridge_stable"],
                ),
            )
        )
        graph.add_node(
            ModelNode(
                name="disagreement_forecaster",
                depends_on=("features_rich", "proj_model_health", "target_disagreement_return", "folds_outer"),
                model_spec=ModelSpec("disagreement_forecaster", "ridge", {"alpha": 0.7}),
                training_recipe=TrainingRecipe(save_training_snapshot=True),
                labels={"role": "error_model"},
                inputs=ModelInputs(
                    feature_tables=["features_rich", "proj_model_health"],
                    target_tables=["target_disagreement_return"],
                    fold_plans=["folds_outer"],
                    prediction_tables=["ret_kernel_rich", "ret_loess_base", "ret_ridge_stable"],
                ),
            )
        )
        graph.add_node(
            ModelNode(
                name="calib_gap_forecaster",
                depends_on=("features_rich", "proj_model_health", "target_calibration_gap_dir", "folds_outer"),
                model_spec=ModelSpec("calib_gap_forecaster", "ridge", {"alpha": 0.6}),
                training_recipe=TrainingRecipe(save_training_snapshot=True),
                labels={"role": "error_model"},
                inputs=ModelInputs(
                    feature_tables=["features_rich", "proj_model_health"],
                    target_tables=["target_calibration_gap_dir"],
                    fold_plans=["folds_outer"],
                    prediction_tables=["dir_logistic_prob"],
                ),
            )
        )

        # Final champion meta model.
        graph.add_node(
            ModelNode(
                name="champion_meta_return",
                depends_on=(
                    "features_rich",
                    "features_regime",
                    "proj_model_health",
                    "ret_loess_base",
                    "ret_kernel_rich",
                    "ret_ridge_stable",
                    "ret_persistence_baseline",
                    "vol_kernel_risk",
                    "dir_logistic_prob",
                    "err_kernel_abs",
                    "err_loess_abs",
                    "signed_resid_kernel_model",
                    "disagreement_forecaster",
                    "calib_gap_forecaster",
                    "diag_ret_loess",
                    "diag_ret_kernel",
                    "diag_ret_ridge",
                    "diag_dir_prob",
                    "target_return",
                    "folds_outer",
                ),
                model_spec=self.meta_model,
                training_recipe=TrainingRecipe(save_training_snapshot=True),
                labels={"task_family": "return", "role": "champion_meta"},
                inputs=ModelInputs(
                    feature_tables=["features_rich", "features_regime", "proj_model_health"],
                    prediction_tables=[
                        "ret_loess_base",
                        "ret_kernel_rich",
                        "ret_ridge_stable",
                        "ret_persistence_baseline",
                        "vol_kernel_risk",
                        "dir_logistic_prob",
                        "err_kernel_abs",
                        "err_loess_abs",
                        "signed_resid_kernel_model",
                        "disagreement_forecaster",
                        "calib_gap_forecaster",
                    ],
                    diagnostic_tables=[
                        "diag_ret_loess",
                        "diag_ret_kernel",
                        "diag_ret_ridge",
                        "diag_dir_prob",
                    ],
                    target_tables=["target_return"],
                    fold_plans=["folds_outer"],
                ),
            )
        )
        graph.add_node(
            DiagnosticNode(
                name="diag_champion_meta",
                depends_on=("champion_meta_return",),
                diagnostic_types=["train_test_deviance_gap", "feature_drift", "residual_std_by_asset"],
                labels={"task_family": "return", "role": "champion_diagnostic"},
                inputs=DiagnosticInputs(
                    prediction_tables=["champion_meta_return"],
                    feature_tables=["features_rich"],
                ),
            )
        )
        graph.add_node(
            SelectionNode(
                name="champion_selector",
                depends_on=(
                    "ret_persistence_baseline",
                    "ret_loess_base",
                    "ret_kernel_rich",
                    "ret_ridge_stable",
                    "champion_meta_return",
                ),
                strategy_name="best_recent_model",
                labels={"task_family": "return", "role": "selector"},
                inputs=SelectionInputs(
                    prediction_tables=[
                        "ret_persistence_baseline",
                        "ret_loess_base",
                        "ret_kernel_rich",
                        "ret_ridge_stable",
                        "champion_meta_return",
                    ]
                ),
            )
        )
        return graph

