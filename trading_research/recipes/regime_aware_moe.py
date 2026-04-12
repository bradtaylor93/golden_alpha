"""Regime-aware mixture-of-experts recipe with learned gating signals."""

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
class RegimeAwareMoERecipe(Recipe):
    """Asset-specific local experts with learned error-based gating."""

    run_name: str = "regime_aware_moe"
    return_horizon: int = 20
    validation_spec: ValidationSpec = ValidationSpec(
        split_type="expanding",
        min_train_size=24 * 90,
        test_size=24 * 15,
        embargo=4,
        rolling_train_size=None,
        mode="pooled",
    )
    expert_trend: ModelSpec = field(
        default_factory=lambda: ModelSpec("expert_trend_ridge", "ridge", {"alpha": 1.2})
    )
    expert_meanrev: ModelSpec = field(
        default_factory=lambda: ModelSpec("expert_meanrev_loess", "loess", {"frac": 0.22, "ridge": 2e-4})
    )
    expert_nonlinear: ModelSpec = field(
        default_factory=lambda: ModelSpec("expert_nonlinear_kernel", "kernel_ridge", {"alpha": 0.45, "gamma": 0.14})
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

        graph.add_node(
            TargetNode(
                name="target_return",
                depends_on=("bars",),
                task_name=f"forward_return_{self.return_horizon}",
                labels={"task_family": "return", "horizon": self.return_horizon},
            )
        )
        graph.add_node(
            TargetNode(
                name="target_volatility",
                depends_on=("bars",),
                task_name=f"forward_realized_volatility_{self.return_horizon}",
                labels={"task_family": "volatility", "horizon": self.return_horizon},
            )
        )

        graph.add_node(
            FeatureNode(
                name="features_base",
                depends_on=("bars",),
                family_name="baseline",
                params={"lookbacks": [1, 2, 4, 10, 20, 40, 80, 120]},
                labels={"role": "features", "pack": "base"},
            )
        )
        graph.add_node(
            FeatureNode(
                name="features_rich",
                depends_on=("bars",),
                family_name="research_pack",
                params={
                    "short_window": 8,
                    "medium_window": 32,
                    "long_window": 160,
                    "lookbacks": [2, 4, 8, 16, 32, 64, 96, 128],
                },
                labels={"role": "features", "pack": "rich_long"},
            )
        )
        graph.add_node(
            FeatureNode(
                name="features_regime",
                depends_on=("bars",),
                family_name="regime",
                params={"window": 48},
                labels={"role": "features", "pack": "regime"},
            )
        )
        graph.add_node(
            FeatureNode(
                name="features_cross",
                depends_on=("bars",),
                family_name="cross_asset",
                labels={"role": "features", "pack": "cross"},
            )
        )

        graph.add_node(
            ModelNode(
                name="vol_context",
                depends_on=("features_rich", "features_regime", "target_volatility", "folds_outer"),
                model_spec=ModelSpec("vol_context", "kernel_ridge", {"alpha": 0.5, "gamma": 0.16}),
                labels={"role": "risk_context"},
                inputs=ModelInputs(
                    feature_tables=["features_rich", "features_regime"],
                    target_tables=["target_volatility"],
                    fold_plans=["folds_outer"],
                    feature_include_regex=["vol_", "down_vol_", "up_vol_", "range_", "trend_", "ema_"],
                    max_features=42,
                    clip_quantiles=(0.01, 0.99),
                    standardize_features=True,
                ),
            )
        )

        graph.add_node(
            ModelNode(
                name="expert_trend_ridge",
                depends_on=("features_rich", "features_regime", "features_cross", "target_return", "folds_outer"),
                model_spec=self.expert_trend,
                labels={"role": "local_expert", "expert": "trend"},
                inputs=ModelInputs(
                    feature_tables=["features_rich", "features_regime", "features_cross"],
                    target_tables=["target_return"],
                    fold_plans=["folds_outer"],
                    feature_include_regex=[
                        "ema_",
                        "trend_",
                        "mom_",
                        "vol_40",
                        "vol_64",
                        "vol_96",
                        "close_range_position",
                        "pct_to_ema_",
                        "resistance_hit_",
                        "support_hit_",
                        "hit_asym_",
                        "sr_distance_asym_",
                        "bb_",
                    ],
                    max_features=30,
                    clip_quantiles=(0.01, 0.99),
                    standardize_features=True,
                ),
            )
        )
        graph.add_node(
            ModelNode(
                name="expert_meanrev_loess",
                depends_on=("features_base", "features_cross", "target_return", "folds_outer"),
                model_spec=self.expert_meanrev,
                labels={"role": "local_expert", "expert": "mean_reversion"},
                inputs=ModelInputs(
                    feature_tables=["features_base", "features_cross"],
                    target_tables=["target_return"],
                    fold_plans=["folds_outer"],
                    feature_include_regex=[
                        "ret_1",
                        "ret_2",
                        "ret_4",
                        "ret_z_",
                        "close_pos_",
                        "range_mean_",
                        "resistance_hit_",
                        "support_hit_",
                        "hit_asym_",
                        "bb_",
                    ],
                    max_features=24,
                    clip_quantiles=(0.02, 0.98),
                    standardize_features=True,
                ),
            )
        )
        graph.add_node(
            ModelNode(
                name="expert_nonlinear_kernel",
                depends_on=("features_rich", "features_regime", "features_cross", "target_return", "folds_outer"),
                model_spec=self.expert_nonlinear,
                labels={"role": "local_expert", "expert": "nonlinear"},
                inputs=ModelInputs(
                    feature_tables=["features_rich", "features_regime", "features_cross"],
                    target_tables=["target_return"],
                    fold_plans=["folds_outer"],
                    feature_include_regex=[
                        "ret_",
                        "vol_",
                        "mom_",
                        "down_vol_",
                        "up_vol_",
                        "ema_",
                        "range_",
                        "market_ret_1",
                        "asset_minus_market_ret_1",
                        "pct_to_ema_",
                        "resistance_hit_",
                        "support_hit_",
                        "hit_asym_",
                        "sr_distance_asym_",
                        "bb_",
                    ],
                    max_features=48,
                    clip_quantiles=(0.01, 0.99),
                    standardize_features=True,
                ),
            )
        )

        for expert_node, family in [
            ("expert_trend_ridge", "trend"),
            ("expert_meanrev_loess", "mean_reversion"),
            ("expert_nonlinear_kernel", "nonlinear"),
        ]:
            graph.add_node(
                DiagnosticNode(
                    name=f"diag_{family}",
                    depends_on=(expert_node,),
                    diagnostic_types=["train_test_deviance_gap", "feature_drift", "residual_std_by_asset"],
                    labels={"role": "expert_diag", "expert": family},
                    inputs=DiagnosticInputs(
                        prediction_tables=[expert_node],
                        feature_tables=["features_rich"],
                    ),
                )
            )
            graph.add_node(
                DerivedTargetNode(
                    name=f"target_abs_err_{family}",
                    depends_on=(expert_node,),
                    task_name=f"derived_abs_error_{family}",
                    target_kind="abs_error",
                    source_prediction_node=expert_node,
                    labels={"role": "gating_target", "expert": family},
                    inputs=DerivedTargetInputs(prediction_tables=[expert_node]),
                )
            )

        graph.add_node(
            ProjectionNode(
                name="proj_expert_health",
                depends_on=("bars", "folds_outer", "diag_trend", "diag_mean_reversion", "diag_nonlinear"),
                labels={"role": "expert_diag_projection"},
                inputs=ProjectionInputs(
                    bars=["bars"],
                    fold_plans=["folds_outer"],
                    diagnostic_tables=["diag_trend", "diag_mean_reversion", "diag_nonlinear"],
                    lag=1,
                    mode="prev_fold_to_test",
                ),
            )
        )

        for family, expert in [
            ("trend", "expert_trend_ridge"),
            ("mean_reversion", "expert_meanrev_loess"),
            ("nonlinear", "expert_nonlinear_kernel"),
        ]:
            graph.add_node(
                ModelNode(
                    name=f"gate_abs_err_{family}",
                    depends_on=(
                        "features_rich",
                        "features_regime",
                        "proj_expert_health",
                        "vol_context",
                        f"target_abs_err_{family}",
                        "folds_outer",
                        expert,
                    ),
                    model_spec=ModelSpec(f"gate_abs_err_{family}", "ridge", {"alpha": 1.4}),
                    labels={"role": "gating_model", "expert": family},
                    inputs=ModelInputs(
                        feature_tables=["features_rich", "features_regime", "proj_expert_health"],
                        prediction_tables=["vol_context", expert],
                        target_tables=[f"target_abs_err_{family}"],
                        fold_plans=["folds_outer"],
                        feature_include_regex=[
                            "regime_",
                            "proj_",
                            "vol_",
                            "down_vol_",
                            "up_vol_",
                            "range_",
                            "trend_",
                            "ema_",
                            "pred_vol_context",
                            "resistance_hit_",
                            "support_hit_",
                            "hit_asym_",
                            "sr_distance_asym_",
                            "bb_",
                        ],
                        max_features=40,
                        clip_quantiles=(0.01, 0.99),
                        standardize_features=True,
                    ),
                )
            )

        graph.add_node(
            SelectionNode(
                name="moe_selector_reference",
                depends_on=("expert_trend_ridge", "expert_meanrev_loess", "expert_nonlinear_kernel"),
                strategy_name="best_recent_model",
                labels={"role": "reference_selector"},
                inputs=SelectionInputs(
                    prediction_tables=["expert_trend_ridge", "expert_meanrev_loess", "expert_nonlinear_kernel"]
                ),
            )
        )
        return graph

