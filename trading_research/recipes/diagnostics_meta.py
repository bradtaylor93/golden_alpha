"""Diagnostic-aware meta recipe."""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_research.models.registry import ModelSpec
from trading_research.recipes.base import Recipe
from trading_research.workflow.graph import WorkflowGraph
from trading_research.workflow.nodes import (
    DataNode,
    FeatureNode,
    FoldPlanNode,
    TargetNode,
    DiagnosticNode,
    DiagnosticInputs,
    ModelInputs,
    ModelNode,
    ProjectionNode,
    ProjectionInputs,
)


@dataclass(frozen=True)
class DiagnosticAwareMetaRecipe(Recipe):
    """Advanced workflow: cross-target base models + projected diagnostics + downstream model."""

    return_task_name: str = "forward_return_20"
    volatility_task_name: str = "forward_realized_volatility_20"
    final_task_name: str = "max_upside_40"
    return_model_spec: ModelSpec = field(
        default_factory=lambda: ModelSpec("ret20_model", "ridge", {"alpha": 0.7})
    )
    vol_model_spec: ModelSpec = field(
        default_factory=lambda: ModelSpec("vol20_model", "ridge", {"alpha": 1.0})
    )
    final_model_spec: ModelSpec = field(
        default_factory=lambda: ModelSpec("meta_upside40", "ridge", {"alpha": 0.8})
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
            )
        )
        graph.add_node(
            TargetNode(
                name="target_ret20",
                depends_on=("bars",),
                task_name=self.return_task_name,
            )
        )
        graph.add_node(
            TargetNode(
                name="target_vol20",
                depends_on=("bars",),
                task_name=self.volatility_task_name,
            )
        )
        graph.add_node(
            TargetNode(
                name="target_final",
                depends_on=("bars",),
                task_name=self.final_task_name,
            )
        )
        graph.add_node(FeatureNode(name="features_baseline", depends_on=("bars",), family_name="baseline"))
        graph.add_node(FeatureNode(name="features_regime", depends_on=("bars",), family_name="regime"))
        graph.add_node(
            ModelNode(
                name="ret20_model",
                depends_on=("features_baseline", "target_ret20", "folds_outer"),
                model_spec=self.return_model_spec,
                inputs=ModelInputs(
                    feature_tables=["features_baseline"],
                    target_tables=["target_ret20"],
                    fold_plans=["folds_outer"],
                ),
            )
        )
        graph.add_node(
            ModelNode(
                name="vol20_model",
                depends_on=("features_regime", "target_vol20", "folds_outer"),
                model_spec=self.vol_model_spec,
                inputs=ModelInputs(
                    feature_tables=["features_regime"],
                    target_tables=["target_vol20"],
                    fold_plans=["folds_outer"],
                ),
            )
        )
        graph.add_node(
            DiagnosticNode(
                name="diag_ret20",
                depends_on=("ret20_model",),
                diagnostic_types=(
                    "train_test_deviance_gap",
                    "residual_std_by_asset",
                    "calibration_gap",
                ),
                inputs=DiagnosticInputs(
                    prediction_tables=["ret20_model"],
                ),
            )
        )
        graph.add_node(
            DiagnosticNode(
                name="diag_vol20",
                depends_on=("vol20_model",),
                diagnostic_types=("train_test_deviance_gap", "feature_drift"),
                inputs=DiagnosticInputs(
                    prediction_tables=["vol20_model"],
                ),
            )
        )
        graph.add_node(
            ProjectionNode(
                name="projected_health",
                depends_on=("diag_ret20", "diag_vol20", "bars"),
                inputs=ProjectionInputs(
                    diagnostic_tables=["diag_ret20", "diag_vol20"],
                    bars=["bars"],
                    lag=1,
                ),
            )
        )
        graph.add_node(
            ModelNode(
                name="meta_upside40",
                depends_on=(
                    "features_baseline",
                    "projected_health",
                    "ret20_model",
                    "vol20_model",
                    "target_final",
                    "folds_outer",
                ),
                model_spec=self.final_model_spec,
                inputs=ModelInputs(
                    feature_tables=["features_baseline", "projected_health"],
                    prediction_tables=["ret20_model", "vol20_model"],
                    target_tables=["target_final"],
                    fold_plans=["folds_outer"],
                ),
            )
        )
        return graph
