"""Typed workflow node declarations and input specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trading_research.models.registry import ModelSpec, TrainingRecipe
from trading_research.validation.splits import ValidationSpec


@dataclass(frozen=True)
class ModelInputs:
    bars: list[str] = field(default_factory=list)
    fold_plans: list[str] = field(default_factory=list)
    feature_tables: list[str] = field(default_factory=list)
    target_tables: list[str] = field(default_factory=list)
    prediction_tables: list[str] = field(default_factory=list)
    diagnostic_tables: list[str] = field(default_factory=list)
    selection_tables: list[str] = field(default_factory=list)
    use_base_features: bool = True


@dataclass(frozen=True)
class DiagnosticInputs:
    prediction_tables: list[str] = field(default_factory=list)
    target_tables: list[str] = field(default_factory=list)
    feature_tables: list[str] = field(default_factory=list)
    model_state_tables: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DerivedFeatureInputs:
    diagnostic_tables: list[str] = field(default_factory=list)
    prediction_tables: list[str] = field(default_factory=list)
    feature_tables: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProjectionInputs:
    bars: list[str] = field(default_factory=list)
    diagnostic_tables: list[str] = field(default_factory=list)
    lag: int = 1


@dataclass(frozen=True)
class SelectionInputs:
    prediction_tables: list[str] = field(default_factory=list)
    diagnostic_tables: list[str] = field(default_factory=list)
    metric_table: str | None = None


@dataclass(frozen=True)
class WorkflowNode:
    name: str
    depends_on: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DataNode(WorkflowNode):
    universe: tuple[str, ...] = field(default_factory=tuple)
    universe_name: str = "default_universe"
    dataset_name: str = "bars"


@dataclass(frozen=True)
class TargetNode(WorkflowNode):
    bars_inputs: list[str] = field(default_factory=lambda: ["bars"])
    prediction_task_name: str | None = None
    task_name: str = "forward_return_20"
    horizon: int | None = None
    embargo: int = 0


@dataclass(frozen=True)
class FoldPlanNode(WorkflowNode):
    bars_inputs: list[str] = field(default_factory=lambda: ["bars"])
    validation: ValidationSpec = field(
        default_factory=lambda: ValidationSpec(
            split_type="expanding",
            min_train_size=80,
            test_size=20,
            embargo=0,
            rolling_train_size=None,
            mode="pooled",
        )
    )
    level: str = "outer"
    validation_name: str = "outer"
    n_folds: int = 4
    build_inner: bool = False
    inner_folds: int = 3


@dataclass(frozen=True)
class FeatureNode(WorkflowNode):
    bars_inputs: list[str] = field(default_factory=lambda: ["bars"])
    family_name: str = "baseline"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformNode(WorkflowNode):
    transform_name: str = "standard_scaler"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelNode(WorkflowNode):
    model_spec: ModelSpec = field(default_factory=lambda: ModelSpec("ridge", "ridge", {}))
    training_recipe: TrainingRecipe = field(default_factory=TrainingRecipe)
    inputs: ModelInputs = field(default_factory=ModelInputs)


@dataclass(frozen=True)
class DiagnosticNode(WorkflowNode):
    diagnostic_types: list[str] = field(default_factory=lambda: ["train_test_deviance_gap"])
    diagnostic_name: str = "train_test_deviance_gap"
    inputs: DiagnosticInputs = field(default_factory=DiagnosticInputs)


@dataclass(frozen=True)
class DerivedFeatureNode(WorkflowNode):
    derived_expressions: list[str] = field(default_factory=list)
    feature_name: str = "residual_std_by_asset"
    inputs: DerivedFeatureInputs = field(default_factory=DerivedFeatureInputs)


@dataclass(frozen=True)
class ProjectionNode(WorkflowNode):
    projection_name: str = "lagged_diagnostics_projection"
    inputs: ProjectionInputs = field(default_factory=ProjectionInputs)


@dataclass(frozen=True)
class SelectionNode(WorkflowNode):
    strategy: str = "best_recent_model"
    strategy_name: str = "best_recent_model"
    inputs: SelectionInputs = field(default_factory=SelectionInputs)


# Aliases used by the runner interface.
DiagnosticNodeInputs = DiagnosticInputs
ProjectionNodeInputs = ProjectionInputs
SelectionNodeInputs = SelectionInputs
