"""Nested tuning recipe."""

from __future__ import annotations

from dataclasses import dataclass

from trading_research.models.registry import HyperoptSpec, ModelSpec, TrainingRecipe
from trading_research.recipes.base import Recipe
from trading_research.workflow.graph import WorkflowGraph
from trading_research.workflow.nodes import FeatureNode, FoldPlanNode, ModelInputs, ModelNode


@dataclass(frozen=True)
class NestedTuningRecipe(Recipe):
    run_name: str = "nested_tuning"
    model_spec: ModelSpec = ModelSpec(name="ridge", algorithm="ridge", params={"alpha": 1.0})

    def compile(self) -> WorkflowGraph:
        graph = self._base_graph()
        training = TrainingRecipe(
            selection_validation=self.validation_spec,
            hyperopt=HyperoptSpec(
                method="grid",
                objective="mse",
                param_grid={"alpha": [0.1, 1.0, 10.0]},
            ),
        )
        graph.add_node(FeatureNode(name="baseline_features", depends_on=("bars",), family_name="baseline"))
        graph.add_node(
            FoldPlanNode(
                name="folds_inner",
                depends_on=("bars",),
                validation=self.validation_spec,
                level="inner",
                build_inner=True,
                inner_folds=3,
            )
        )
        graph.add_node(
            ModelNode(
                name="ridge_nested",
                depends_on=("baseline_features", "targets", "folds_outer", "folds_inner"),
                model_spec=self.model_spec,
                training_recipe=training,
                inputs=ModelInputs(
                    feature_tables=["baseline_features"],
                    target_tables=["targets"],
                    fold_plans=["folds_outer"],
                ),
            )
        )
        return graph
