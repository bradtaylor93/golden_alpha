"""Stacked model recipe."""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_research.models.registry import ModelSpec, TrainingRecipe
from trading_research.recipes.base import Recipe
from trading_research.workflow.graph import WorkflowGraph
from trading_research.workflow.nodes import FeatureNode, ModelInputs, ModelNode


@dataclass(frozen=True)
class StackedModelRecipe(Recipe):
    base_model_spec: ModelSpec = field(
        default_factory=lambda: ModelSpec(name="ret20_base", algorithm="ridge", params={"alpha": 1.0})
    )
    meta_model_spec: ModelSpec = field(
        default_factory=lambda: ModelSpec(name="ret20_meta", algorithm="ridge", params={"alpha": 0.6})
    )
    training_recipe: TrainingRecipe = field(default_factory=TrainingRecipe)

    def compile(self) -> WorkflowGraph:
        graph = self._base_graph()
        graph.add_node(FeatureNode(name="features", depends_on=("bars",), family_name="baseline"))
        graph.add_node(
            ModelNode(
                name="base_model",
                depends_on=("features", "targets", "folds_outer"),
                model_spec=self.base_model_spec,
                training_recipe=self.training_recipe,
                inputs=ModelInputs(
                    feature_tables=["features"],
                    target_tables=["targets"],
                    fold_plans=["folds_outer"],
                ),
            )
        )
        graph.add_node(
            ModelNode(
                name="meta_model",
                depends_on=("features", "base_model", "targets", "folds_outer"),
                model_spec=self.meta_model_spec,
                training_recipe=self.training_recipe,
                inputs=ModelInputs(
                    feature_tables=["features"],
                    prediction_tables=["base_model"],
                    target_tables=["targets"],
                    fold_plans=["folds_outer"],
                    use_base_features=True,
                ),
            )
        )
        return graph
