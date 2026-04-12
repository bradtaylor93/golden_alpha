"""Single model run recipe."""

from __future__ import annotations

from dataclasses import dataclass

from trading_research.recipes.base import Recipe
from trading_research.workflow.graph import WorkflowGraph
from trading_research.workflow.nodes import FeatureNode, ModelInputs, ModelNode


@dataclass(frozen=True)
class SingleModelRecipe(Recipe):
    """Compile a baseline single-model workflow."""

    def compile(self) -> WorkflowGraph:
        graph = self._base_graph()
        graph.add_node(FeatureNode(name="features", depends_on=("bars",), family_name="baseline"))
        graph.add_node(
            ModelNode(
                name="model",
                depends_on=("features", "targets", "folds_outer"),
                model_spec=self.model_spec,
                training_recipe=self.training_recipe,
                inputs=ModelInputs(
                    feature_tables=["features"],
                    target_tables=["targets"],
                    fold_plans=["folds_outer"],
                ),
            )
        )
        return graph
