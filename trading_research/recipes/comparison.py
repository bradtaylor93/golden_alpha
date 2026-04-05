"""Feature comparison recipe."""

from __future__ import annotations

from dataclasses import dataclass

from trading_research.recipes.base import Recipe
from trading_research.workflow.graph import WorkflowGraph
from trading_research.workflow.nodes import FeatureNode, ModelInputs, ModelNode


@dataclass(frozen=True)
class FeatureComparisonRecipe(Recipe):
    feature_families: tuple[str, ...] = ("baseline", "field_adapter", "cross_asset")

    def compile(self) -> WorkflowGraph:
        graph = self._base_graph()
        for family in self.feature_families:
            graph.add_node(
                FeatureNode(
                    name=f"features_{family}",
                    depends_on=("bars",),
                    family_name=family,
                    params={},
                )
            )
            graph.add_node(
                ModelNode(
                    name=f"model_{family}",
                    depends_on=(f"features_{family}", "targets", "folds_outer"),
                    model_spec=self.model_spec,
                    training_recipe=self.training_recipe,
                    inputs=ModelInputs(
                        feature_tables=[f"features_{family}"],
                        target_tables=["targets"],
                        fold_plans=["folds_outer"],
                    ),
                )
            )
        return graph
