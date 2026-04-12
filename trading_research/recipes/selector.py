"""Selector and ensemble recipe."""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_research.models.registry import ModelSpec
from trading_research.recipes.base import Recipe
from trading_research.workflow.graph import WorkflowGraph
from trading_research.workflow.nodes import (
    FeatureNode,
    ModelInputs,
    ModelNode,
    SelectionInputs,
    SelectionNode,
)


@dataclass(frozen=True)
class SelectorRecipe(Recipe):
    """Train multiple models and select best recent."""

    candidates: tuple[ModelSpec, ...] = field(
        default_factory=lambda: (
            ModelSpec(name="model_a", algorithm="ridge", params={"alpha": 0.5}),
            ModelSpec(name="model_b", algorithm="ridge", params={"alpha": 1.0}),
            ModelSpec(name="model_c", algorithm="ridge", params={"alpha": 2.0}),
        )
    )
    strategy: str = "best_recent_model"

    def compile(self) -> WorkflowGraph:
        graph = self._base_graph()
        graph.add_node(FeatureNode(name="features_baseline", depends_on=("bars",), family_name="baseline"))
        pred_nodes: list[str] = []
        for spec in self.candidates:
            pred_nodes.append(spec.name)
            graph.add_node(
                ModelNode(
                    name=spec.name,
                    depends_on=("features_baseline", "targets", "folds_outer"),
                    model_spec=spec,
                    inputs=ModelInputs(
                        feature_tables=["features_baseline"],
                        target_tables=["targets"],
                        fold_plans=["folds_outer"],
                    ),
                )
            )

        graph.add_node(
            SelectionNode(
                name="selector",
                depends_on=tuple(pred_nodes),
                strategy_name=self.strategy,
                inputs=SelectionInputs(prediction_tables=pred_nodes),
            )
        )
        return graph
