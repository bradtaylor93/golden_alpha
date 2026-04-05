"""User-facing recipe abstraction compiled to typed workflow graphs."""

from __future__ import annotations

from dataclasses import dataclass

from trading_research.models.registry import ModelSpec, TrainingRecipe
from trading_research.validation.splits import ValidationSpec
from trading_research.workflow.graph import WorkflowGraph
from trading_research.workflow.nodes import DataNode, FoldPlanNode, TargetNode


@dataclass(frozen=True)
class Recipe:
    """Base recipe with shared defaults and base graph helper."""

    dataset_name: str = "default_universe"
    universe: tuple[str, ...] = ("SPY",)
    task_name: str = "forward_return_20"
    model_spec: ModelSpec = ModelSpec(name="ridge_default", algorithm="ridge", params={"alpha": 1.0})
    training_recipe: TrainingRecipe = TrainingRecipe()
    validation_spec: ValidationSpec = ValidationSpec(
        split_type="expanding",
        min_train_size=80,
        test_size=20,
        embargo=0,
        rolling_train_size=None,
        mode="pooled",
    )
    run_name: str = "recipe_run"

    def _base_graph(self) -> WorkflowGraph:
        graph = WorkflowGraph(name=self.run_name)
        graph.add_node(
            DataNode(
                name="bars",
                universe=self.universe,
                dataset_name=self.dataset_name,
            )
        )
        graph.add_node(
            TargetNode(
                name="targets",
                depends_on=("bars",),
                task_name=self.task_name,
            )
        )
        graph.add_node(
            FoldPlanNode(
                name="folds_outer",
                depends_on=("bars",),
                validation=self.validation_spec,
                level="outer",
            )
        )
        return graph

    def compile(self) -> WorkflowGraph:
        raise NotImplementedError
