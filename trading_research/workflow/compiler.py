"""Recipe -> graph compilation utilities."""

from __future__ import annotations

from typing import Protocol

from trading_research.workflow.graph import WorkflowGraph


class CompilableRecipe(Protocol):
    def compile(self) -> WorkflowGraph:
        """Compile a user recipe into a typed workflow graph."""


class RecipeCompiler:
    def compile(self, recipe: CompilableRecipe) -> WorkflowGraph:
        return recipe.compile()

