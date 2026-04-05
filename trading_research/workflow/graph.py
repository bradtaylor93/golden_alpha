"""Typed execution graph representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any

from trading_research.workflow.nodes import WorkflowNode


@dataclass
class WorkflowGraph:
    name: str = "workflow"
    nodes: list[WorkflowNode] = field(default_factory=list)

    def add_node(self, node: WorkflowNode) -> None:
        if any(existing.name == node.name for existing in self.nodes):
            raise ValueError(f"Duplicate node name: {node.name}")
        self.nodes.append(node)

    def topological(self) -> list[WorkflowNode]:
        by_name = {n.name: n for n in self.nodes}
        visited: set[str] = set()
        ordering: list[WorkflowNode] = []

        def dfs(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            node = by_name[name]
            for dep in node.depends_on:
                if dep not in by_name:
                    raise KeyError(f"Node {name} depends on missing node {dep}")
                dfs(dep)
            ordering.append(node)

        for node in self.nodes:
            dfs(node.name)
        return ordering

    def execution_order(self) -> list[WorkflowNode]:
        return self.topological()

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "name": self.name,
            "nodes": [asdict(n) | {"node_class": n.__class__.__name__} for n in self.nodes],
        }
        return json.dumps(payload, indent=2, sort_keys=True, default=str)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.to_json())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "WorkflowGraph":
        """Best-effort loader for inspection use cases."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph = cls(name=payload.get("name", "workflow"), nodes=[])
        # We intentionally do not rebuild concrete node classes here.
        # Inspection methods only need node names.
        for node in payload.get("nodes", []):
            graph.nodes.append(WorkflowNode(name=node.get("name", "unknown"), depends_on=tuple(node.get("depends_on", []))))
        return graph
