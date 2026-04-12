"""Run inspection APIs for auditability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.workflow.artifacts import ArtifactRef, ArtifactStore
from trading_research.workflow.graph import WorkflowGraph
from trading_research.workflow.manifest import RunManifest


@dataclass
class RunInspector:
    """Provides helper methods to inspect run outputs."""

    run_id: str
    run_dir: Path
    graph: WorkflowGraph
    manifest: RunManifest
    artifact_store: ArtifactStore

    @classmethod
    def load_run(cls, run_id: str, runs_root: Path | str = "runs") -> "RunInspector":
        root = Path(runs_root)
        run_dir = root / run_id
        if not run_dir.exists():
            alt = root / "runs" / run_id
            if alt.exists():
                run_dir = alt
        graph = WorkflowGraph.from_json(run_dir / "config" / "workflow_graph.json")
        manifest = RunManifest.from_json(run_dir / "manifest.json")
        artifact_store = ArtifactStore(run_dir)
        return cls(run_id=run_id, run_dir=run_dir, graph=graph, manifest=manifest, artifact_store=artifact_store)

    def list_nodes(self) -> list[str]:
        return [node.name for node in self.graph.nodes]

    def list_artifacts(self) -> list[ArtifactRef]:
        return self.artifact_store.index.list_artifacts()

    def load_artifact(
        self,
        node_name: str,
        artifact_type: str | None = None,
        outer_fold_id: int | None = None,
        split_role: str | None = None,
    ) -> Any:
        refs = self.artifact_store.index.list_artifacts(node_name=node_name, artifact_type=artifact_type)
        if outer_fold_id is not None:
            refs = [r for r in refs if r.scope.outer_fold_id == outer_fold_id]
        if split_role is not None:
            refs = [r for r in refs if r.scope.split_role == split_role]
        if not refs:
            raise ValueError(f"No artifact found for node={node_name}, type={artifact_type}, fold={outer_fold_id}")
        refs = sorted(refs, key=lambda x: x.path)
        return self.artifact_store.load(refs[-1].artifact_id)

    def show_lineage(self, node_name: str, outer_fold_id: int | None = None) -> list[dict[str, Any]]:
        refs = self.artifact_store.index.list_artifacts(node_name=node_name)
        if outer_fold_id is not None:
            refs = [r for r in refs if r.scope.outer_fold_id == outer_fold_id]
        return [
            {
                "artifact_id": ref.artifact_id,
                "artifact_type": ref.artifact_type,
                "scope": ref.scope.to_dict(),
                "upstream_ids": ref.upstream_ids,
            }
            for ref in refs
        ]

    def load_predictions(self, node_name: str, outer_fold_id: int | None = None) -> pd.DataFrame:
        return self.load_artifact(
            node_name=node_name,
            artifact_type="PredictionArtifact",
            outer_fold_id=outer_fold_id,
            split_role="oof",
        )

    def load_inner_validation(self, node_name: str, outer_fold_id: int | None = None) -> pd.DataFrame:
        return self.load_artifact(
            node_name=node_name,
            artifact_type="SelectionArtifact",
            outer_fold_id=outer_fold_id,
            split_role="validation",
        )

    def load_training_snapshot(self, node_name: str, outer_fold_id: int | None = None) -> pd.DataFrame:
        return self.load_artifact(
            node_name=node_name,
            artifact_type="TrainingDatasetSnapshotArtifact",
            outer_fold_id=outer_fold_id,
            split_role="train",
        )


def load_run(run_id: str, runs_root: Path | str = "runs") -> RunInspector:
    """Module-level convenience helper."""
    return RunInspector.load_run(run_id=run_id, runs_root=runs_root)
