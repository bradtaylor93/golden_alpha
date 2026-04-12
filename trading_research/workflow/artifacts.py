"""Artifact abstractions, persistence, and lineage indexing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import uuid

import pandas as pd

from trading_research.utils.io import read_table, write_table
from trading_research.utils.typing import ArtifactType, CvLevel, JsonDict, SplitRole


@dataclass(frozen=True)
class ArtifactScope:
    cv_level: CvLevel
    split_role: SplitRole
    outer_fold_id: int | None = None
    inner_fold_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    artifact_type: ArtifactType
    node_name: str
    path: str
    scope: ArtifactScope
    schema: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)
    upstream_ids: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def create(
        artifact_type: ArtifactType,
        node_name: str,
        path: str,
        scope: ArtifactScope,
        schema: JsonDict | None = None,
        metadata: JsonDict | None = None,
        upstream_ids: tuple[str, ...] | None = None,
    ) -> "ArtifactRef":
        return ArtifactRef(
            artifact_id=str(uuid.uuid4()),
            artifact_type=artifact_type,
            node_name=node_name,
            path=path,
            scope=scope,
            schema=schema or {},
            metadata=metadata or {},
            upstream_ids=upstream_ids or tuple(),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["artifact_type"] = str(self.artifact_type.value if hasattr(self.artifact_type, "value") else self.artifact_type)
        return payload


class ArtifactIndex:
    """Persisted artifact registry used for inspection and lineage queries."""

    def __init__(self, items: list[ArtifactRef] | None = None) -> None:
        self.items: list[ArtifactRef] = items or []

    def add(self, ref: ArtifactRef) -> None:
        self.items.append(ref)

    def refs(self) -> list[ArtifactRef]:
        return list(self.items)

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for ref in self.items:
            payload = asdict(ref)
            payload["artifact_type"] = ref.artifact_type.value
            payload["scope"] = json.dumps(payload["scope"], sort_keys=True)
            payload["schema"] = json.dumps(payload["schema"], sort_keys=True)
            payload["metadata"] = json.dumps(payload["metadata"], sort_keys=True)
            payload["upstream_ids"] = json.dumps(list(ref.upstream_ids))
            rows.append(payload)
        return pd.DataFrame(rows)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "ArtifactIndex":
        items: list[ArtifactRef] = []
        for _, row in frame.iterrows():
            scope = ArtifactScope(**json.loads(row["scope"]))
            items.append(
                ArtifactRef(
                    artifact_id=str(row["artifact_id"]),
                    artifact_type=ArtifactType(str(row["artifact_type"])),
                    node_name=str(row["node_name"]),
                    path=str(row["path"]),
                    scope=scope,
                    schema=json.loads(row["schema"]),
                    metadata=json.loads(row["metadata"]),
                    upstream_ids=tuple(json.loads(row["upstream_ids"])),
                )
            )
        return cls(items)

    def save(self, path: Path) -> None:
        write_table(self.to_frame(), path)

    @classmethod
    def load(cls, path: Path) -> "ArtifactIndex":
        return cls.from_frame(read_table(path))

    def list_nodes(self) -> list[str]:
        return sorted({i.node_name for i in self.items})

    def by_node(self, node_name: str) -> list[ArtifactRef]:
        return [i for i in self.items if i.node_name == node_name]

    def list_artifacts(
        self,
        *,
        node_name: str | None = None,
        artifact_type: str | None = None,
    ) -> list[ArtifactRef]:
        refs = self.items
        if node_name is not None:
            refs = [r for r in refs if r.node_name == node_name]
        if artifact_type is not None:
            refs = [r for r in refs if str(r.artifact_type.value if hasattr(r.artifact_type, "value") else r.artifact_type) == artifact_type]
        return refs

    def filter(
        self,
        *,
        node_name: str | None = None,
        artifact_type: ArtifactType | None = None,
        outer_fold_id: int | None = None,
        split_role: SplitRole | None = None,
    ) -> list[ArtifactRef]:
        out = self.items
        if node_name is not None:
            out = [i for i in out if i.node_name == node_name]
        if artifact_type is not None:
            out = [i for i in out if i.artifact_type == artifact_type]
        if outer_fold_id is not None:
            out = [i for i in out if i.scope.outer_fold_id == outer_fold_id]
        if split_role is not None:
            out = [i for i in out if i.scope.split_role == split_role]
        return out

    def lineage(self, artifact_id: str) -> list[ArtifactRef]:
        lookup = {i.artifact_id: i for i in self.items}
        ordered: list[ArtifactRef] = []

        def _visit(aid: str) -> None:
            if aid not in lookup:
                return
            node = lookup[aid]
            if node in ordered:
                return
            for up in node.upstream_ids:
                _visit(up)
            ordered.append(node)

        _visit(artifact_id)
        return ordered


def load_artifact_table(ref: ArtifactRef) -> pd.DataFrame:
    return read_table(Path(ref.path))


class ArtifactStore:
    """Filesystem-backed artifact store with typed references and index."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.index_path = run_dir / "artifact_index.parquet"
        if self.index_path.exists():
            self.index = ArtifactIndex.load(self.index_path)
        else:
            self.index = ArtifactIndex()

    def _scope_dir(self, scope: ArtifactScope) -> Path:
        if scope.cv_level == "global":
            return self.run_dir / "global"
        if scope.outer_fold_id is None:
            return self.run_dir / "outer_fold_000"
        return self.run_dir / f"outer_fold_{scope.outer_fold_id:03d}"

    def path_for(
        self,
        *,
        node_name: str,
        artifact_type: str,
        scope: ArtifactScope,
        suffix: str = "",
    ) -> str:
        d = self._scope_dir(scope) / node_name
        d.mkdir(parents=True, exist_ok=True)
        safe_suffix = f"_{suffix}" if suffix else ""
        filename = f"{artifact_type}{safe_suffix}.parquet"
        return str((d / filename).relative_to(self.run_dir))

    def create_ref(
        self,
        *,
        artifact_type: str,
        node_name: str,
        path: str,
        scope: ArtifactScope,
        schema: dict[str, object],
        metadata: dict[str, object],
        upstream_ids: list[str],
    ) -> ArtifactRef:
        try:
            a_type = ArtifactType(artifact_type)
        except Exception:
            # Fallback to a known value to keep minimal engine permissive.
            a_type = ArtifactType.FEATURE_TABLE
        ref = ArtifactRef.create(
            artifact_type=a_type,
            node_name=node_name,
            path=path,
            scope=scope,
            schema=schema,
            metadata=metadata,
            upstream_ids=tuple(upstream_ids),
        )
        self.index.add(ref)
        self.index.save(self.index_path)
        return ref

    def write(
        self,
        *,
        artifact_type: str,
        node_name: str,
        scope: ArtifactScope,
        data: pd.DataFrame,
        schema: dict[str, object],
        metadata: dict[str, object],
        upstream_ids: list[str],
        suffix: str = "",
    ) -> ArtifactRef:
        rel_path = self.path_for(
            node_name=node_name,
            artifact_type=artifact_type,
            scope=scope,
            suffix=suffix,
        )
        abs_path = self.run_dir / rel_path
        write_table(data, abs_path)
        return self.create_ref(
            artifact_type=artifact_type,
            node_name=node_name,
            path=rel_path,
            scope=scope,
            schema=schema,
            metadata=metadata,
            upstream_ids=upstream_ids,
        )

    def persist_index(self, index: ArtifactIndex) -> None:
        self.index = index
        self.index.save(self.index_path)

    def load(self, ref_or_id: ArtifactRef | str) -> pd.DataFrame:
        if isinstance(ref_or_id, ArtifactRef):
            ref = ref_or_id
        else:
            lookup = {r.artifact_id: r for r in self.index.items}
            ref = lookup[ref_or_id]
        return read_table(self.run_dir / ref.path)
