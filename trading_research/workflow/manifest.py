"""Run manifest generation and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
import uuid
from typing import Any


def generate_run_id(prefix: str = "run") -> str:
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"{prefix}_{now}_{short}"


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at_utc: str
    code_version: str
    config_hash: str
    notes: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)

    @classmethod
    def from_json(cls, path: Path) -> "RunManifest":
        return cls.load(path)


def build_manifest(
    *,
    run_id: str,
    workflow_graph: dict[str, Any],
    artifact_index_path: str,
    metadata: dict[str, Any] | None = None,
) -> RunManifest:
    digest = json.dumps(
        {
            "workflow_graph": workflow_graph,
            "artifact_index_path": artifact_index_path,
            "metadata": metadata or {},
        },
        sort_keys=True,
        default=str,
    )
    return RunManifest(
        run_id=run_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        code_version="dev",
        config_hash=str(abs(hash(digest))),
        notes=json.dumps(metadata or {}, sort_keys=True),
    )


def write_manifest(path: Path, manifest: RunManifest) -> None:
    manifest.save(path)
