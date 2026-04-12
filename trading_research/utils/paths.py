"""Filesystem path helpers for run output layout."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(base_runs_dir: Path, run_id: str) -> Path:
    """Return runs/<run_id> directory."""
    return base_runs_dir / run_id


def run_root(base_runs_dir: Path, run_id: str) -> Path:
    """Backward-compatible alias for run directory."""
    return run_dir(base_runs_dir, run_id)


def scope_dir(base_run_dir: Path, cv_level: str, outer_fold_id: int | None = None) -> Path:
    """Return a fold-aware subdirectory path inside a run."""
    if cv_level == "global":
        return base_run_dir / "global"
    if cv_level in {"outer_fold", "inner_fold"}:
        if outer_fold_id is None:
            raise ValueError("outer_fold_id is required for fold-scoped artifacts")
        return base_run_dir / f"outer_fold_{outer_fold_id:03d}"
    raise ValueError(f"Unsupported cv_level: {cv_level}")
