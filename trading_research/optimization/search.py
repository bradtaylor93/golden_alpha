"""Lightweight hyperparameter search support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trading_research.models.registry import HyperoptSpec


@dataclass(frozen=True)
class SearchResult:
    """Single hyperparameter trial result."""

    params: dict[str, Any]
    score: float


def enumerate_candidates(spec: HyperoptSpec | None) -> list[dict[str, Any]]:
    """Return parameter candidates for simple grid/random search."""
    if spec is None:
        return [{}]
    if spec.method == "grid" and spec.param_grid:
        keys = list(spec.param_grid.keys())
        values = [spec.param_grid[k] for k in keys]
        rows: list[dict[str, Any]] = []
        def build(idx: int, acc: dict[str, Any]) -> None:
            if idx == len(keys):
                rows.append(dict(acc))
                return
            key = keys[idx]
            for value in values[idx]:
                acc[key] = value
                build(idx + 1, acc)
        build(0, {})
        return rows or [{}]
    return [{}]

