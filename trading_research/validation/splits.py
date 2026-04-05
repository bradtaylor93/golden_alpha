"""Walk-forward split generation for outer/inner validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.utils.typing import ValidationMode, ValidationSplitType
from trading_research.validation.embargo import trim_train_with_embargo


@dataclass(frozen=True)
class ValidationSpec:
    split_type: ValidationSplitType
    min_train_size: int
    test_size: int
    embargo: int = 0
    rolling_train_size: int | None = None
    mode: ValidationMode = "pooled"


@dataclass(frozen=True)
class Fold:
    outer_fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class FoldPlan:
    name: str
    folds: tuple[Fold, ...]
    validation_spec: ValidationSpec


def _time_positions(data: pd.DataFrame) -> pd.Index:
    return pd.Index(sorted(pd.to_datetime(data["timestamp"], utc=True).unique()))


def _positions_for_timestamps(
    data: pd.DataFrame,
    timestamps: pd.Index,
) -> np.ndarray:
    mask = pd.to_datetime(data["timestamp"], utc=True).isin(timestamps)
    return np.where(mask.to_numpy())[0]


def build_walk_forward_plan(
    data: pd.DataFrame,
    spec: ValidationSpec,
    n_folds: int,
    plan_name: str = "outer",
) -> FoldPlan:
    """Build expanding or rolling folds using timestamps as split boundaries."""
    unique_ts = _time_positions(data)
    if len(unique_ts) < spec.min_train_size + spec.test_size:
        raise ValueError("Insufficient history for requested split configuration")

    folds: list[Fold] = []
    cursor = spec.min_train_size
    for fold_id in range(n_folds):
        train_end = cursor
        test_end = cursor + spec.test_size
        if test_end > len(unique_ts):
            break
        if spec.split_type == "expanding":
            train_start = 0
        else:
            rolling = spec.rolling_train_size or spec.min_train_size
            train_start = max(0, train_end - rolling)

        train_ts = unique_ts[train_start:train_end]
        test_ts = unique_ts[train_end:test_end]
        train_idx = _positions_for_timestamps(data, train_ts)
        test_idx = _positions_for_timestamps(data, test_ts)
        if len(test_idx) == 0 or len(train_idx) == 0:
            break

        train_idx = trim_train_with_embargo(
            train_idx, test_start_position=int(test_idx.min()), embargo=spec.embargo
        )
        if len(train_idx) == 0:
            break

        folds.append(
            Fold(outer_fold_id=fold_id, train_indices=train_idx, test_indices=test_idx)
        )
        cursor = test_end

    if not folds:
        raise ValueError("No folds were generated for the supplied ValidationSpec")
    return FoldPlan(name=plan_name, folds=tuple(folds), validation_spec=spec)


def build_walk_forward_splits(
    bars: pd.DataFrame,
    *,
    split_type: ValidationSplitType,
    min_train_size: int,
    test_size: int,
    embargo: int = 0,
    rolling_train_size: int | None = None,
    mode: ValidationMode = "pooled",
) -> pd.DataFrame:
    """Compatibility helper returning a tabular fold plan artifact."""
    spec = ValidationSpec(
        split_type=split_type,
        min_train_size=min_train_size,
        test_size=test_size,
        embargo=embargo,
        rolling_train_size=rolling_train_size,
        mode=mode,
    )
    plan = build_walk_forward_plan(data=bars, spec=spec, n_folds=32, plan_name="outer")
    rows: list[dict[str, int]] = []
    for fold in plan.folds:
        rows.append(
            {
                "outer_fold_id": fold.outer_fold_id,
                "train_start_idx": int(fold.train_indices.min()),
                "train_end_idx": int(fold.train_indices.max()),
                "test_start_idx": int(fold.test_indices.min()),
                "test_end_idx": int(fold.test_indices.max()),
            }
        )
    return pd.DataFrame(rows)
