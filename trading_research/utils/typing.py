"""Shared typing aliases and enum definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, TypeAlias

JsonDict: TypeAlias = dict[str, Any]

CvLevel: TypeAlias = Literal["global", "outer_fold", "inner_fold"]
SplitRole: TypeAlias = Literal["full", "train", "test", "oof", "validation"]
TaskType: TypeAlias = Literal["regression", "classification"]
ValidationSplitType: TypeAlias = Literal["expanding", "rolling"]
ValidationMode: TypeAlias = Literal["pooled", "per_asset"]


class ArtifactType(StrEnum):
    BARS = "BarsArtifact"
    TARGET_TABLE = "TargetTableArtifact"
    FEATURE_TABLE = "FeatureTableArtifact"
    FOLD_PLAN = "FoldPlanArtifact"
    PREDICTION = "PredictionArtifact"
    DIAGNOSTIC = "DiagnosticArtifact"
    SELECTION = "SelectionArtifact"
    MODEL_STATE = "ModelStateArtifact"
    TRAINING_SNAPSHOT = "TrainingDatasetSnapshotArtifact"
