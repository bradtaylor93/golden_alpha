"""Configuration models and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"
    asset: str = "SPY"
    start: str = "2010-01-01"
    end: str | None = None
    period: str | None = None
    interval: str = "1d"
    csv_path: str | None = None
    date_col: str = "date"
    close_col: str = "close"
    random_state: int = 42
    synthetic_rows: int = 1500


@dataclass(frozen=True)
class VolatilityConfig:
    method: str = "ewma_std"
    return_type: str = "log"
    rolling_window: int = 20
    ewma_span: int = 20


@dataclass(frozen=True)
class FeatureConfig:
    lookback_window: int = 50
    modes: tuple[str, ...] = ("summary", "quantile", "temporal")
    quantile_points: int = 25
    histogram_bins: int = 20
    temporal_downsample: int = 12
    temporal_dct_components: int = 8
    temporal_acf_lags: int = 5
    spike_quantile: float = 0.90
    run_quantile: float = 0.75
    pacf_lags: int = 3


@dataclass(frozen=True)
class EmbeddingConfig:
    method: str = "pca"
    standardize: bool = True
    pca_variance_threshold: float = 0.95
    max_components: int = 20


@dataclass(frozen=True)
class ClusteringConfig:
    methods: tuple[str, ...] = ("kmeans", "gmm")
    k_min: int = 3
    k_max: int = 6
    random_state: int = 42
    min_state_occupancy: float = 0.05
    select_metric: str = "silhouette"


@dataclass(frozen=True)
class TransitionConfig:
    markov_order: int = 1
    laplace_smoothing: float = 1.0
    fit_second_order: bool = False
    fit_conditional_model: bool = True
    conditional_max_iter: int = 500


@dataclass(frozen=True)
class EvaluationConfig:
    train_ratio: float = 0.70
    top_k_accuracy: int = 2


@dataclass(frozen=True)
class OutputConfig:
    base_dir: str = "/workspace/vol_shape_regimes/outputs"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    data: DataConfig
    volatility: VolatilityConfig
    features: FeatureConfig
    embedding: EmbeddingConfig
    clustering: ClusteringConfig
    transitions: TransitionConfig
    evaluation: EvaluationConfig
    output: OutputConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config at {path} must be a mapping.")
    return raw


def _resolve_inheritance(path: Path) -> dict[str, Any]:
    cfg = _load_yaml(path)
    inherits = cfg.pop("inherits", None)
    if not inherits:
        return cfg
    parent_path = Path(inherits).expanduser()
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()
    parent_cfg = _resolve_inheritance(parent_path)
    return _deep_merge(parent_cfg, cfg)


def _tupleize(items: Any) -> tuple[str, ...]:
    if isinstance(items, tuple):
        return items
    if isinstance(items, list):
        return tuple(str(v) for v in items)
    raise TypeError("Expected list/tuple for tupleized config field.")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load experiment config from YAML with optional inheritance."""
    resolved = _resolve_inheritance(Path(path).expanduser().resolve())
    return ExperimentConfig(
        experiment_name=str(resolved.get("experiment_name", "experiment_default")),
        data=DataConfig(**resolved.get("data", {})),
        volatility=VolatilityConfig(**resolved.get("volatility", {})),
        features=FeatureConfig(
            **{
                **resolved.get("features", {}),
                "modes": _tupleize(resolved.get("features", {}).get("modes", ["summary", "quantile", "temporal"])),
            }
        ),
        embedding=EmbeddingConfig(**resolved.get("embedding", {})),
        clustering=ClusteringConfig(
            **{
                **resolved.get("clustering", {}),
                "methods": _tupleize(resolved.get("clustering", {}).get("methods", ["kmeans", "gmm"])),
            }
        ),
        transitions=TransitionConfig(**resolved.get("transitions", {})),
        evaluation=EvaluationConfig(**resolved.get("evaluation", {})),
        output=OutputConfig(**resolved.get("output", {})),
    )
