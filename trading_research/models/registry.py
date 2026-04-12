"""Model and training specification layer."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import numpy as np

from trading_research.models.advanced_regression import KernelRidgeRegressor, LoessRegressor
from trading_research.models.classification import LogisticClassifier
from trading_research.models.regression import RidgeRegressor
from trading_research.validation.splits import ValidationSpec


@dataclass(frozen=True)
class ModelSpec:
    name: str
    algorithm: str
    params: dict[str, Any]


@dataclass(frozen=True)
class HyperoptSpec:
    method: str
    objective: str
    param_grid: dict[str, list[Any]] | None = None
    random_space: dict[str, tuple[float, float]] | None = None
    n_iter: int = 10


@dataclass(frozen=True)
class TrainingRecipe:
    selection_validation: ValidationSpec | None = None
    hyperopt: HyperoptSpec | None = None
    refit_on_full_outer_train: bool = True
    generate_train_oof_predictions: bool = True
    save_training_snapshot: bool = True


class ModelRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def register(self, algorithm: str, builder: Callable[[dict[str, Any]], Any]) -> None:
        self._builders[algorithm] = builder

    def build(self, spec: ModelSpec) -> Any:
        if spec.algorithm not in self._builders:
            raise KeyError(f"Unknown model algorithm: {spec.algorithm}")
        return self._builders[spec.algorithm](spec.params)


def default_model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("ridge", lambda p: RidgeRegressor(alpha=float(p.get("alpha", 1.0))))
    registry.register(
        "kernel_ridge",
        lambda p: KernelRidgeRegressor(
            alpha=float(p.get("alpha", 1.0)),
            gamma=float(p.get("gamma", 0.5)),
        ),
    )
    registry.register(
        "loess",
        lambda p: LoessRegressor(
            frac=float(p.get("frac", 0.3)),
            ridge=float(p.get("ridge", 1e-4)),
        ),
    )
    registry.register(
        "logistic",
        lambda p: LogisticClassifier(
            learning_rate=float(p.get("learning_rate", 0.05)),
            steps=int(p.get("steps", 400)),
            l2=float(p.get("l2", 1e-3)),
        ),
    )
    return registry


def grid_candidates(spec: HyperoptSpec | None) -> list[dict[str, Any]]:
    if spec is None or spec.param_grid is None:
        return [{}]
    keys = list(spec.param_grid.keys())
    values = [spec.param_grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]


def regression_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if metric_name == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric_name == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    raise ValueError(f"Unsupported regression metric: {metric_name}")


def classification_metric(metric_name: str, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if metric_name == "accuracy":
        pred = (y_prob >= 0.5).astype(float)
        return float(np.mean(pred == y_true))
    if metric_name == "logloss":
        p = np.clip(y_prob, 1e-8, 1 - 1e-8)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    raise ValueError(f"Unsupported classification metric: {metric_name}")
