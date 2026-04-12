"""Prediction task abstraction and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from trading_research.utils.typing import TaskType


TargetBuilder = Callable[[pd.DataFrame, int], pd.DataFrame]


@dataclass(frozen=True)
class PredictionTask:
    name: str
    task_type: TaskType
    horizon: int
    build_target_fn: TargetBuilder
    metric_names: tuple[str, ...]
    embargo: int = 0

    def build_target(self, bars: pd.DataFrame) -> pd.DataFrame:
        frame = self.build_target_fn(bars, self.horizon)
        if "timestamp" not in frame.columns or "asset" not in frame.columns or "target" not in frame.columns:
            raise ValueError("Target builder must return timestamp/asset/target columns")
        return frame.dropna().sort_values(["timestamp", "asset"]).reset_index(drop=True)


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, PredictionTask] = {}

    def register(self, task: PredictionTask) -> None:
        self._tasks[task.name] = task

    def get(self, name: str) -> PredictionTask:
        if name not in self._tasks:
            raise KeyError(f"Unknown prediction task: {name}")
        return self._tasks[name]

    def build_target_table(self, bars: pd.DataFrame, task_name: str) -> pd.DataFrame:
        task = self.get(task_name)
        out = task.build_target(bars)
        out["task_name"] = task.name
        return out.reset_index(drop=True)
