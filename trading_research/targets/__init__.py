"""Prediction task exports and default registry builder."""

from trading_research.targets.classification import make_up_down_task
from trading_research.targets.extremes import make_max_drawdown_task, make_max_upside_task
from trading_research.targets.registry import PredictionTask, TaskRegistry
from trading_research.targets.returns import make_forward_return_task
from trading_research.targets.volatility import make_forward_realized_volatility_task


def default_task_registry() -> TaskRegistry:
    registry = TaskRegistry()
    for h in (20, 40):
        registry.register(make_forward_return_task(h))
        registry.register(make_forward_realized_volatility_task(h))
        registry.register(make_up_down_task(h))
        registry.register(make_max_upside_task(h))
        registry.register(make_max_drawdown_task(h))
    return registry


__all__ = ["PredictionTask", "TaskRegistry", "default_task_registry"]
