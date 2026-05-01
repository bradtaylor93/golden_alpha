"""Performance metric helpers for daily strategy returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class PerformanceSummary:
    """Annualized and average statistics for one return stream."""

    observations: int
    total_return: float
    annual_return: float
    annual_std: float
    sharpe: float
    average_daily_return: float
    average_daily_turnover: float
    max_drawdown: float
    hit_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "observations": self.observations,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "annual_std": self.annual_std,
            "sharpe": self.sharpe,
            "average_daily_return": self.average_daily_return,
            "average_daily_turnover": self.average_daily_turnover,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
        }


def summarize_returns(returns: pd.Series, turnover: pd.Series | None = None) -> PerformanceSummary:
    """Summarize daily returns with annualized and simple average statistics."""

    clean = returns.dropna().astype(float)
    if clean.empty:
        return PerformanceSummary(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    equity = (1.0 + clean).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = max(len(clean) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    annual_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    annual_std = float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(clean) > 1 else 0.0
    sharpe = float(annual_return / annual_std) if annual_std > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    aligned_turnover = turnover.reindex(clean.index).fillna(0.0) if turnover is not None else pd.Series(0.0, index=clean.index)

    return PerformanceSummary(
        observations=int(len(clean)),
        total_return=total_return,
        annual_return=annual_return,
        annual_std=annual_std,
        sharpe=sharpe,
        average_daily_return=float(clean.mean()),
        average_daily_turnover=float(aligned_turnover.mean()),
        max_drawdown=float(drawdown.min()),
        hit_rate=float((clean > 0.0).mean()),
    )


def summarize_by_group(
    daily: pd.DataFrame,
    group_column: str,
    return_column: str = "net_return",
    turnover_column: str = "turnover",
) -> pd.DataFrame:
    """Compute performance summaries by a categorical daily grouping column."""

    rows: list[dict[str, float | int | str]] = []
    for group_name, group in daily.groupby(group_column, dropna=False):
        summary = summarize_returns(group.set_index("date")[return_column], group.set_index("date")[turnover_column])
        row = {"group": str(group_name)}
        row.update(summary.as_dict())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
