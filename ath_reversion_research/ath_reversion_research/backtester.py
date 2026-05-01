"""Walk-forward portfolio backtester for causal daily strategy signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .metrics import summarize_returns
from .regimes import RegimeConfig, classify_market_regimes
from .strategies import Strategy


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for out-of-sample walk-forward evaluation."""

    train_days: int = 756
    test_days: int = 126
    step_days: int = 126
    execution_lag_days: int = 1
    transaction_cost_bps: float = 10.0
    annualization: int = 252
    max_gross_exposure: float = 1.0
    benchmark_symbol: str = "SPY"


@dataclass(frozen=True)
class WalkForwardFold:
    """Date bounds for one walk-forward test segment."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardBacktester:
    """Run strategies over explicit OOS walk-forward folds."""

    def __init__(
        self,
        bars: pd.DataFrame,
        config: BacktestConfig | None = None,
        regime_config: RegimeConfig | None = None,
    ) -> None:
        self.bars = bars.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
        self.config = config or BacktestConfig()
        self.regime_config = regime_config or RegimeConfig(
            benchmark_symbol=self.config.benchmark_symbol
        )

    def run(self, strategies: list[Strategy]) -> dict[str, pd.DataFrame]:
        """Run all strategies and return positions, returns, folds, and metrics."""

        folds = self.build_folds()
        if not folds:
            raise ValueError("Not enough history to create a walk-forward fold")

        returns_by_strategy: list[pd.DataFrame] = []
        positions_by_strategy: list[pd.DataFrame] = []
        for strategy in strategies:
            positions = strategy.generate_positions(self.bars)
            portfolio_returns = self._evaluate_strategy(strategy.name, positions, folds)
            returns_by_strategy.append(portfolio_returns)
            positions_by_strategy.append(positions.assign(strategy=strategy.name))

        returns = pd.concat(returns_by_strategy, ignore_index=True)
        positions = pd.concat(positions_by_strategy, ignore_index=True)
        regimes = classify_market_regimes(self.bars, self.regime_config)
        returns = returns.merge(regimes, on="date", how="left")
        summary = self._summarize(returns)
        regime_summary = self._summarize_by_regime(returns)
        folds_frame = pd.DataFrame([asdict(fold) for fold in folds])

        return {
            "positions": positions,
            "returns": returns,
            "folds": folds_frame,
            "summary": summary,
            "regime_summary": regime_summary,
        }

    def write_report(self, results: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
        """Persist machine-readable CSV outputs for downstream review."""

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in results.items():
            frame.to_csv(out / f"{name}.csv", index=False)

    def build_folds(self) -> list[WalkForwardFold]:
        """Create expanding walk-forward folds from available trading dates."""

        dates = pd.Index(sorted(self.bars["date"].dropna().unique()))
        folds: list[WalkForwardFold] = []
        start = self.config.train_days
        fold_id = 0
        while start < len(dates):
            test_end_idx = min(start + self.config.test_days, len(dates))
            if test_end_idx <= start:
                break
            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=pd.Timestamp(dates[0]),
                    train_end=pd.Timestamp(dates[start - 1]),
                    test_start=pd.Timestamp(dates[start]),
                    test_end=pd.Timestamp(dates[test_end_idx - 1]),
                )
            )
            fold_id += 1
            start += self.config.step_days
        return folds

    def _evaluate_strategy(
        self,
        strategy_name: str,
        positions: pd.DataFrame,
        folds: list[WalkForwardFold],
    ) -> pd.DataFrame:
        panel = self._asset_return_panel(positions)
        rows: list[pd.DataFrame] = []
        for fold in folds:
            mask = (panel["date"] >= fold.test_start) & (panel["date"] <= fold.test_end)
            fold_panel = panel.loc[mask].copy()
            if fold_panel.empty:
                continue
            fold_portfolio = self._portfolio_returns(fold_panel)
            fold_portfolio["strategy"] = strategy_name
            fold_portfolio["fold_id"] = fold.fold_id
            rows.append(fold_portfolio)
        return pd.concat(rows, ignore_index=True) if rows else _empty_returns()

    def _asset_return_panel(self, positions: pd.DataFrame) -> pd.DataFrame:
        returns = self.bars.loc[:, ["date", "symbol", "close"]].copy()
        returns["asset_return"] = returns.groupby("symbol")["close"].pct_change()
        merged = returns.merge(
            positions.loc[:, ["date", "symbol", "position"]],
            on=["date", "symbol"],
            how="left",
        )
        merged["position"] = merged["position"].fillna(0.0)
        merged["executed_position"] = merged.groupby("symbol")["position"].shift(
            self.config.execution_lag_days
        )
        merged["executed_position"] = merged["executed_position"].fillna(0.0)
        merged["position_change"] = (
            merged.groupby("symbol")["executed_position"].diff().abs().fillna(
                merged["executed_position"].abs()
            )
        )
        cost = self.config.transaction_cost_bps / 10_000.0
        merged["net_asset_return"] = (
            merged["executed_position"] * merged["asset_return"].fillna(0.0)
            - merged["position_change"] * cost
        )
        return merged

    def _portfolio_returns(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Average active asset returns and cap aggregate gross exposure."""

        def aggregate(day: pd.DataFrame) -> pd.Series:
            gross = float(day["executed_position"].abs().sum())
            if gross <= 0.0:
                return pd.Series(
                    {
                        "net_return": 0.0,
                        "gross_exposure": 0.0,
                        "active_positions": 0,
                        "turnover": float(day["position_change"].sum()),
                    }
                )
            scale = min(1.0, self.config.max_gross_exposure / gross)
            return pd.Series(
                {
                    "net_return": float((day["net_asset_return"] * scale).sum()),
                    "gross_exposure": gross * scale,
                    "active_positions": int((day["executed_position"] != 0.0).sum()),
                    "turnover": float(day["position_change"].sum() * scale),
                }
            )

        daily = panel.groupby("date", group_keys=False).apply(aggregate)
        return daily.reset_index()

    def _summarize(self, returns: pd.DataFrame) -> pd.DataFrame:
        summaries = []
        for strategy, group in returns.groupby("strategy"):
            daily = group.set_index("date")
            summary = summarize_returns(daily["net_return"], daily["turnover"]).as_dict()
            summary.update(
                {
                    "strategy": strategy,
                    "avg_gross_exposure": float(group["gross_exposure"].mean()),
                    "avg_active_positions": float(group["active_positions"].mean()),
                    "avg_daily_turnover": float(group["turnover"].mean()),
                    "folds": int(group["fold_id"].nunique()),
                }
            )
            summaries.append(summary)
        return pd.DataFrame(summaries).sort_values("strategy").reset_index(drop=True)

    def _summarize_by_regime(self, returns: pd.DataFrame) -> pd.DataFrame:
        summaries = []
        for (strategy, regime), group in returns.groupby(["strategy", "regime"], dropna=False):
            daily = group.set_index("date")
            summary = summarize_returns(daily["net_return"], daily["turnover"]).as_dict()
            summary.update(
                {
                    "strategy": strategy,
                    "regime": regime if pd.notna(regime) else "unclassified",
                    "avg_gross_exposure": float(group["gross_exposure"].mean()),
                    "avg_active_positions": float(group["active_positions"].mean()),
                    "avg_daily_turnover": float(group["turnover"].mean()),
                }
            )
            summaries.append(summary)
        return pd.DataFrame(summaries).sort_values(["strategy", "regime"]).reset_index(drop=True)


def _empty_returns() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "net_return",
            "gross_exposure",
            "active_positions",
            "turnover",
            "strategy",
            "fold_id",
        ]
    )
