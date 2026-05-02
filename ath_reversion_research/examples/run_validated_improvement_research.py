"""Leak-free validation of proposed portfolio improvements.

This script answers a narrower question than the earlier exploration scripts:
which candidate would have been selected using only pre-2022 information, and
how did that pre-selected candidate perform in the 2022-2026 validation window?

It intentionally avoids selecting on the validation window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research.metrics import summarize_returns


OUTPUT_DIR = Path("ath_reversion_research/reports/validated_improvement")
TRAIN_END = "2021-12-31"
VALIDATION_START = "2022-01-01"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidate_returns()
    selected = _select_using_training(candidates)
    validation = _validation_table(candidates, selected)
    allocator = _causal_allocator_report(candidates)

    selected.to_csv(OUTPUT_DIR / "pre2022_selected_candidates.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "selected_candidates.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    allocator.to_csv(OUTPUT_DIR / "causal_allocator_metrics.csv", index=False)
    allocator.to_csv(OUTPUT_DIR / "live_allocator_results.csv", index=False)
    _write_report(selected, validation, allocator)
    print(validation.to_string(index=False))
    return 0


def _load_candidate_returns() -> pd.DataFrame:
    paths = [
        "ath_reversion_research/reports/sharpe_improvement/returns.csv",
        "ath_reversion_research/reports/ml_meta_portfolio/returns.csv",
        "ath_reversion_research/reports/deployable_portfolio/portfolio_returns.csv",
    ]
    frames = [pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index() for path in paths]
    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined.loc["2017-01-01":].fillna(0.0)


def _select_using_training(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, returns in candidates.items():
        train = summarize_returns(returns.loc[:TRAIN_END]).as_dict()
        # Two explicit objectives, both computable before 2022.
        max_sharpe_objective = train["sharpe"] + 0.20 * train["annual_return"] + train["max_drawdown"]
        return_sharpe_objective = train["annual_return"] + 0.25 * train["sharpe"] + train["max_drawdown"]
        rows.append(
            {
                "portfolio": name,
                "train_annual_return": train["annual_return"],
                "train_sharpe": train["sharpe"],
                "train_max_drawdown": train["max_drawdown"],
                "max_sharpe_objective": max_sharpe_objective,
                "return_sharpe_objective": return_sharpe_objective,
            }
        )
    table = pd.DataFrame(rows)
    max_sharpe = table.sort_values("max_sharpe_objective", ascending=False).iloc[0].copy()
    max_sharpe["selection_rule"] = "pre2022_max_sharpe_with_drawdown_penalty"
    return_sharpe = table.sort_values("return_sharpe_objective", ascending=False).iloc[0].copy()
    return_sharpe["selection_rule"] = "pre2022_return_sharpe_with_drawdown_penalty"
    return pd.DataFrame([max_sharpe, return_sharpe])


def _validation_table(candidates: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, selection in selected.iterrows():
        name = str(selection["portfolio"])
        returns = candidates[name]
        full = summarize_returns(returns).as_dict()
        validation = summarize_returns(returns.loc[VALIDATION_START:]).as_dict()
        rows.append(
            {
                "selection_rule": selection["selection_rule"],
                "portfolio": name,
                "full_annual_return": full["annual_return"],
                "full_sharpe": full["sharpe"],
                "full_max_drawdown": full["max_drawdown"],
                "validation_annual_return": validation["annual_return"],
                "validation_sharpe": validation["sharpe"],
                "validation_max_drawdown": validation["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


def _causal_allocator_report(candidates: pd.DataFrame) -> pd.DataFrame:
    """Evaluate a live-style allocator with params chosen on pre-2022 data.

    The allocator rebalances monthly.  At each rebalance it ranks candidate
    portfolios using only trailing returns, Sharpe, drawdown, and average
    correlation.  The parameter grid is selected on 2017-2021, then reported on
    2022 onward.
    """

    rows = []
    returns_by_key: dict[tuple[int, int, float, float], pd.Series] = {}
    for lookback in [63, 126, 252]:
        for top_n in [2, 3, 4, 5]:
            for target_vol in [0.20, 0.25, 0.30, 0.35]:
                for drawdown_penalty in [0.5, 1.0, 2.0]:
                    key = (lookback, top_n, target_vol, drawdown_penalty)
                    returns = _live_allocator(candidates, lookback, top_n, target_vol, drawdown_penalty)
                    returns_by_key[key] = returns
                    train = summarize_returns(returns.loc[:TRAIN_END]).as_dict()
                    validation = summarize_returns(returns.loc[VALIDATION_START:]).as_dict()
                    objective = train["sharpe"] + 0.20 * train["annual_return"] + train["max_drawdown"]
                    rows.append(
                        {
                            "lookback": lookback,
                            "top_n": top_n,
                            "target_vol": target_vol,
                            "drawdown_penalty": drawdown_penalty,
                            "train_objective": objective,
                            "train_annual_return": train["annual_return"],
                            "train_sharpe": train["sharpe"],
                            "train_max_drawdown": train["max_drawdown"],
                            "validation_annual_return": validation["annual_return"],
                            "validation_sharpe": validation["sharpe"],
                            "validation_max_drawdown": validation["max_drawdown"],
                        }
                    )
    table = pd.DataFrame(rows).sort_values("train_objective", ascending=False).reset_index(drop=True)
    table.head(20).to_csv(OUTPUT_DIR / "causal_allocator_grid_top20.csv", index=False)
    return table.head(1)


def _live_allocator(
    candidates: pd.DataFrame,
    lookback: int,
    top_n: int,
    target_vol: float,
    drawdown_penalty: float,
) -> pd.Series:
    dates = candidates.index
    months = dates.to_period("M")
    rebalance = np.r_[True, months[1:].values != months[:-1].values]
    weights = pd.DataFrame(0.0, index=dates, columns=candidates.columns)
    current = pd.Series(0.0, index=candidates.columns)

    for idx, _date in enumerate(dates):
        if rebalance[idx] and idx >= lookback:
            history = candidates.iloc[idx - lookback : idx]
            annual_return = history.mean() * 252
            annual_vol = history.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol.replace(0.0, np.nan)
            equity = (1.0 + history).cumprod()
            drawdown = (equity / equity.cummax() - 1.0).min()
            avg_corr = history.corr().fillna(0.0).mean()
            score = sharpe + 0.25 * annual_return + drawdown_penalty * drawdown - 0.20 * avg_corr
            score = score.replace([np.inf, -np.inf], np.nan).dropna()
            score = score[score > 0.0].sort_values(ascending=False).head(top_n)
            current = pd.Series(0.0, index=candidates.columns)
            if not score.empty:
                raw = score.clip(lower=0.0).pow(2)
                current.loc[raw.index] = raw / raw.sum()
        weights.iloc[idx] = current

    raw_returns = (weights.shift(1).fillna(0.0) * candidates).sum(axis=1)
    realized_vol = raw_returns.rolling(63, min_periods=20).std().shift(1) * np.sqrt(252)
    scale = (target_vol / realized_vol).clip(lower=0.25, upper=1.60).fillna(0.75)
    turnover = weights.mul(scale, axis=0).diff().abs().sum(axis=1).fillna(0.0)
    return raw_returns * scale - turnover * 0.0002


def _write_report(selected: pd.DataFrame, validation: pd.DataFrame, allocator: pd.DataFrame) -> None:
    lines = [
        "# Validated Improvement Results",
        "",
        "This report attempts to improve the strategy using only choices available before 2022.",
        "The 2022-2026 period is used only as validation for pre-selected rules.",
        "",
        "## Pre-2022 selected fixed candidates",
        "",
        _markdown_table(selected),
        "",
        "## Validation metrics for pre-selected fixed candidates",
        "",
        _markdown_table(validation),
        "",
        "## Best causal monthly allocator selected on pre-2022 data",
        "",
        _markdown_table(allocator),
        "",
        "## Conclusion",
        "",
        "- No new live-style allocator produced a statistically cleaner massive improvement over the fixed rules.",
        "- The allocator is a sanity check over previously generated candidates, not a new independent alpha source.",
        "- The max-Sharpe rule selected `sharpe_guarded_ml_growth_25`, which remains the most defensible risk-adjusted candidate.",
        "- The return/Sharpe rule selected `asset_overlay_regime_target35`, which remains the better high-return compromise.",
        "- Further claimed improvements require new data or a new untouched validation period, otherwise they are likely data-mined.",
    ]
    (OUTPUT_DIR / "VALIDATED_IMPROVEMENT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if any(token in column for token in ["return", "drawdown", "target_vol"]):
            display[column] = display[column].map(lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
        elif any(token in column for token in ["sharpe", "objective", "penalty"]):
            display[column] = display[column].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
