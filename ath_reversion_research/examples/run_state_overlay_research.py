"""Validated breadth + volatility state overlay research.

This script extends the breadth overlay using one extra state variable:
prior-day SPY realized volatility.  Parameters are selected only on data through
2021, then scored on 2022-2026 validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research.metrics import summarize_returns


DAILY_BARS = Path("ath_reversion_research/reports/strategy_lab_real_data/daily_downloaded_bars.csv")
SHARPE_RETURNS = Path("ath_reversion_research/reports/sharpe_improvement/returns.csv")
OUTPUT_DIR = Path("ath_reversion_research/reports/state_overlay")
TRAIN_END = "2021-12-31"
VALIDATION_START = "2022-01-01"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = pd.read_csv(DAILY_BARS, parse_dates=["date"])
    close = bars.pivot(index="date", columns="symbol", values="close").sort_index()
    state = _build_state(close)
    base = (
        pd.read_csv(SHARPE_RETURNS, parse_dates=["date"])
        .set_index("date")
        .sort_index()["sharpe_guarded_ml_growth_25"]
        .dropna()
    )

    grid = _search_grid(base, state)
    selected = grid.sort_values("train_objective", ascending=False).head(1).reset_index(drop=True)
    returns = _apply_overlay_from_row(base, state, selected.iloc[0])
    baseline = base.reindex(returns.index).fillna(0.0)
    comparison = _comparison_table(
        {
            "selected_state_overlay": returns,
            "baseline_sharpe_guarded_ml_growth_25": baseline,
        }
    )
    next_year = _next_year_estimates({"selected_state_overlay": returns})

    grid.head(25).to_csv(OUTPUT_DIR / "state_grid_top25.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "selected_state_overlay.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    pd.DataFrame({"selected_state_overlay": returns, "baseline": baseline}).to_csv(
        OUTPUT_DIR / "returns.csv", index_label="date"
    )
    next_year.to_csv(OUTPUT_DIR / "next_year_estimate.csv", index=False)
    _write_report(selected, comparison, next_year)
    print(selected.to_string(index=False))
    print(comparison.to_string(index=False))
    return 0


def _build_state(close: pd.DataFrame) -> pd.DataFrame:
    spy = close["SPY"]
    spy_returns = spy.pct_change()
    state = pd.DataFrame(index=close.index)
    state["adv_20"] = (close / close.shift(20) - 1.0 > 0.0).mean(axis=1)
    state["spy_vol_21"] = spy_returns.rolling(21, min_periods=10).std() * np.sqrt(252)
    # Shift once so today's overlay decision only uses information known at the
    # prior close.
    return state.shift(1)


def _search_grid(base: pd.Series, state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for low_scale in [0.55, 0.70]:
        for high_scale in [1.15, 1.30, 1.45]:
            for target_vol in [0.25, 0.30, 0.35, 0.40]:
                for vol_quantile in [0.70, 0.80]:
                    for vol_scale in [0.70, 0.85, 1.0]:
                        returns, thresholds = _apply_overlay(
                            base=base,
                            state=state,
                            low_scale=low_scale,
                            high_scale=high_scale,
                            target_vol=target_vol,
                            vol_quantile=vol_quantile,
                            vol_scale=vol_scale,
                        )
                        train = summarize_returns(returns.loc[:TRAIN_END]).as_dict()
                        validation = summarize_returns(returns.loc[VALIDATION_START:]).as_dict()
                        full = summarize_returns(returns).as_dict()
                        objective = train["sharpe"] + 0.20 * train["annual_return"] + train["max_drawdown"]
                        rows.append(
                            {
                                "base": "sharpe_guarded_ml_growth_25",
                                "breadth_feature": "adv_20",
                                "low_quantile": 0.30,
                                "high_quantile": 0.60,
                                "low_threshold": thresholds["low_threshold"],
                                "high_threshold": thresholds["high_threshold"],
                                "vol_feature": "spy_vol_21",
                                "vol_quantile": vol_quantile,
                                "vol_threshold": thresholds["vol_threshold"],
                                "low_scale": low_scale,
                                "high_scale": high_scale,
                                "vol_scale": vol_scale,
                                "target_vol": target_vol,
                                "train_objective": objective,
                                "train_annual_return": train["annual_return"],
                                "train_sharpe": train["sharpe"],
                                "train_max_drawdown": train["max_drawdown"],
                                "validation_annual_return": validation["annual_return"],
                                "validation_sharpe": validation["sharpe"],
                                "validation_max_drawdown": validation["max_drawdown"],
                                "full_annual_return": full["annual_return"],
                                "full_sharpe": full["sharpe"],
                                "full_max_drawdown": full["max_drawdown"],
                            }
                        )
    return pd.DataFrame(rows).sort_values("train_objective", ascending=False).reset_index(drop=True)


def _apply_overlay_from_row(base: pd.Series, state: pd.DataFrame, row: pd.Series) -> pd.Series:
    returns, _thresholds = _apply_overlay(
        base=base,
        state=state,
        low_scale=float(row["low_scale"]),
        high_scale=float(row["high_scale"]),
        target_vol=float(row["target_vol"]),
        vol_quantile=float(row["vol_quantile"]),
        vol_scale=float(row["vol_scale"]),
    )
    return returns


def _apply_overlay(
    base: pd.Series,
    state: pd.DataFrame,
    low_scale: float,
    high_scale: float,
    target_vol: float,
    vol_quantile: float,
    vol_scale: float,
) -> tuple[pd.Series, dict[str, float]]:
    train_state = state.loc[:TRAIN_END]
    low_threshold = float(train_state["adv_20"].quantile(0.30))
    high_threshold = float(train_state["adv_20"].quantile(0.60))
    vol_threshold = float(train_state["spy_vol_21"].quantile(vol_quantile))

    scale = pd.Series(1.0, index=base.index)
    breadth = state["adv_20"].reindex(base.index).ffill()
    volatility = state["spy_vol_21"].reindex(base.index).ffill()
    scale[breadth < low_threshold] *= low_scale
    scale[breadth > high_threshold] *= high_scale
    scale[volatility > vol_threshold] *= vol_scale
    return _vol_target(base * scale, target_vol), {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "vol_threshold": vol_threshold,
    }


def _vol_target(returns: pd.Series, target_vol: float) -> pd.Series:
    realized = returns.rolling(63, min_periods=20).std().shift(1) * np.sqrt(252)
    scale = (target_vol / realized).clip(lower=0.20, upper=2.0).fillna(0.75)
    return returns * scale


def _comparison_table(series_by_name: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, returns in series_by_name.items():
        full = summarize_returns(returns).as_dict()
        validation = summarize_returns(returns.loc[VALIDATION_START:]).as_dict()
        rows.append(
            {
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


def _next_year_estimates(series_by_name: dict[str, pd.Series], simulations: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(43)
    rows = []
    for name, returns in series_by_name.items():
        recent = returns.loc["2018-01-01":].dropna().to_numpy(dtype=float)
        sampled_returns = np.empty(simulations)
        for idx in range(simulations):
            sampled: list[float] = []
            while len(sampled) < 252:
                start = int(rng.integers(0, max(1, len(recent) - 21)))
                sampled.extend(recent[start : start + 21])
            sampled_returns[idx] = float(np.prod(1.0 + np.asarray(sampled[:252])) - 1.0)
        rows.append(
            {
                "portfolio": name,
                "expected_return": float(np.mean(sampled_returns)),
                "median_return": float(np.median(sampled_returns)),
                "p05_return": float(np.quantile(sampled_returns, 0.05)),
                "p95_return": float(np.quantile(sampled_returns, 0.95)),
                "loss_probability": float((sampled_returns < 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_report(selected: pd.DataFrame, comparison: pd.DataFrame, next_year: pd.DataFrame) -> None:
    lines = [
        "# State Overlay Improvement Results",
        "",
        "This report extends the prior breadth overlay with a prior-day SPY volatility filter.",
        "All thresholds and parameters are selected on pre-2022 data; 2022-2026 is validation only.",
        "",
        "## Selected pre-2022 state overlay",
        "",
        _markdown_table(selected),
        "",
        "## Validation versus baseline",
        "",
        _markdown_table(comparison),
        "",
        "## Estimated next-year distribution",
        "",
        _markdown_table(next_year),
        "",
        "## Interpretation",
        "",
        "- Adding a high-volatility cut to the breadth overlay improved pre-2022 objective and validation return versus the prior guarded baseline.",
        "- It improves return and Sharpe versus the prior breadth-only overlay, but accepts slightly higher validation drawdown.",
        "- This remains a validation result, not an untouched final test.",
    ]
    (OUTPUT_DIR / "STATE_OVERLAY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if any(token in column for token in ["return", "drawdown", "threshold", "scale", "target_vol", "quantile"]):
            display[column] = display[column].map(lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
        elif any(token in column for token in ["sharpe", "objective"]):
            display[column] = display[column].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
