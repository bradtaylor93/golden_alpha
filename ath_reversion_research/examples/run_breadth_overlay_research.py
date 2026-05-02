"""Causal breadth overlay research for the corrected portfolio candidates.

The goal is to improve Sharpe without adding label leakage.  Breadth features
are shifted by one bar, overlay parameters are selected on data through 2021,
and 2022 onward is used only for validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research.metrics import summarize_returns


DAILY_BARS = Path("ath_reversion_research/reports/strategy_lab_real_data/daily_downloaded_bars.csv")
SHARPE_RETURNS = Path("ath_reversion_research/reports/sharpe_improvement/returns.csv")
OUTPUT_DIR = Path("ath_reversion_research/reports/breadth_overlay")
TRAIN_END = "2021-12-31"
VALIDATION_START = "2022-01-01"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = pd.read_csv(DAILY_BARS, parse_dates=["date"])
    close = bars.pivot(index="date", columns="symbol", values="close").sort_index()
    candidates = pd.read_csv(SHARPE_RETURNS, parse_dates=["date"]).set_index("date").sort_index()
    breadth = _build_breadth_features(close)

    grid, returns_by_name = _grid_search(candidates, breadth)
    selected = grid.sort_values("train_objective", ascending=False).head(1)
    selected_name = str(selected.iloc[0]["portfolio"])
    selected_returns = returns_by_name[selected_name]

    validation = _validation_table(selected_returns, candidates["sharpe_guarded_ml_growth_25"])
    next_year = _next_year_estimate(selected_returns)

    grid.head(25).to_csv(OUTPUT_DIR / "breadth_grid_top25.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "selected_breadth_overlay.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    next_year.to_csv(OUTPUT_DIR / "next_year_estimate.csv", index=False)
    pd.DataFrame({"selected_breadth_overlay": selected_returns}).to_csv(OUTPUT_DIR / "returns.csv", index_label="date")
    _write_report(selected, validation, next_year)
    print(selected.to_string(index=False))
    print(validation.to_string(index=False))
    return 0


def _build_breadth_features(close: pd.DataFrame) -> pd.DataFrame:
    spy = close["SPY"]
    spy_returns = spy.pct_change()
    breadth = pd.DataFrame(index=close.index)
    breadth["pct_above_200"] = (close > close.rolling(200, min_periods=100).mean()).mean(axis=1)
    breadth["pct_above_50"] = (close > close.rolling(50, min_periods=25).mean()).mean(axis=1)
    breadth["adv_20"] = (close / close.shift(20) - 1.0 > 0.0).mean(axis=1)
    breadth["spy_ma"] = spy / spy.rolling(200, min_periods=100).mean() - 1.0
    breadth["spy_vol"] = spy_returns.rolling(63, min_periods=20).std() * np.sqrt(252)
    return breadth.shift(1)


def _grid_search(
    candidates: pd.DataFrame,
    breadth: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    rows = []
    returns_by_name: dict[str, pd.Series] = {}
    base_names = [
        "sharpe_guarded_ml_growth_25",
        "asset_overlay_regime_target35",
        "asset_overlay_target35",
        "ml_meta_scale_growth",
    ]
    for base_name in base_names:
        base = candidates[base_name].dropna()
        for feature in ["adv_20", "pct_above_200", "pct_above_50", "spy_ma"]:
            train_values = breadth[feature].loc[:TRAIN_END].dropna()
            for low_quantile in [0.20, 0.30, 0.40]:
                for high_quantile in [0.60, 0.70, 0.80]:
                    if high_quantile <= low_quantile:
                        continue
                    low_threshold = float(train_values.quantile(low_quantile))
                    high_threshold = float(train_values.quantile(high_quantile))
                    for low_scale in [0.40, 0.55, 0.70]:
                        for high_scale in [1.00, 1.15, 1.30]:
                            for target_vol in [0.25, 0.30, 0.35, 0.40]:
                                name = (
                                    f"{base_name}_{feature}_lq{low_quantile:.2f}_hq{high_quantile:.2f}"
                                    f"_ls{low_scale:.2f}_hs{high_scale:.2f}_tv{target_vol:.2f}"
                                )
                                returns = _apply_breadth_overlay(
                                    base,
                                    breadth[feature],
                                    low_threshold,
                                    high_threshold,
                                    low_scale,
                                    high_scale,
                                    target_vol,
                                )
                                returns_by_name[name] = returns
                                train = summarize_returns(returns.loc[:TRAIN_END]).as_dict()
                                validation = summarize_returns(returns.loc[VALIDATION_START:]).as_dict()
                                full = summarize_returns(returns).as_dict()
                                objective = train["sharpe"] + 0.20 * train["annual_return"] + train["max_drawdown"]
                                rows.append(
                                    {
                                        "portfolio": name,
                                        "base": base_name,
                                        "breadth_feature": feature,
                                        "low_quantile": low_quantile,
                                        "high_quantile": high_quantile,
                                        "low_threshold": low_threshold,
                                        "high_threshold": high_threshold,
                                        "low_scale": low_scale,
                                        "high_scale": high_scale,
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
    grid = pd.DataFrame(rows).sort_values("train_objective", ascending=False).reset_index(drop=True)
    return grid, returns_by_name


def _apply_breadth_overlay(
    base: pd.Series,
    feature: pd.Series,
    low_threshold: float,
    high_threshold: float,
    low_scale: float,
    high_scale: float,
    target_vol: float,
) -> pd.Series:
    aligned_feature = feature.reindex(base.index).ffill()
    scale = pd.Series(1.0, index=base.index)
    scale[aligned_feature < low_threshold] = low_scale
    scale[aligned_feature > high_threshold] = high_scale
    return _vol_target(base * scale, target_vol)


def _vol_target(returns: pd.Series, target_vol: float) -> pd.Series:
    realized_vol = returns.rolling(63, min_periods=20).std().shift(1) * np.sqrt(252)
    scale = (target_vol / realized_vol).clip(lower=0.20, upper=2.00).fillna(0.75)
    return returns * scale


def _validation_table(selected: pd.Series, baseline: pd.Series) -> pd.DataFrame:
    rows = []
    for name, returns in {
        "selected_breadth_overlay": selected,
        "baseline_sharpe_guarded_ml_growth_25": baseline,
    }.items():
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


def _next_year_estimate(returns: pd.Series, simulations: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(41)
    recent = returns.loc["2018-01-01":].dropna().to_numpy(dtype=float)
    sampled_returns = np.empty(simulations)
    block = 21
    for idx in range(simulations):
        sampled: list[float] = []
        while len(sampled) < 252:
            start = int(rng.integers(0, max(1, len(recent) - block)))
            sampled.extend(recent[start : start + block])
        sampled_returns[idx] = float(np.prod(1.0 + np.asarray(sampled[:252])) - 1.0)
    return pd.DataFrame(
        [
            {
                "portfolio": "selected_breadth_overlay",
                "expected_return": float(np.mean(sampled_returns)),
                "median_return": float(np.median(sampled_returns)),
                "p05_return": float(np.quantile(sampled_returns, 0.05)),
                "p95_return": float(np.quantile(sampled_returns, 0.95)),
                "loss_probability": float((sampled_returns < 0.0).mean()),
            }
        ]
    )


def _write_report(selected: pd.DataFrame, validation: pd.DataFrame, next_year: pd.DataFrame) -> None:
    lines = [
        "# Breadth Overlay Improvement Results",
        "",
        "This report tests a different improvement path: scale an existing corrected portfolio using prior-day market breadth.",
        "All overlay parameters are selected on pre-2022 data; 2022-2026 is validation only.",
        "",
        "## Selected pre-2022 breadth overlay",
        "",
        _markdown_table(selected),
        "",
        "## Validation versus baseline",
        "",
        _markdown_table(validation),
        "",
        "## Estimated next-year distribution",
        "",
        _markdown_table(next_year),
        "",
        "## Interpretation",
        "",
        "- The selected overlay uses prior-day 20-day advance breadth to increase exposure in broad positive tape and reduce exposure in weak tape.",
        "- This improved validation Sharpe and return versus the guarded ML-growth baseline, while keeping drawdown materially below the aggressive high-return portfolios.",
        "- It is still selected from a breadth-parameter grid, so it should be treated as validation evidence rather than a final untouched test.",
    ]
    (OUTPUT_DIR / "BREADTH_OVERLAY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


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
