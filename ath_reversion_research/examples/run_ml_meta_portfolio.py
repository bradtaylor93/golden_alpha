"""ML-style meta allocation on top of the high-return portfolio.

The model is deliberately simple and auditable: an expanding walk-forward ridge
regression predicts the next 21-trading-day return of the current best
high-return portfolio from lagged market and portfolio state features.  The
prediction is converted into a bounded exposure multiplier.  This is a meta
allocation layer, not a new raw alpha signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research.metrics import summarize_returns


PORTFOLIO_RETURNS = Path("ath_reversion_research/reports/deployable_portfolio/portfolio_returns.csv")
DAILY_BARS = Path("ath_reversion_research/reports/strategy_lab_real_data/daily_downloaded_bars.csv")
OUTPUT_DIR = Path("ath_reversion_research/reports/ml_meta_portfolio")
ANNUALIZATION = 252
META_COST_BPS = 2.0


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    portfolio_returns = pd.read_csv(PORTFOLIO_RETURNS, parse_dates=["date"]).set_index("date").sort_index()
    bars = pd.read_csv(DAILY_BARS, parse_dates=["date"])
    close = bars.pivot(index="date", columns="symbol", values="close").sort_index()

    base = portfolio_returns["max_return_plus_asset_overlay"].loc["2013-01-04":].dropna()
    spy = close["SPY"].reindex(base.index).ffill()
    features = _build_features(base, spy).dropna()
    base = base.reindex(features.index).fillna(0.0)

    predictions = _walk_forward_ridge_predictions(features, _forward_return_target(base, horizon=21))
    aligned_base = base.reindex(predictions.index).fillna(0.0)

    variants = {
        "base_asset_overlay": aligned_base,
        "ml_meta_scale_balanced": _scale_returns(aligned_base, predictions, amplitude=0.25),
        "ml_meta_scale_growth": _scale_returns(aligned_base, predictions, amplitude=0.50),
        "ml_meta_scale_max": _scale_returns(aligned_base, predictions, amplitude=0.75),
    }
    summary = _summary_table(variants)
    holdout = _summary_table({name: returns.loc["2022-01-01":] for name, returns in variants.items()})
    next_year = _next_year_estimates(variants)
    diagnostics = pd.DataFrame(
        {
            "date": predictions.index,
            "base_return": aligned_base,
            "prediction": predictions,
            "balanced_scale": _prediction_scale(predictions, amplitude=0.25),
            "growth_scale": _prediction_scale(predictions, amplitude=0.50),
            "max_scale": _prediction_scale(predictions, amplitude=0.75),
        }
    )

    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    holdout.to_csv(OUTPUT_DIR / "holdout_summary.csv", index=False)
    next_year.to_csv(OUTPUT_DIR / "next_year_estimate.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "diagnostics.csv", index=False)
    pd.DataFrame(variants).to_csv(OUTPUT_DIR / "returns.csv", index_label="date")
    _write_report(summary, holdout, next_year)
    print(summary.to_string(index=False))
    return 0


def _build_features(base: pd.Series, spy: pd.Series) -> pd.DataFrame:
    spy_ret = spy.pct_change()
    equity = (1.0 + base).cumprod()
    features = pd.DataFrame(index=base.index)
    features["spy_mom_21"] = spy / spy.shift(21) - 1.0
    features["spy_mom_63"] = spy / spy.shift(63) - 1.0
    features["spy_mom_126"] = spy / spy.shift(126) - 1.0
    features["spy_ma_dist"] = spy / spy.rolling(200, min_periods=100).mean() - 1.0
    features["spy_vol_21"] = spy_ret.rolling(21, min_periods=10).std() * np.sqrt(ANNUALIZATION)
    features["spy_vol_63"] = spy_ret.rolling(63, min_periods=20).std() * np.sqrt(ANNUALIZATION)
    features["base_mom_21"] = (1.0 + base).rolling(21, min_periods=10).apply(np.prod, raw=True) - 1.0
    features["base_mom_63"] = (1.0 + base).rolling(63, min_periods=20).apply(np.prod, raw=True) - 1.0
    features["base_vol_21"] = base.rolling(21, min_periods=10).std() * np.sqrt(ANNUALIZATION)
    features["base_dd_63"] = equity / equity.rolling(63, min_periods=20).max() - 1.0
    return features.shift(1)


def _forward_return_target(returns: pd.Series, horizon: int) -> pd.Series:
    return (1.0 + returns).rolling(horizon).apply(np.prod, raw=True).shift(-(horizon - 1)) - 1.0


def _walk_forward_ridge_predictions(
    features: pd.DataFrame,
    target: pd.Series,
    alpha: float = 10.0,
    first_test_year: int = 2017,
    embargo_days: int = 21,
) -> pd.Series:
    predictions: list[pd.Series] = []
    valid_target = target.dropna()
    for year in range(first_test_year, int(features.index.max().year) + 1):
        test_idx = features.index[features.index.year == year]
        if test_idx.empty:
            continue
        test_start_position = features.index.get_loc(test_idx.min())
        if test_start_position < embargo_days:
            continue
        train_cutoff = features.index[test_start_position - embargo_days]
        train_idx = features.index[features.index <= train_cutoff].intersection(valid_target.index)
        if len(train_idx) < 500:
            continue
        predictions.append(_ridge_predict(features.loc[train_idx], target.loc[train_idx], features.loc[test_idx], alpha))
    return pd.concat(predictions).sort_index()


def _ridge_predict(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, alpha: float) -> pd.Series:
    mean = x_train.mean()
    std = x_train.std().replace(0.0, 1.0)
    x = (x_train - mean) / std
    t = (x_test - mean) / std
    x_mat = np.c_[np.ones(len(x)), x.to_numpy(dtype=float)]
    t_mat = np.c_[np.ones(len(t)), t.to_numpy(dtype=float)]
    penalty = np.eye(x_mat.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_mat.T @ x_mat + alpha * penalty, x_mat.T @ y_train.to_numpy(dtype=float))
    return pd.Series(t_mat @ beta, index=x_test.index)


def _prediction_scale(predictions: pd.Series, amplitude: float) -> pd.Series:
    expanding_mean = predictions.expanding().mean().shift(1)
    expanding_std = predictions.expanding().std().shift(1)
    z_score = ((predictions - expanding_mean) / expanding_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + amplitude * np.tanh(z_score)).clip(lower=0.25, upper=1.75)


def _scale_returns(base: pd.Series, predictions: pd.Series, amplitude: float) -> pd.Series:
    scale = _prediction_scale(predictions, amplitude=amplitude)
    meta_turnover_cost = scale.diff().abs().fillna(0.0) * (META_COST_BPS / 10_000.0)
    return base.reindex(scale.index).fillna(0.0) * scale - meta_turnover_cost


def _summary_table(variants: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, returns in variants.items():
        summary = summarize_returns(returns).as_dict()
        summary["portfolio"] = name
        rows.append(summary)
    return pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False).reset_index(drop=True)


def _next_year_estimates(variants: dict[str, pd.Series], simulations: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows = []
    for name, returns in variants.items():
        recent = returns.loc["2018-01-01":].dropna().to_numpy(dtype=float)
        block = 21
        paths = np.empty(simulations)
        for idx in range(simulations):
            sampled: list[float] = []
            while len(sampled) < 252:
                start = int(rng.integers(0, max(1, len(recent) - block)))
                sampled.extend(recent[start : start + block])
            paths[idx] = float(np.prod(1.0 + np.asarray(sampled[:252])) - 1.0)
        rows.append(
            {
                "portfolio": name,
                "expected_return": float(np.mean(paths)),
                "median_return": float(np.median(paths)),
                "p05_return": float(np.quantile(paths, 0.05)),
                "p95_return": float(np.quantile(paths, 0.95)),
                "loss_probability": float((paths < 0.0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("expected_return", ascending=False).reset_index(drop=True)


def _write_report(summary: pd.DataFrame, holdout: pd.DataFrame, next_year: pd.DataFrame) -> None:
    best = summary[summary["portfolio"] == "ml_meta_scale_growth"].iloc[0]
    best_holdout = holdout[holdout["portfolio"] == "ml_meta_scale_growth"].iloc[0]
    forecast = next_year[next_year["portfolio"] == "ml_meta_scale_growth"].iloc[0]
    lines = [
        "# ML Meta Portfolio Results",
        "",
        "An expanding ridge model predicts 21-day forward return of the high-return asset-overlay portfolio from lagged SPY trend/volatility and portfolio state features.",
        "Predictions are converted to bounded exposure multipliers.  This is a simple auditable ML meta layer, not a black-box model.",
        "Training uses a 21-trading-day purge/embargo before each test year so forward-return labels cannot overlap the validation year.",
        "",
        "## Full-sample comparison",
        "",
        _markdown_table(summary),
        "",
        "## 2022-2026 validation window",
        "",
        _markdown_table(holdout),
        "",
        "## Estimated next-year distribution",
        "",
        _markdown_table(next_year),
        "",
        "## Preferred ML variant",
        "",
        "`ml_meta_scale_growth` is the best return/risk compromise.  The max variant has higher return but pushes drawdown beyond the already aggressive target.  Training uses a 21-trading-day embargo so forward-return labels do not overlap the test year.",
        "",
        f"- Annual return: {_pct(best['annual_return'])}.",
        f"- Annual std: {_pct(best['annual_std'])}.",
        f"- Sharpe: {best['sharpe']:.2f}.",
        f"- Max drawdown: {_pct(best['max_drawdown'])}.",
        f"- Validation annual return: {_pct(best_holdout['annual_return'])}.",
        f"- Validation Sharpe: {best_holdout['sharpe']:.2f}.",
        f"- Next-year expected return: {_pct(forecast['expected_return'])}.",
        f"- Next-year 5th/95th percentile: {_pct(forecast['p05_return'])} / {_pct(forecast['p95_return'])}.",
        f"- Estimated loss probability: {forecast['loss_probability']:.1%}.",
        "",
        "Caveat: this meta layer increases leverage when the model is positive.  It should be paper-traded and revalidated on survivorship-free data before live use.",
    ]
    (OUTPUT_DIR / "ML_META_PORTFOLIO_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "portfolio",
        "annual_return",
        "annual_std",
        "sharpe",
        "max_drawdown",
        "expected_return",
        "median_return",
        "p05_return",
        "p95_return",
        "loss_probability",
    ]
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in [
        "annual_return",
        "annual_std",
        "max_drawdown",
        "expected_return",
        "median_return",
        "p05_return",
        "p95_return",
        "loss_probability",
    ]:
        if column in display:
            display[column] = display[column].map(_pct)
    if "sharpe" in display:
        display["sharpe"] = display["sharpe"].map(lambda value: f"{float(value):.2f}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def _pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
