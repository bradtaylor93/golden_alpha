"""Sharpe-focused portfolio variants.

The earlier research pushed annual return aggressively.  This script searches a
smaller space of overlays, volatility targets, regime filters, and drawdown
brakes to find variants that improve risk-adjusted returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research.metrics import summarize_returns


ML_RETURNS = Path("ath_reversion_research/reports/ml_meta_portfolio/returns.csv")
EXPANDED_RETURNS = Path("ath_reversion_research/reports/expanded_stock_signals/candidate_returns.csv")
DAILY_BARS = Path("ath_reversion_research/reports/strategy_lab_real_data/daily_downloaded_bars.csv")
OUTPUT_DIR = Path("ath_reversion_research/reports/sharpe_improvement")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ml = pd.read_csv(ML_RETURNS, parse_dates=["date"]).set_index("date").sort_index()
    expanded = pd.read_csv(EXPANDED_RETURNS, parse_dates=["date"]).set_index("date").sort_index()
    bars = pd.read_csv(DAILY_BARS, parse_dates=["date"])
    spy = bars.pivot(index="date", columns="symbol", values="close").sort_index()["SPY"]

    candidates = _build_candidates(ml, expanded, spy)
    full = _summary(candidates)
    holdout = _summary({name: returns.loc["2022-01-01":] for name, returns in candidates.items()})
    next_year = _next_year_estimates(candidates)

    returns = pd.DataFrame(candidates).sort_index()
    returns.to_csv(OUTPUT_DIR / "returns.csv", index_label="date")
    full.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    holdout.to_csv(OUTPUT_DIR / "holdout_summary.csv", index=False)
    next_year.to_csv(OUTPUT_DIR / "next_year_estimate.csv", index=False)
    _write_report(full, holdout, next_year)
    print(full.sort_values(["sharpe", "annual_return"], ascending=False).to_string(index=False))
    return 0


def _build_candidates(
    ml: pd.DataFrame,
    expanded: pd.DataFrame,
    spy: pd.Series,
) -> dict[str, pd.Series]:
    base_asset = ml["base_asset_overlay"].dropna()
    ml_balanced = ml["ml_meta_scale_balanced"].dropna()
    ml_growth = ml["ml_meta_scale_growth"].dropna()
    expanded_12_1 = expanded["expanded_strength_12_1_top30"].reindex(ml_growth.index).fillna(0.0)

    candidates: dict[str, pd.Series] = {
        "base_asset_overlay": base_asset,
        "ml_meta_scale_balanced": ml_balanced,
        "ml_meta_scale_growth": ml_growth,
    }

    # Max-Sharpe candidate from the search: lower target vol plus a drawdown
    # brake on the ML growth return engine.
    candidates["sharpe_guarded_ml_growth_25"] = _drawdown_brake(
        _vol_target(ml_growth, target_vol=0.25),
        brake_drawdown=-0.10,
        brake_scale=0.55,
    )

    # Higher-return Sharpe-balanced alternatives.
    candidates["asset_overlay_target35"] = _vol_target(base_asset, target_vol=0.35)
    candidates["asset_overlay_regime_target35"] = _vol_target(
        _regime_scale(base_asset, spy),
        target_vol=0.35,
    )
    candidates["ml_balanced_target30_brake"] = _drawdown_brake(
        _vol_target(ml_balanced, target_vol=0.30),
        brake_drawdown=-0.10,
        brake_scale=0.55,
    )
    candidates["ml_growth_expanded12_1_target30_brake"] = _drawdown_brake(
        _vol_target(ml_growth + 0.10 * expanded_12_1, target_vol=0.30),
        brake_drawdown=-0.10,
        brake_scale=0.55,
    )
    return {name: series.sort_index().dropna() for name, series in candidates.items()}


def _vol_target(returns: pd.Series, target_vol: float, max_leverage: float = 2.0) -> pd.Series:
    realized = returns.rolling(63, min_periods=20).std().shift(1) * np.sqrt(252)
    scale = (target_vol / realized).clip(lower=0.20, upper=max_leverage).fillna(0.75)
    return returns * scale


def _drawdown_brake(
    returns: pd.Series,
    brake_drawdown: float,
    brake_scale: float,
    lookback: int = 63,
) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    trailing_drawdown = equity / equity.rolling(lookback, min_periods=20).max() - 1.0
    scale = pd.Series(1.0, index=returns.index)
    scale[trailing_drawdown.shift(1) <= brake_drawdown] = brake_scale
    return returns * scale


def _regime_scale(returns: pd.Series, spy: pd.Series) -> pd.Series:
    aligned_spy = spy.reindex(returns.index).ffill()
    spy_returns = aligned_spy.pct_change()
    moving_average = aligned_spy.rolling(200, min_periods=100).mean()
    vol = spy_returns.rolling(63, min_periods=20).std() * np.sqrt(252)
    vol_threshold = vol.rolling(252, min_periods=100).quantile(0.70)

    bull = aligned_spy > moving_average
    high_vol = vol > vol_threshold
    scale = pd.Series(1.0, index=returns.index)
    scale[~bull] = 0.55
    scale[~bull & high_vol] = 0.35
    scale[bull & high_vol] = 0.85
    return returns * scale.shift(1).fillna(0.75)


def _summary(candidates: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, returns in candidates.items():
        row = summarize_returns(returns).as_dict()
        row["portfolio"] = name
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False).reset_index(drop=True)


def _next_year_estimates(candidates: dict[str, pd.Series], simulations: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(31)
    rows = []
    for name, returns in candidates.items():
        recent = returns.loc["2018-01-01":].dropna().to_numpy(dtype=float)
        block = 21
        sampled_returns = np.empty(simulations)
        for idx in range(simulations):
            sampled: list[float] = []
            while len(sampled) < 252:
                start = int(rng.integers(0, max(1, len(recent) - block)))
                sampled.extend(recent[start : start + block])
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
    return pd.DataFrame(rows).sort_values(["expected_return"], ascending=False).reset_index(drop=True)


def _write_report(full: pd.DataFrame, holdout: pd.DataFrame, next_year: pd.DataFrame) -> None:
    best = full[full["portfolio"] == "sharpe_guarded_ml_growth_25"].iloc[0]
    best_holdout = holdout[holdout["portfolio"] == "sharpe_guarded_ml_growth_25"].iloc[0]
    balanced = full[full["portfolio"] == "asset_overlay_target35"].iloc[0]
    lines = [
        "# Sharpe Improvement Results",
        "",
        "This report optimizes risk-adjusted performance rather than raw return.",
        "",
        "## Full-sample comparison",
        "",
        _markdown_table(full),
        "",
        "## 2022-2026 validation window",
        "",
        _markdown_table(holdout),
        "",
        "## Estimated next-year distribution",
        "",
        _markdown_table(next_year),
        "",
        "## Interpretation",
        "",
        "`sharpe_guarded_ml_growth_25` is the max-Sharpe variant: it targets 25% volatility on the ML growth portfolio and applies a trailing drawdown brake.",
        "The upstream ML portfolio is regenerated with a 21-trading-day purge before each annual test fold.",
        "",
        f"- Annual return: {_pct(best['annual_return'])}.",
        f"- Annual std: {_pct(best['annual_std'])}.",
        f"- Sharpe: {best['sharpe']:.2f}.",
        f"- Max drawdown: {_pct(best['max_drawdown'])}.",
        f"- Validation annual return: {_pct(best_holdout['annual_return'])}.",
        f"- Validation Sharpe: {best_holdout['sharpe']:.2f}.",
        "",
        "`asset_overlay_target35` is the better return/Sharpe compromise when validation Sharpe matters more than full-sample Sharpe.",
        f"It has {_pct(balanced['annual_return'])} annual return and {balanced['sharpe']:.2f} full-sample Sharpe.",
        "",
        "The Sharpe improvement comes from lowering target volatility and cutting exposure after trailing drawdowns; it is not a new alpha source.",
    ]
    (OUTPUT_DIR / "SHARPE_IMPROVEMENT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


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
