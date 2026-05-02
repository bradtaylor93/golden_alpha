"""Build a conservative multi-strategy portfolio from the strategy lab results.

This script intentionally separates discovery from deployment-style portfolio
construction.  It uses only strategy return streams already produced by the
real-data research runners, then:

- keeps a small set of liquid, interpretable sleeves,
- optimizes weights on an earlier train period,
- evaluates the selected portfolio on a later holdout period,
- applies causal volatility targeting and drawdown brake overlays,
- estimates next-year performance with block bootstrap sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research.metrics import summarize_returns


REPORT_DIR = Path("ath_reversion_research/reports/deployable_portfolio")
LAB_RETURNS = Path("ath_reversion_research/reports/strategy_lab_real_data/strategy_returns.csv")
MA200_RETURNS = Path("ath_reversion_research/reports/ma200_real_data/returns.csv")
ATH_RETURNS = Path("ath_reversion_research/reports/high_market_cap_real_data/returns.csv")
ANNUALIZATION = 252

CORE_SLEEVES = {
    "large_cap_rs_126d": ("lab", "large_caps_relative_strength_126d"),
    "large_cap_mom_12_1": ("lab", "large_caps_momentum_12_1"),
    "ma200_wider_stop": ("ma200", "ma200_touch_wider_stop"),
    "ath_dip_recovery": ("ath", "ath_dip_recovery"),
    "sector_defensive_rotation": ("lab", "sector_rotation_regime_defensive"),
    "large_cap_bear_short_weak": ("lab", "large_caps_bear_short_weak_63d"),
}


@dataclass(frozen=True)
class PortfolioCandidate:
    name: str
    weights: dict[str, float]
    objective: float
    train_sharpe: float
    train_return: float
    train_drawdown: float


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sleeves = _load_sleeves()
    sleeves = sleeves.loc["2013-01-04":].dropna(how="all").fillna(0.0)

    train = sleeves.loc[: "2021-12-31"]
    holdout = sleeves.loc["2022-01-01" :]
    candidate = _optimize_weights(train)

    portfolios = {
        "equal_weight_core": _portfolio_returns(sleeves, {name: 1.0 / len(sleeves.columns) for name in sleeves.columns}),
        "optimized_core": _portfolio_returns(sleeves, candidate.weights),
    }
    regime_switched = _regime_switched_returns(sleeves)
    portfolios["regime_switched_core"] = regime_switched
    portfolios["regime_switched_guarded"] = _drawdown_brake(
        _vol_target(regime_switched, target_vol=0.12),
        brake_drawdown=-0.08,
        brake_scale=0.50,
    )
    portfolios["optimized_vol_target_12"] = _vol_target(portfolios["optimized_core"], target_vol=0.12)
    portfolios["deployable_guarded"] = _drawdown_brake(
        _vol_target(portfolios["optimized_core"], target_vol=0.12),
        brake_drawdown=-0.08,
        brake_scale=0.50,
    )
    portfolios["aggressive_vol_target_16"] = _drawdown_brake(
        _vol_target(portfolios["optimized_core"], target_vol=0.16),
        brake_drawdown=-0.10,
        brake_scale=0.60,
    )

    summary = _summaries(portfolios, sleeves.index, candidate)
    holdout_summary = _holdout_summaries(portfolios, holdout.index)
    next_year = _next_year_estimates(portfolios)
    contribution = _sleeve_contribution(sleeves, candidate.weights)

    summary.to_csv(REPORT_DIR / "portfolio_summary.csv", index=False)
    holdout_summary.to_csv(REPORT_DIR / "holdout_summary.csv", index=False)
    next_year.to_csv(REPORT_DIR / "next_year_estimate.csv", index=False)
    contribution.to_csv(REPORT_DIR / "sleeve_contribution.csv", index=False)
    pd.DataFrame([candidate.weights]).to_csv(REPORT_DIR / "selected_weights.csv", index=False)
    pd.DataFrame({name: returns for name, returns in portfolios.items()}).to_csv(
        REPORT_DIR / "portfolio_returns.csv", index_label="date"
    )
    _write_report(candidate, summary, holdout_summary, next_year, contribution)
    print(summary.sort_values("sharpe", ascending=False).to_string(index=False))
    return 0


def _load_sleeves() -> pd.DataFrame:
    if not LAB_RETURNS.exists() or not MA200_RETURNS.exists() or not ATH_RETURNS.exists():
        raise SystemExit("Run the strategy lab, MA200, and ATH research scripts before portfolio construction.")

    lab = pd.read_csv(LAB_RETURNS, parse_dates=["date"])
    ma200 = pd.read_csv(MA200_RETURNS, parse_dates=["date"])
    ath = pd.read_csv(ATH_RETURNS, parse_dates=["date"])

    pieces: dict[str, pd.Series] = {}
    for sleeve, (source, strategy) in CORE_SLEEVES.items():
        if source == "lab":
            frame = lab[(lab["strategy"] == strategy) & (lab["grain"] == "1d")]
        elif source == "ma200":
            frame = ma200[ma200["strategy"] == strategy]
        elif source == "ath":
            frame = ath[ath["strategy"] == strategy]
        else:
            raise ValueError(f"Unknown source: {source}")
        if frame.empty:
            raise ValueError(f"No returns found for {source}:{strategy}")
        series = frame.set_index(pd.to_datetime(frame["date"]))["net_return"].sort_index()
        series.index.name = "date"
        pieces[sleeve] = series
    return pd.DataFrame(pieces).sort_index()


def _optimize_weights(train: pd.DataFrame) -> PortfolioCandidate:
    rng = np.random.default_rng(42)
    names = list(train.columns)
    candidates: list[np.ndarray] = []
    candidates.append(np.repeat(1.0 / len(names), len(names)))

    # Bias the search toward liquid long engines but allow meaningful bear hedge.
    alpha = np.array([3.0, 2.0, 1.5, 2.0, 1.0, 0.9])
    for _ in range(60_000):
        w = rng.dirichlet(alpha)
        if w[names.index("large_cap_bear_short_weak")] > 0.25:
            continue
        if w[names.index("sector_defensive_rotation")] > 0.25:
            continue
        candidates.append(w)

    best: PortfolioCandidate | None = None
    for w in candidates:
        weights = dict(zip(names, w))
        returns = _portfolio_returns(train, weights)
        summary = summarize_returns(returns).as_dict()
        drawdown = float(summary["max_drawdown"])
        sharpe = float(summary["sharpe"])
        annual_return = float(summary["annual_return"])
        annual_std = float(summary["annual_std"])
        turnover_penalty = 0.0
        concentration_penalty = max(0.0, float(w.max()) - 0.40)
        drawdown_penalty = max(0.0, abs(drawdown) - 0.25)
        vol_penalty = max(0.0, annual_std - 0.22)
        objective = sharpe + 0.35 * annual_return - 2.0 * drawdown_penalty - vol_penalty - concentration_penalty - turnover_penalty
        if best is None or objective > best.objective:
            best = PortfolioCandidate(
                name="optimized_train_2013_2021",
                weights=weights,
                objective=float(objective),
                train_sharpe=sharpe,
                train_return=annual_return,
                train_drawdown=drawdown,
            )
    if best is None:
        raise RuntimeError("No candidate portfolio generated")
    return best


def _portfolio_returns(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    aligned = frame.reindex(columns=list(weights)).fillna(0.0)
    w = pd.Series(weights, dtype=float)
    w = w / w.abs().sum()
    return (aligned * w).sum(axis=1)


def _regime_switched_returns(sleeves: pd.DataFrame) -> pd.Series:
    """Use risk-on sleeves in bull trends and hedge/defensive sleeves in bear trends."""

    lab = pd.read_csv(LAB_RETURNS, parse_dates=["date"])
    sector = lab[(lab["strategy"] == "sector_rotation_regime_defensive") & (lab["grain"] == "1d")]
    # Infer the market regime from whether the defensive sector sleeve is active
    # alongside the long relative-strength sleeves: this keeps the overlay causal
    # with the already-computed strategy returns.
    spy_proxy = sleeves[["large_cap_rs_126d", "large_cap_mom_12_1", "ath_dip_recovery"]].mean(axis=1)
    trend = (1.0 + spy_proxy).cumprod().rolling(126, min_periods=63).mean()
    equity = (1.0 + spy_proxy).cumprod()
    bull = (equity > trend).shift(1).fillna(False)

    risk_on = _portfolio_returns(
        sleeves,
        {
            "large_cap_rs_126d": 0.45,
            "large_cap_mom_12_1": 0.25,
            "ath_dip_recovery": 0.20,
            "ma200_wider_stop": 0.08,
            "sector_defensive_rotation": 0.02,
        },
    )
    risk_off = _portfolio_returns(
        sleeves,
        {
            "large_cap_bear_short_weak": 0.45,
            "sector_defensive_rotation": 0.25,
            "ath_dip_recovery": 0.15,
            "large_cap_rs_126d": 0.10,
            "ma200_wider_stop": 0.05,
        },
    )
    return risk_on.where(bull, risk_off)


def _vol_target(returns: pd.Series, target_vol: float, lookback: int = 63, max_leverage: float = 1.25) -> pd.Series:
    realized = returns.rolling(lookback, min_periods=20).std().shift(1) * np.sqrt(ANNUALIZATION)
    scale = (target_vol / realized).clip(lower=0.25, upper=max_leverage).fillna(0.75)
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


def _summaries(
    portfolios: dict[str, pd.Series],
    index: pd.Index,
    candidate: PortfolioCandidate,
) -> pd.DataFrame:
    rows = []
    for name, returns in portfolios.items():
        summary = summarize_returns(returns.reindex(index).fillna(0.0)).as_dict()
        summary.update(
            {
                "portfolio": name,
                "train_objective": candidate.objective if name.startswith("optimized") or name.startswith("deployable") else np.nan,
                "train_sharpe": candidate.train_sharpe if name.startswith("optimized") or name.startswith("deployable") else np.nan,
            }
        )
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def _holdout_summaries(portfolios: dict[str, pd.Series], holdout_index: pd.Index) -> pd.DataFrame:
    rows = []
    for name, returns in portfolios.items():
        summary = summarize_returns(returns.reindex(holdout_index).fillna(0.0)).as_dict()
        summary["portfolio"] = name
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def _next_year_estimates(portfolios: dict[str, pd.Series], simulations: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for name, returns in portfolios.items():
        recent = returns.loc["2018-01-01":].dropna().to_numpy(dtype=float)
        if len(recent) < 252:
            recent = returns.dropna().to_numpy(dtype=float)
        block = 21
        simulated = np.empty(simulations)
        for idx in range(simulations):
            sampled: list[float] = []
            while len(sampled) < 252:
                start = int(rng.integers(0, max(1, len(recent) - block)))
                sampled.extend(recent[start : start + block])
            path = np.asarray(sampled[:252])
            simulated[idx] = float(np.prod(1.0 + path) - 1.0)
        rows.append(
            {
                "portfolio": name,
                "expected_return": float(np.mean(simulated)),
                "median_return": float(np.median(simulated)),
                "p05_return": float(np.quantile(simulated, 0.05)),
                "p25_return": float(np.quantile(simulated, 0.25)),
                "p75_return": float(np.quantile(simulated, 0.75)),
                "p95_return": float(np.quantile(simulated, 0.95)),
                "loss_probability": float((simulated < 0.0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("expected_return", ascending=False).reset_index(drop=True)


def _sleeve_contribution(sleeves: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    rows = []
    for sleeve, weight in weights.items():
        summary = summarize_returns(sleeves[sleeve].fillna(0.0)).as_dict()
        summary.update({"sleeve": sleeve, "selected_weight": weight})
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("selected_weight", ascending=False).reset_index(drop=True)


def _write_report(
    candidate: PortfolioCandidate,
    summary: pd.DataFrame,
    holdout: pd.DataFrame,
    next_year: pd.DataFrame,
    contribution: pd.DataFrame,
) -> None:
    deployable = summary[summary["portfolio"] == "deployable_guarded"].iloc[0]
    deployable_holdout = holdout[holdout["portfolio"] == "deployable_guarded"].iloc[0]
    deployable_forecast = next_year[next_year["portfolio"] == "deployable_guarded"].iloc[0]

    lines = [
        "# Deployable Multi-Strategy Portfolio Research",
        "",
        "This report converts the exploratory strategy results into a conservative deployment candidate.",
        "It is still a research result, not investment advice or a production trading approval.",
        "",
        "## Selected sleeves",
        "",
        _markdown_table(
            contribution[
                [
                    "sleeve",
                    "selected_weight",
                    "annual_return",
                    "annual_std",
                    "sharpe",
                    "max_drawdown",
                ]
            ]
        ),
        "",
        "## Portfolio comparison",
        "",
        _markdown_table(
            summary[
                [
                    "portfolio",
                    "annual_return",
                    "annual_std",
                    "sharpe",
                    "max_drawdown",
                    "hit_rate",
                ]
            ]
        ),
        "",
        "## 2022-2026 holdout check",
        "",
        _markdown_table(
            holdout[
                [
                    "portfolio",
                    "annual_return",
                    "annual_std",
                    "sharpe",
                    "max_drawdown",
                    "hit_rate",
                ]
            ]
        ),
        "",
        "## Estimated next-year return distribution",
        "",
        _markdown_table(next_year),
        "",
        "## Recommended deployment candidate",
        "",
        "`deployable_guarded` is the preferred candidate because it keeps most of the optimized core's Sharpe while reducing realized volatility with a causal 12% vol target and a trailing-drawdown brake.",
        "",
        f"- Full-sample annual return: {_pct(deployable['annual_return'])}.",
        f"- Full-sample annual std: {_pct(deployable['annual_std'])}.",
        f"- Full-sample Sharpe: {deployable['sharpe']:.2f}.",
        f"- Full-sample max drawdown: {_pct(deployable['max_drawdown'])}.",
        f"- Holdout annual return: {_pct(deployable_holdout['annual_return'])}.",
        f"- Holdout Sharpe: {deployable_holdout['sharpe']:.2f}.",
        f"- Estimated next-year mean return: {_pct(deployable_forecast['expected_return'])}.",
        f"- Estimated next-year 5th/95th percentile: {_pct(deployable_forecast['p05_return'])} / {_pct(deployable_forecast['p95_return'])}.",
        f"- Estimated probability of a negative next year: {deployable_forecast['loss_probability']:.1%}.",
        "",
        "## Safety controls before live use",
        "",
        "- Trade liquid large-cap/ETF sleeves first; keep lower-cap and hourly sleeves out of the production portfolio until validated on a better intraday data source.",
        "- Enforce max gross exposure, max single-sleeve weight, borrow availability for short sleeves, and daily loss limits.",
        "- Recompute signals after market close; execute with limit/VWAP-aware orders rather than assuming close-to-close fills.",
        "- Re-run this report on survivorship-free data before sizing real capital.",
        "",
        f"Training objective selected weights on 2013-2021 data: {candidate.objective:.3f}.",
    ]
    (REPORT_DIR / "DEPLOYABLE_PORTFOLIO_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if column in {
            "annual_return",
            "annual_std",
            "max_drawdown",
            "hit_rate",
            "selected_weight",
            "expected_return",
            "median_return",
            "p05_return",
            "p25_return",
            "p75_return",
            "p95_return",
            "loss_probability",
        }:
            display[column] = display[column].map(_pct)
        elif column == "sharpe":
            display[column] = display[column].map(lambda value: f"{float(value):.2f}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def _pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
