"""Portfolio backtest from purged S&P 500 annual prediction ranks.

The input predictions are already generated out-of-sample with purged training.
This script does not use realized forward returns for weighting.  It forms
target weights from prediction ranks on each signal date, executes with a
one-day lag, and holds each signal until its forecast end date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research import download_yahoo_ohlcv
from ath_reversion_research.metrics import summarize_returns


OUTPUT_DIR = Path("ath_reversion_research/reports/prediction_weighted_portfolio")
PREDICTIONS = Path("ath_reversion_research/reports/sp500_annual_improvement/walk_forward_predictions.csv")
COST_BPS = 10.0


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["trade_date", "fwd_12m_return_end_date"])
    universe = predictions[
        (predictions["feature_set"] == "financial_plus_price")
        & (predictions["forward_window"] == "fwd_12m_return")
        & (predictions["regression_bucket"] == "large_cap")
    ].copy()
    symbols = sorted(universe["symbol"].dropna().astype(str).unique())
    prices = download_yahoo_ohlcv(symbols, start="2024-01-01", chunk_size=25)
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()

    schemes = {
        "top_quintile_equal": _target_weights(universe, close.index, top_frac=0.20, scheme="equal", side="long"),
        "top_quintile_rank_weight": _target_weights(universe, close.index, top_frac=0.20, scheme="rank", side="long"),
        "top_decile_equal": _target_weights(universe, close.index, top_frac=0.10, scheme="equal", side="long"),
        "top_decile_rank_weight": _target_weights(universe, close.index, top_frac=0.10, scheme="rank", side="long"),
        "top_bottom_quintile_long_short": _target_weights(universe, close.index, top_frac=0.20, scheme="rank", side="long_short"),
    }
    returns = {}
    turnovers = {}
    active_masks = {}
    for name, targets in schemes.items():
        returns[name], turnovers[name] = _portfolio_returns(close, targets)
        active_masks[name] = targets.abs().sum(axis=1) > 0.0
    returns_frame = pd.DataFrame(returns)
    turnover_frame = pd.DataFrame(turnovers)
    first_active = min(mask[mask].index.min() for mask in active_masks.values() if mask.any())
    returns_for_metrics = returns_frame.loc[first_active:]
    turnover_for_metrics = turnover_frame.loc[first_active:]
    summary = _summary(returns_for_metrics, turnover_for_metrics)
    yearly = _yearly_summary(returns_for_metrics)

    returns_frame.to_csv(OUTPUT_DIR / "daily_returns.csv", index_label="date")
    turnover_frame.to_csv(OUTPUT_DIR / "daily_turnover.csv", index_label="date")
    summary.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False)
    yearly.to_csv(OUTPUT_DIR / "yearly_summary.csv", index=False)
    universe.to_csv(OUTPUT_DIR / "prediction_events_used.csv", index=False)
    _write_report(summary, yearly, universe, first_active)
    print(summary.to_string(index=False))
    return 0


def _target_weights(
    predictions: pd.DataFrame,
    trading_index: pd.Index,
    top_frac: float,
    scheme: str,
    side: str,
) -> pd.DataFrame:
    symbols = sorted(predictions["symbol"].dropna().astype(str).unique())
    targets = pd.DataFrame(0.0, index=trading_index, columns=symbols)
    for trade_date, group in predictions.groupby("trade_date"):
        group = group.dropna(subset=["prediction"]).sort_values("prediction", ascending=False).copy()
        if group.empty:
            continue
        n_select = max(1, int(np.ceil(len(group) * top_frac)))
        signal = pd.Series(0.0, index=symbols)
        top = group.head(n_select)
        if scheme == "equal":
            top_weights = pd.Series(1.0 / len(top), index=top["symbol"].astype(str))
        else:
            ranks = np.arange(len(top), 0, -1, dtype=float)
            top_weights = pd.Series(ranks / ranks.sum(), index=top["symbol"].astype(str))
        signal.loc[top_weights.index] = top_weights

        if side == "long_short":
            bottom = group.tail(n_select)
            ranks = np.arange(1, len(bottom) + 1, dtype=float)
            short_weights = pd.Series(ranks / ranks.sum(), index=bottom["symbol"].astype(str))
            signal.loc[short_weights.index] -= short_weights
            signal *= 0.5

        start_pos = trading_index.searchsorted(pd.Timestamp(trade_date), side="left")
        if start_pos >= len(trading_index):
            continue
        end_date = pd.Timestamp(group["fwd_12m_return_end_date"].dropna().max())
        end_pos = trading_index.searchsorted(end_date, side="right")
        targets.iloc[start_pos:end_pos] = targets.iloc[start_pos:end_pos].add(signal, axis=1)
    gross = targets.abs().sum(axis=1).replace(0.0, np.nan)
    return targets.div(gross, axis=0).fillna(0.0)


def _portfolio_returns(close: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    returns = close.pct_change().reindex_like(targets).fillna(0.0)
    executed = targets.shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    net = (executed * returns).sum(axis=1) - turnover * (COST_BPS / 10_000.0)
    return net, turnover


def _summary(returns: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in returns:
        row = summarize_returns(returns[name], turnover[name]).as_dict()
        row["portfolio"] = name
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False).reset_index(drop=True)


def _yearly_summary(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in returns.groupby(returns.index.year):
        if group.empty:
            continue
        for name in returns.columns:
            row = summarize_returns(group[name]).as_dict()
            row["year"] = int(year)
            row["portfolio"] = name
            rows.append(row)
    return pd.DataFrame(rows)


def _write_report(summary: pd.DataFrame, yearly: pd.DataFrame, predictions: pd.DataFrame, first_active: pd.Timestamp) -> None:
    lines = [
        "# Prediction-Weighted Portfolio Backtest",
        "",
        "Uses purged out-of-sample large-cap 12m financial+price predictions to form portfolios.",
        "",
        "## Leakage controls",
        "",
        "- Predictions are from the purged S&P annual improvement study.",
        "- Weights are based only on prediction ranks at each trade date.",
        "- Daily portfolio returns use one-day-lagged target weights.",
        "- Realized forward returns are not used for sizing.",
        "",
        "## Dataset",
        "",
        f"- Prediction events used: {len(predictions)}.",
        f"- Symbols: {predictions['symbol'].nunique()}.",
        f"- Trade dates: {predictions['trade_date'].min()} to {predictions['trade_date'].max()}.",
        f"- Metrics start after first active portfolio date: {first_active.date()}.",
        "",
        "## Portfolio summary",
        "",
        _markdown_table(summary),
        "",
        "## Yearly summary",
        "",
        _markdown_table(yearly),
        "",
        "## Interpretation",
        "",
        "- This is the more realistic deployment translation of the rank signal: it tests capital-weighted daily P&L, not just event forward returns.",
        "- The sample is still short and based on current S&P 500 membership, so survivorship bias remains.",
    ]
    (OUTPUT_DIR / "PREDICTION_WEIGHTED_PORTFOLIO.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "year"}:
            if any(token in col for token in ["return", "std", "sharpe", "drawdown", "rate", "turnover"]):
                display[col] = display[col].map(lambda v: f"{float(v) * 100:.2f}%" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
