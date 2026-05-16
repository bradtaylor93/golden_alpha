"""Dynamic allocation from annual prediction ranks plus intra-year deviance.

The annual model produces a fixed prediction rank at the financial availability
date.  This script asks whether, during the following year, high-ranked stocks
that temporarily lag their high-rank cohort are better buys.

No realized forward return is used for sizing.  Daily target weights are based
only on:

- the original purged annual prediction rank,
- price movement observed from the signal date through the current close,
- recent trailing returns available at the current close.

Targets are applied with a one-day lag in the return calculation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research import download_yahoo_ohlcv
from ath_reversion_research.metrics import summarize_returns


OUTPUT_DIR = Path("ath_reversion_research/reports/prediction_deviation_portfolio")
PREDICTIONS = Path("ath_reversion_research/reports/sp500_annual_improvement/walk_forward_predictions.csv")
COST_BPS = 10.0


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["trade_date", "fwd_12m_return_end_date"])
    signals = predictions[
        (predictions["feature_set"] == "financial_plus_price")
        & (predictions["forward_window"] == "fwd_12m_return")
        & (predictions["regression_bucket"] == "large_cap")
    ].copy()
    symbols = sorted(signals["symbol"].dropna().astype(str).unique())
    prices = download_yahoo_ohlcv(symbols, start="2024-01-01", chunk_size=25)
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()

    targets = {
        "baseline_top_quintile_equal": _static_top_quantile(signals, close, top_frac=0.20),
        "deviation_pullback_equal": _deviation_targets(signals, close, top_frac=0.30, mode="pullback_equal"),
        "deviation_weighted": _deviation_targets(signals, close, top_frac=0.40, mode="deviation_weighted"),
        "prediction_x_deviation_weighted": _deviation_targets(signals, close, top_frac=0.40, mode="prediction_x_deviation"),
        "recent_dip_high_rank": _deviation_targets(signals, close, top_frac=0.30, mode="recent_dip"),
        "base_plus_deviation_overlay": _blend_targets(
            _static_top_quantile(signals, close, top_frac=0.20),
            _deviation_targets(signals, close, top_frac=0.40, mode="prediction_x_deviation"),
            base_weight=0.50,
        ),
    }

    returns = {}
    turnovers = {}
    for name, target in targets.items():
        returns[name], turnovers[name] = _portfolio_returns(close, target)
    returns_frame = pd.DataFrame(returns)
    turnover_frame = pd.DataFrame(turnovers)
    first_active = min((target.abs().sum(axis=1) > 0).idxmax() for target in targets.values())
    returns_for_metrics = returns_frame.loc[first_active:]
    turnover_for_metrics = turnover_frame.loc[first_active:]
    summary = _summary(returns_for_metrics, turnover_for_metrics)
    yearly = _yearly_summary(returns_for_metrics)

    returns_frame.to_csv(OUTPUT_DIR / "daily_returns.csv", index_label="date")
    turnover_frame.to_csv(OUTPUT_DIR / "daily_turnover.csv", index_label="date")
    summary.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False)
    yearly.to_csv(OUTPUT_DIR / "yearly_summary.csv", index=False)
    signals.to_csv(OUTPUT_DIR / "prediction_events_used.csv", index=False)
    _write_report(summary, yearly, signals, first_active)
    print(summary.to_string(index=False))
    return 0


def _static_top_quantile(signals: pd.DataFrame, close: pd.DataFrame, top_frac: float) -> pd.DataFrame:
    target = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for trade_date, group in signals.groupby("trade_date"):
        group = group.sort_values("prediction", ascending=False)
        selected = group.head(max(1, int(np.ceil(len(group) * top_frac))))
        weights = pd.Series(1.0 / len(selected), index=selected["symbol"].astype(str))
        _add_signal_weights(target, trade_date, group["fwd_12m_return_end_date"].max(), weights)
    return _normalize(target)


def _deviation_targets(signals: pd.DataFrame, close: pd.DataFrame, top_frac: float, mode: str) -> pd.DataFrame:
    target = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    returns_20d = close.pct_change(20)
    for trade_date, group in signals.groupby("trade_date"):
        group = group.sort_values("prediction", ascending=False).copy()
        n = max(1, int(np.ceil(len(group) * top_frac)))
        high_rank = group.head(n).copy()
        symbols = high_rank["symbol"].astype(str).tolist()
        start_pos = close.index.searchsorted(pd.Timestamp(trade_date), side="left")
        if start_pos >= len(close.index):
            continue
        end_pos = close.index.searchsorted(pd.Timestamp(group["fwd_12m_return_end_date"].max()), side="right")
        trade_close = high_rank.set_index("symbol")["trade_close"].astype(float)
        prediction_rank = high_rank.set_index("symbol")["prediction"].rank(pct=True).astype(float)
        for date in close.index[start_pos:end_pos]:
            current = close.loc[date, symbols].dropna()
            if current.empty:
                continue
            since_signal = current / trade_close.reindex(current.index) - 1.0
            cohort_relative = since_signal - since_signal.mean()
            deviance = (-cohort_relative).clip(lower=0.0)
            recent_dip = (-returns_20d.loc[date, current.index]).clip(lower=0.0).fillna(0.0)
            if mode == "pullback_equal":
                selected = deviance[deviance > deviance.quantile(0.50)]
                weights = pd.Series(1.0 / len(selected), index=selected.index) if len(selected) else pd.Series(dtype=float)
            elif mode == "deviation_weighted":
                weights = _positive_normalize(deviance)
            elif mode == "prediction_x_deviation":
                weights = _positive_normalize(deviance * prediction_rank.reindex(deviance.index).fillna(0.0))
            elif mode == "recent_dip":
                weights = _positive_normalize(recent_dip * prediction_rank.reindex(recent_dip.index).fillna(0.0))
            else:
                raise ValueError(f"unknown deviation mode: {mode}")
            if not weights.empty:
                target.loc[date, weights.index] += weights
    return _normalize(target)


def _blend_targets(base: pd.DataFrame, overlay: pd.DataFrame, base_weight: float) -> pd.DataFrame:
    blended = base_weight * base.add(0.0, fill_value=0.0) + (1.0 - base_weight) * overlay.add(0.0, fill_value=0.0)
    return _normalize(blended.fillna(0.0))


def _add_signal_weights(target: pd.DataFrame, trade_date: pd.Timestamp, end_date: pd.Timestamp, weights: pd.Series) -> None:
    start_pos = target.index.searchsorted(pd.Timestamp(trade_date), side="left")
    end_pos = target.index.searchsorted(pd.Timestamp(end_date), side="right")
    if start_pos >= end_pos or weights.empty:
        return
    active_index = target.index[start_pos:end_pos]
    valid_symbols = [symbol for symbol in weights.index if symbol in target.columns]
    if valid_symbols:
        target.loc[active_index, valid_symbols] = target.loc[active_index, valid_symbols].add(
            weights.loc[valid_symbols],
            axis=1,
        )


def _positive_normalize(values: pd.Series) -> pd.Series:
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values > 0.0]
    total = values.sum()
    if total <= 0.0:
        return pd.Series(dtype=float)
    return values / total


def _normalize(target: pd.DataFrame) -> pd.DataFrame:
    gross = target.abs().sum(axis=1).replace(0.0, np.nan)
    return target.div(gross, axis=0).fillna(0.0)


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
        for name in returns.columns:
            row = summarize_returns(group[name]).as_dict()
            row["year"] = int(year)
            row["portfolio"] = name
            rows.append(row)
    return pd.DataFrame(rows)


def _write_report(summary: pd.DataFrame, yearly: pd.DataFrame, signals: pd.DataFrame, first_active: pd.Timestamp) -> None:
    lines = [
        "# Prediction-Deviation Portfolio Backtest",
        "",
        "Tests whether high-ranked annual prediction stocks that lag during the holding year are better buys.",
        "",
        "## Leakage controls",
        "",
        "- Annual predictions are purged out-of-sample predictions from the S&P annual improvement study.",
        "- Intra-year deviance uses only price movement observed from signal date through current close.",
        "- Targets are executed with one-day lag.",
        "- Realized forward returns are not used for sizing.",
        "",
        "## Dataset",
        "",
        f"- Prediction events used: {len(signals)}.",
        f"- Symbols: {signals['symbol'].nunique()}.",
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
        "- The best deviance variant is a blend: keep half the static top-quintile basket and allocate half to high-ranked names lagging their cohort.",
        "- That blend did not increase raw return versus the static top-quintile basket, but it improved Sharpe, hit rate, volatility, and max drawdown.",
        "- Pure recent-dip buying performed poorly, so the annual prediction rank must remain the anchor.",
        "- The sample is short and still based on current S&P 500 membership.",
    ]
    (OUTPUT_DIR / "PREDICTION_DEVIATION_PORTFOLIO.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if col == "sharpe":
            display[col] = display[col].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "")
        elif pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "year"}:
            if any(token in col for token in ["return", "std", "drawdown", "rate", "turnover"]):
                display[col] = display[col].map(lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
