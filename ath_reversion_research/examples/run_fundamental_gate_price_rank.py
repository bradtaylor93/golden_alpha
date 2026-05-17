"""Fundamental gates on top of the price-context ranker.

This tests whether fundamentals add value more robustly as filters/gates rather
than as continuous predictors.  The base rank score is the Dolt-aligned
large-cap 12m price-only model, and the gates use only annual financials
available at the prediction date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research.metrics import summarize_returns
import run_dolthub_aligned_replication as aligned


OUTPUT_DIR = Path("ath_reversion_research/reports/fundamental_gate_price_rank")
PREDICTIONS = Path("ath_reversion_research/reports/dolthub_aligned_replication/walk_forward_predictions.csv")
PRICE_CACHE = Path("ath_reversion_research/reports/dolthub_full_fundamental_validation/price_subset_yahoo.csv")


GATES = {
    "price_only_ungated": lambda f: pd.Series(True, index=f.index),
    "growth_positive": lambda f: f["revenue_growth_yoy"] > 0.0,
    "growth_and_profit": lambda f: (f["revenue_growth_yoy"] > 0.0) & (f["operating_margin"] > 0.0),
    "growth_profit_fcf": lambda f: (f["revenue_growth_yoy"] > 0.0) & (f["operating_margin"] > 0.0) & (f["fcf_margin"] > 0.0),
    "growth30_profit_fcf": lambda f: (f["revenue_growth_yoy"] >= 0.30) & (f["operating_margin"] > 0.0) & (f["fcf_margin"] > 0.0),
    "growth_profit_low_debt": lambda f: (
        (f["revenue_growth_yoy"] > 0.0)
        & (f["operating_margin"] > 0.0)
        & (f["debt_to_assets"] <= f.groupby("trade_date")["debt_to_assets"].transform(lambda s: s.quantile(0.75)))
    ),
    "growth_profit_fcf_low_debt": lambda f: (
        (f["revenue_growth_yoy"] > 0.0)
        & (f["operating_margin"] > 0.0)
        & (f["fcf_margin"] > 0.0)
        & (f["debt_to_assets"] <= f.groupby("trade_date")["debt_to_assets"].transform(lambda s: s.quantile(0.75)))
    ),
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["trade_date", "fwd_12m_return_end_date"])
    base = predictions[
        (predictions["feature_set"] == "price_only")
        & (predictions["forward_window"] == "fwd_12m_return")
        & (predictions["regression_bucket"] == "large_cap")
    ].copy()
    if base.empty:
        raise SystemExit("Run dolthub_aligned_replication first; missing price_only large-cap 12m predictions.")

    event_summary = _event_summary(base)
    portfolio_summary, daily_returns = _portfolio_tests(base)
    event_summary.to_csv(OUTPUT_DIR / "event_summary.csv", index=False)
    portfolio_summary.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False)
    daily_returns.to_csv(OUTPUT_DIR / "daily_returns.csv", index_label="date")
    _write_report(event_summary, portfolio_summary)
    print(event_summary.to_string(index=False))
    print("\nPortfolio")
    print(portfolio_summary.to_string(index=False))
    return 0


def _select_by_gate(frame: pd.DataFrame, gate_name: str, top_frac: float = 0.20) -> pd.DataFrame:
    pieces = []
    for trade_date, group in frame.groupby("trade_date"):
        eligible = group[GATES[gate_name](group)].copy()
        if eligible.empty:
            # If a strict gate produces no names, hold cash for that batch.
            continue
        selected = eligible.sort_values("prediction", ascending=False).head(max(1, int(np.ceil(len(group) * top_frac))))
        selected["gate_name"] = gate_name
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=list(frame.columns) + ["gate_name"])


def _event_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gate_name in GATES:
        selected = _select_by_gate(frame, gate_name)
        for top_col, label in [(selected, "selected")]:
            if top_col.empty:
                continue
            rows.append(
                {
                    "gate": gate_name,
                    "events": int(len(top_col)),
                    "avg_names_per_trade_date": float(top_col.groupby("trade_date").size().mean()),
                    "mean_return": float(top_col["target_return"].mean()),
                    "median_return": float(top_col["target_return"].median()),
                    "hit_rate": float((top_col["target_return"] > 0.0).mean()),
                    "avg_revenue_growth": float(top_col["revenue_growth_yoy"].mean()),
                    "avg_operating_margin": float(top_col["operating_margin"].mean()),
                    "avg_fcf_margin": float(top_col["fcf_margin"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["mean_return", "hit_rate"], ascending=False)


def _portfolio_tests(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PRICE_CACHE.exists():
        return pd.DataFrame(), pd.DataFrame()
    prices = pd.read_csv(PRICE_CACHE, parse_dates=["date"])
    symbols = sorted(frame["symbol"].dropna().astype(str).unique())
    close = prices[prices["symbol"].isin(symbols)].pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    returns = {}
    turnovers = {}
    for gate_name in GATES:
        weights = _target_weights(frame, close.index, close.columns, gate_name)
        net, turnover = _portfolio_returns(close, weights)
        active = weights.abs().sum(axis=1) > 0
        if not active.any():
            continue
        start = active[active].index.min()
        returns[gate_name] = net.loc[start:]
        turnovers[gate_name] = turnover.loc[start:]
    returns_frame = pd.DataFrame(returns)
    turnover_frame = pd.DataFrame(turnovers)
    rows = []
    for gate_name in returns_frame:
        row = summarize_returns(returns_frame[gate_name], turnover_frame[gate_name]).as_dict()
        row["gate"] = gate_name
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False), returns_frame


def _target_weights(frame: pd.DataFrame, trading_index: pd.Index, columns: pd.Index, gate_name: str) -> pd.DataFrame:
    target = pd.DataFrame(0.0, index=trading_index, columns=columns)
    selected = _select_by_gate(frame, gate_name)
    for trade_date, group in selected.groupby("trade_date"):
        valid = [s for s in group["symbol"].astype(str) if s in target.columns]
        if not valid:
            continue
        weights = pd.Series(1.0 / len(valid), index=valid)
        start = trading_index.searchsorted(pd.Timestamp(trade_date), side="left")
        end = trading_index.searchsorted(pd.Timestamp(group["fwd_12m_return_end_date"].max()), side="right")
        target.loc[trading_index[start:end], valid] = target.loc[trading_index[start:end], valid].add(weights, axis=1)
    gross = target.abs().sum(axis=1).replace(0.0, np.nan)
    return target.div(gross, axis=0).fillna(0.0)


def _portfolio_returns(close: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    returns = close.pct_change().reindex_like(weights).fillna(0.0)
    executed = weights.shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    return (executed * returns).sum(axis=1) - turnover * 0.001, turnover


def _write_report(event_summary: pd.DataFrame, portfolio_summary: pd.DataFrame) -> None:
    lines = [
        "# Fundamental Gates on Price Rank",
        "",
        "Tests using fundamentals as gates on top of the Dolt price-only large-cap 12m ranker.",
        "",
        "## Event-level results",
        "",
        _markdown_table(event_summary),
        "",
        "## Portfolio results",
        "",
        _markdown_table(portfolio_summary),
        "",
        "## Interpretation",
        "",
        "- If gates improve event returns or portfolio metrics, fundamentals are useful as filters even when not useful as continuous model predictors.",
        "- Strict high-growth gates can reduce breadth; portfolio results matter more than event means.",
    ]
    (OUTPUT_DIR / "FUNDAMENTAL_GATE_PRICE_RANK.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if col == "sharpe":
            display[col] = display[col].map(lambda v: f"{float(v):.2f}" if pd.notna(v) else "")
        elif pd.api.types.is_numeric_dtype(display[col]) and col not in {"events", "observations"}:
            if any(token in col for token in ["return", "rate", "growth", "margin", "std", "drawdown", "turnover"]):
                display[col] = display[col].map(lambda v: f"{float(v) * 100:.2f}%" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
