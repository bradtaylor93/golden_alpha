"""Intra-year laggard overlay for high-ranked annual stocks.

Hypothesis: if the annual model ranks a stock highly, negative intra-year
relative performance may be a buying opportunity rather than a reason to sell.

This script starts from the best annual portfolio construction found so far:

* 50% large-cap sleeve using the prior-best valuation model;
* 50% mid-cap sleeve using the sector-aware residual model;
* top 20% selected inside each sleeve.

It then tests monthly overlays that overweight high-ranked eligible names whose
1m/3m returns are weak relative to their eligible sleeve peers, provided 12m
trend is positive and trailing volatility is not extreme. Targets trade with a
one-day lag and transaction costs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research.metrics import summarize_returns

import run_historical_market_cap_valuation as hist
import run_portfolio_construction_improvement as construction


OUTPUT_DIR = Path("ath_reversion_research/reports/bucket_laggard_overlay")
PRICE_FILE = Path("ath_reversion_research/reports/dolthub_full_fundamental_validation/price_subset_yahoo.csv")
COST_BPS = 10.0
TOP_FRAC = 0.20
SLEEVE_WEIGHT = 0.50


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred = construction._build_prediction_set(construction.peer._load_events())
    sleeve_signals = _select_bucket_sleeve_signals(pred)
    close = _load_close(sorted(sleeve_signals["symbol"].astype(str).unique()))

    baseline = _static_bucket_targets(sleeve_signals, close)
    laggard_filtered = _monthly_laggard_targets(
        sleeve_signals,
        close,
        select_frac=0.50,
        trend_filter=True,
        max_vol_quantile=0.75,
        score_power=1.0,
    )
    laggard_aggressive = _monthly_laggard_targets(
        sleeve_signals,
        close,
        select_frac=0.35,
        trend_filter=True,
        max_vol_quantile=0.85,
        score_power=1.0,
    )
    laggard_soft = _monthly_laggard_targets(
        sleeve_signals,
        close,
        select_frac=0.60,
        trend_filter=True,
        max_vol_quantile=0.75,
        score_power=0.5,
    )
    laggard_no_trend = _monthly_laggard_targets(
        sleeve_signals,
        close,
        select_frac=0.50,
        trend_filter=False,
        max_vol_quantile=0.75,
        score_power=1.0,
    )
    moderate_stabilized = _monthly_laggard_targets(
        sleeve_signals,
        close,
        select_frac=0.40,
        trend_filter=True,
        max_vol_quantile=0.70,
        score_power=0.5,
        moderate_only=True,
        require_5d_recovery=True,
    )

    targets = {
        "bucket_static_equal": baseline,
        "laggard_overlay_only": laggard_filtered,
        "laggard_blend_95_05": _blend_targets(baseline, laggard_filtered, 0.95),
        "laggard_blend_90_10": _blend_targets(baseline, laggard_filtered, 0.90),
        "laggard_blend_80_20": _blend_targets(baseline, laggard_filtered, 0.80),
        "laggard_blend_70_30": _blend_targets(baseline, laggard_filtered, 0.70),
        "laggard_blend_50_50": _blend_targets(baseline, laggard_filtered, 0.50),
        "moderate_recovery_blend_95_05": _blend_targets(baseline, moderate_stabilized, 0.95),
        "moderate_recovery_blend_90_10": _blend_targets(baseline, moderate_stabilized, 0.90),
        "moderate_recovery_blend_80_20": _blend_targets(baseline, moderate_stabilized, 0.80),
        "laggard_aggressive_blend_70_30": _blend_targets(baseline, laggard_aggressive, 0.70),
        "laggard_soft_blend_70_30": _blend_targets(baseline, laggard_soft, 0.70),
        "laggard_no_trend_blend_70_30": _blend_targets(baseline, laggard_no_trend, 0.70),
    }

    returns = {}
    turnovers = {}
    for name, target in targets.items():
        returns[name], turnovers[name] = _portfolio_returns(close, target)
    returns_frame = pd.DataFrame(returns)
    turnover_frame = pd.DataFrame(turnovers)
    first_active = min((target.abs().sum(axis=1) > 0).idxmax() for target in targets.values())
    returns_frame = returns_frame.loc[first_active:]
    turnover_frame = turnover_frame.loc[first_active:]

    summary = _summary(returns_frame, turnover_frame)
    yearly = _yearly_summary(returns_frame)
    summary.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False)
    yearly.to_csv(OUTPUT_DIR / "yearly_summary.csv", index=False)
    _write_report(summary, yearly, sleeve_signals, first_active)
    print(summary.to_string(index=False))
    return 0


def _select_bucket_sleeve_signals(pred: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("large_best_known", "large_cap", "large_cap_prior_best", SLEEVE_WEIGHT),
        ("mid_sector_aware_residuals", "mid_cap", "mid_cap_sector_residual", SLEEVE_WEIGHT),
    ]
    rows = []
    for experiment, bucket, sleeve, sleeve_weight in specs:
        subset = pred[(pred["experiment"] == experiment) & (pred["regression_bucket"] == bucket)].copy()
        for year, group in subset.groupby("test_year"):
            selected = _top_prediction_group(group, TOP_FRAC).copy()
            selected["sleeve"] = sleeve
            selected["sleeve_weight"] = sleeve_weight
            selected["rank_pct_in_sleeve"] = selected["prediction"].rank(pct=True)
            rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def _top_prediction_group(group: pd.DataFrame, top_frac: float) -> pd.DataFrame:
    group = group.dropna(subset=["prediction"]).sort_values("prediction", ascending=False)
    group = group.drop_duplicates("symbol", keep="first")
    n = max(1, int(np.ceil(len(group) * top_frac)))
    return group.head(n)


def _load_close(symbols: list[str]) -> pd.DataFrame:
    prices = pd.read_csv(PRICE_FILE, usecols=["date", "symbol", "close"], parse_dates=["date"])
    prices = prices[prices["symbol"].isin(symbols)]
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    return close.loc["2015-01-01":].ffill(limit=5)


def _static_bucket_targets(signals: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    target = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for (_year, sleeve), group in signals.groupby(["test_year", "sleeve"]):
        symbols = [symbol for symbol in group["symbol"].astype(str) if symbol in close.columns]
        if not symbols:
            continue
        per_name = float(group["sleeve_weight"].iloc[0]) / len(symbols)
        weights = pd.Series(per_name, index=symbols)
        _add_row_active_weights(target, group, weights)
    return _normalize(target)


def _monthly_laggard_targets(
    signals: pd.DataFrame,
    close: pd.DataFrame,
    select_frac: float,
    trend_filter: bool,
    max_vol_quantile: float,
    score_power: float,
    moderate_only: bool = False,
    require_5d_recovery: bool = False,
) -> pd.DataFrame:
    target = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    returns_21 = close.pct_change(21)
    returns_63 = close.pct_change(63)
    returns_252 = close.pct_change(252)
    returns_5 = close.pct_change(5)
    vol_63 = close.pct_change().rolling(63, min_periods=30).std() * np.sqrt(252)

    for (_year, sleeve), group in signals.groupby(["test_year", "sleeve"]):
        group = group.copy()
        active_start = pd.Timestamp(group["trade_date"].min())
        active_end = pd.Timestamp(group["fwd_12m_return_end_date"].max())
        active_dates = close.index[(close.index >= active_start) & (close.index <= active_end)]
        if active_dates.empty:
            continue
        rebalance_dates = _month_starts(active_dates)
        current_weights = pd.Series(dtype=float)
        for date in active_dates:
            if date in rebalance_dates:
                active = group[
                    (pd.to_datetime(group["trade_date"]) <= date)
                    & (pd.to_datetime(group["fwd_12m_return_end_date"]) >= date)
                ].copy()
                symbols = [symbol for symbol in active["symbol"].astype(str) if symbol in close.columns]
                if not symbols:
                    current_weights = pd.Series(dtype=float)
                    continue
                r1 = returns_21.loc[date, symbols]
                r3 = returns_63.loc[date, symbols]
                r12 = returns_252.loc[date, symbols]
                vol = vol_63.loc[date, symbols]
                weak_score = -(0.50 * (r1 - r1.mean()) + 0.50 * (r3 - r3.mean()))
                rank = active.set_index("symbol")["rank_pct_in_sleeve"].reindex(symbols).astype(float)
                score = weak_score.clip(lower=0.0) * rank.fillna(rank.median())
                if moderate_only:
                    lower = weak_score.quantile(0.50)
                    upper = weak_score.quantile(0.85)
                    score = score.where((weak_score >= lower) & (weak_score <= upper), 0.0)
                if require_5d_recovery:
                    r5 = returns_5.loc[date, symbols]
                    score = score.where(r5 > 0.0, 0.0)
                if score_power != 1.0:
                    score = score.clip(lower=0.0) ** score_power
                if trend_filter:
                    score = score.where(r12 > 0.0, 0.0)
                vol_threshold = vol.quantile(max_vol_quantile)
                score = score.where(vol <= vol_threshold, 0.0)
                score = score.replace([np.inf, -np.inf], np.nan).dropna()
                score = score[score > 0.0].sort_values(ascending=False)
                n_select = max(1, int(np.ceil(len(symbols) * select_frac)))
                selected = score.head(n_select)
                current_weights = _positive_normalize(selected) * float(group["sleeve_weight"].iloc[0])
            if not current_weights.empty:
                target.loc[date, current_weights.index] += current_weights
    return _normalize(target)


def _month_starts(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    periods = index.to_period("M")
    flags = np.r_[True, periods[1:].values != periods[:-1].values]
    return set(index[flags])


def _add_row_active_weights(target: pd.DataFrame, group: pd.DataFrame, weights: pd.Series) -> None:
    for _, row in group.iterrows():
        symbol = str(row["symbol"])
        if symbol not in weights.index or symbol not in target.columns:
            continue
        active = target.index[
            (target.index >= pd.Timestamp(row["trade_date"]))
            & (target.index <= pd.Timestamp(row["fwd_12m_return_end_date"]))
        ]
        target.loc[active, symbol] = target.loc[active, symbol].to_numpy() + float(weights.loc[symbol])


def _blend_targets(base: pd.DataFrame, overlay: pd.DataFrame, base_weight: float) -> pd.DataFrame:
    out = base_weight * base.add(0.0, fill_value=0.0) + (1.0 - base_weight) * overlay.add(0.0, fill_value=0.0)
    return _normalize(out.fillna(0.0))


def _portfolio_returns(close: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    returns = close.pct_change().reindex_like(target).fillna(0.0)
    executed = target.shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    net = (executed * returns).sum(axis=1) - turnover * (COST_BPS / 10_000.0)
    return net, turnover


def _normalize(target: pd.DataFrame) -> pd.DataFrame:
    gross = target.abs().sum(axis=1).replace(0.0, np.nan)
    return target.div(gross, axis=0).fillna(0.0)


def _positive_normalize(values: pd.Series) -> pd.Series:
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values > 0.0]
    total = values.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    return values / total


def _summary(returns: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in returns.columns:
        row = summarize_returns(returns[name], turnover[name]).as_dict()
        row["portfolio"] = name
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False)


def _yearly_summary(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in returns.groupby(returns.index.year):
        for name in returns.columns:
            row = summarize_returns(group[name]).as_dict()
            row["year"] = int(year)
            row["portfolio"] = name
            rows.append(row)
    return pd.DataFrame(rows)


def _write_report(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    signals: pd.DataFrame,
    first_active: pd.Timestamp,
) -> None:
    lines = [
        "# Bucket sleeve intra-year laggard overlay",
        "",
        "Tests whether high-ranked annual-model stocks that underperform inside the year are better buys.",
        "",
        "## No-leakage setup",
        "",
        "- Annual predictions are the same purged walk-forward predictions used in the latest bucket-sleeve portfolio.",
        "- Intra-year overlay uses only trailing 1m/3m/12m returns and trailing 63-day volatility observed at rebalance.",
        "- Targets are rebalanced monthly and executed with a one-day lag.",
        "- Realized forward returns are not used for selection or sizing.",
        "",
        "## Baseline",
        "",
        "- 50% large-cap prior-best model, top 20%.",
        "- 50% mid-cap sector-aware residual model, top 20%.",
        "- Equal-weight within each sleeve.",
        "",
        "## Dataset",
        "",
        f"- Active signals: {len(signals)}.",
        f"- Symbols: {signals['symbol'].nunique()}.",
        f"- Metrics start: {first_active.date()}.",
        "",
        "## Summary",
        "",
        hist._markdown_table(summary),
        "",
        "## Yearly summary",
        "",
        hist._markdown_table(yearly),
        "",
        "## Interpretation",
        "",
        _interpretation(summary),
    ]
    (OUTPUT_DIR / "BUCKET_LAGGARD_OVERLAY.md").write_text("\n".join(lines), encoding="utf-8")


def _interpretation(summary: pd.DataFrame) -> str:
    baseline = summary[summary["portfolio"] == "bucket_static_equal"].iloc[0]
    best = summary.sort_values("sharpe", ascending=False).iloc[0]
    return "\n".join(
        [
            f"- Best Sharpe variant: `{best['portfolio']}` at {best['sharpe']:.2f}.",
            f"- Static bucket baseline Sharpe: {baseline['sharpe']:.2f}.",
            f"- Sharpe delta vs baseline: {best['sharpe'] - baseline['sharpe']:+.2f}.",
            "- If laggard overlays underperform, it means the ranker identifies a good group, but intra-year negative volatility is not reliably a buying opportunity after costs and trend/vol filters.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
