"""Ablation suite for potential fundamental ranking upgrades.

Tests proposed upgrades against the current S&P annual model:

- raw return vs SPY/sector-excess targets,
- raw return regression vs rank target vs top-quintile target,
- valuation-style features,
- sector/macro features,
- combined best candidates.

All models use purged walk-forward training by test year.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research import download_yahoo_ohlcv
import run_sp500_annual_improvement as base


OUTPUT_DIR = Path("ath_reversion_research/reports/fundamental_upgrade_ablation")
EVENTS = Path("ath_reversion_research/reports/sp500_sector_macro_improvement/enhanced_sector_macro_events.csv")

SECTOR_ETF = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

BASE_FEATURES = base.FEATURE_SETS["financial_plus_price"]
VALUATION_FEATURES = [
    "sales_to_market_cap",
    "earnings_yield",
    "fcf_yield",
    "gross_profit_to_market_cap",
    "growth_to_sales_multiple",
]
SECTOR_MACRO_FEATURES = [
    "sector_rel_revenue_growth_yoy",
    "sector_rel_operating_margin",
    "sector_rel_fcf_margin",
    "sector_rel_trailing_6m",
    "sector_median_revenue_growth_yoy",
    "sector_median_operating_margin",
    "sector_median_fcf_margin",
    "macro_spy_6m",
    "macro_spy_12m",
    "macro_spy_vol_3m",
    "macro_qqq_vs_spy_6m",
    "macro_iwm_vs_spy_6m",
    "macro_tlt_6m",
    "macro_hyg_lqd_6m_spread",
]


EXPERIMENTS = [
    ("baseline_raw_return", "raw_return", BASE_FEATURES),
    ("valuation_raw_return", "raw_return", BASE_FEATURES + VALUATION_FEATURES),
    ("sector_macro_raw_return", "raw_return", BASE_FEATURES + SECTOR_MACRO_FEATURES),
    ("combined_raw_return", "raw_return", BASE_FEATURES + VALUATION_FEATURES + SECTOR_MACRO_FEATURES),
    ("baseline_rank_target", "rank_target", BASE_FEATURES),
    ("valuation_rank_target", "rank_target", BASE_FEATURES + VALUATION_FEATURES),
    ("combined_rank_target", "rank_target", BASE_FEATURES + VALUATION_FEATURES + SECTOR_MACRO_FEATURES),
    ("baseline_top_quintile_target", "top_quintile_target", BASE_FEATURES),
    ("valuation_top_quintile_target", "top_quintile_target", BASE_FEATURES + VALUATION_FEATURES),
    ("combined_top_quintile_target", "top_quintile_target", BASE_FEATURES + VALUATION_FEATURES + SECTOR_MACRO_FEATURES),
    ("spy_excess_raw_return", "spy_excess_return", BASE_FEATURES + VALUATION_FEATURES),
    ("sector_excess_raw_return", "sector_excess_return", BASE_FEATURES + VALUATION_FEATURES),
    ("sector_excess_rank_target", "sector_excess_rank_target", BASE_FEATURES + VALUATION_FEATURES + SECTOR_MACRO_FEATURES),
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS, parse_dates=["trade_date", "fwd_3m_return_end_date", "fwd_6m_return_end_date", "fwd_12m_return_end_date"])
    events = _add_valuation_features(events)
    events = _add_excess_return_targets(events)
    events.to_csv(OUTPUT_DIR / "ablation_events.csv", index=False)
    predictions = []
    for name, target_type, features in EXPERIMENTS:
        predictions.append(_walk_forward(events, name, target_type, features))
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    perf = _performance(pred)
    portfolio = _portfolio_backtest(pred)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    portfolio.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False)
    _write_report(perf, portfolio)
    print(perf.to_string(index=False))
    print("\nPortfolio")
    print(portfolio.to_string(index=False))
    return 0


def _add_valuation_features(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    cap = frame["current_market_cap"].replace(0, np.nan)
    revenue = frame["total_revenue"].replace(0, np.nan)
    frame["sales_to_market_cap"] = revenue / cap
    frame["earnings_yield"] = frame["net_income"] / cap
    frame["fcf_yield"] = (frame["fcf_margin"] * revenue) / cap
    frame["gross_profit_to_market_cap"] = frame["gross_profit"] / cap
    sales_multiple = cap / revenue
    frame["growth_to_sales_multiple"] = frame["revenue_growth_yoy"] / sales_multiple.replace(0, np.nan)
    return frame


def _add_excess_return_targets(events: pd.DataFrame) -> pd.DataFrame:
    symbols = sorted(set(["SPY", *SECTOR_ETF.values()]))
    prices = download_yahoo_ohlcv(symbols, start="2023-01-01", chunk_size=25)
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    rows = []
    for row in events.to_dict("records"):
        trade_date = pd.Timestamp(row["trade_date"])
        start_pos = close.index.searchsorted(trade_date, side="left")
        end_date = pd.Timestamp(row["fwd_12m_return_end_date"])
        end_pos = close.index.searchsorted(end_date, side="left")
        out = dict(row)
        if start_pos < len(close) and end_pos < len(close):
            spy_return = close["SPY"].iloc[end_pos] / close["SPY"].iloc[start_pos] - 1.0
            sector_symbol = SECTOR_ETF.get(row.get("gics_sector"))
            if sector_symbol in close:
                sector_return = close[sector_symbol].iloc[end_pos] / close[sector_symbol].iloc[start_pos] - 1.0
            else:
                sector_return = spy_return
            out["fwd_12m_spy_excess_return"] = out["fwd_12m_return"] - spy_return
            out["fwd_12m_sector_excess_return"] = out["fwd_12m_return"] - sector_return
        else:
            out["fwd_12m_spy_excess_return"] = np.nan
            out["fwd_12m_sector_excess_return"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _target_column(target_type: str) -> str:
    if target_type in {"spy_excess_return"}:
        return "fwd_12m_spy_excess_return"
    if target_type in {"sector_excess_return", "sector_excess_rank_target"}:
        return "fwd_12m_sector_excess_return"
    return "fwd_12m_return"


def _walk_forward(events: pd.DataFrame, experiment: str, target_type: str, features: list[str]) -> pd.DataFrame:
    rows = []
    subset = events[(events["market_cap_bucket"] == "large_cap")].dropna(subset=["fwd_12m_return", "fwd_12m_return_end_date", _target_column(target_type)])
    for year in sorted(subset["trade_date"].dt.year.unique()):
        start = pd.Timestamp(f"{int(year)}-01-01")
        test = subset[subset["trade_date"].dt.year == year].copy()
        train = subset[(subset["trade_date"] < start) & (subset["fwd_12m_return_end_date"] < start)].copy()
        if len(train) < 25 or test.empty:
            continue
        y_train = _build_target(train, target_type)
        test["prediction"] = _fit_predict(train, test, features, y_train)
        test["target_return"] = test["fwd_12m_return"]
        test["ranking_target"] = _build_target(test, target_type)
        test["experiment"] = experiment
        test["target_type"] = target_type
        test["test_year"] = int(year)
        rows.append(test)
    return pd.concat(rows, ignore_index=True)


def _build_target(frame: pd.DataFrame, target_type: str) -> pd.Series:
    raw = frame[_target_column(target_type)].astype(float)
    if target_type in {"rank_target", "sector_excess_rank_target"}:
        return raw.groupby(frame["trade_date"]).rank(pct=True)
    if target_type == "top_quintile_target":
        return (raw >= raw.groupby(frame["trade_date"]).transform(lambda s: s.quantile(0.80))).astype(float)
    return raw


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], y_train: pd.Series) -> np.ndarray:
    usable = [f for f in features if f in train and train[f].notna().mean() >= 0.50 and train[f].astype(float).std(skipna=True) > 0]
    low = train[usable].quantile(0.01)
    high = train[usable].quantile(0.99)
    med = train[usable].median()
    x_train = train[usable].astype(float).clip(lower=low, upper=high, axis=1).fillna(med)
    x_test = test[usable].astype(float).clip(lower=low, upper=high, axis=1).fillna(med)
    mean = x_train.mean()
    std = x_train.std().replace(0, 1)
    x = ((x_train - mean) / std).fillna(0).to_numpy(float)
    t = ((x_test - mean) / std).fillna(0).to_numpy(float)
    y = y_train.to_numpy(float)
    y_mean = y.mean()
    x_mat = np.c_[np.ones(len(x)), x]
    t_mat = np.c_[np.ones(len(t)), t]
    penalty = np.eye(x_mat.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.solve(x_mat.T @ x_mat + 10.0 * penalty, x_mat.T @ (y - y_mean))
    beta[0] += y_mean
    return t_mat @ beta


def _performance(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, group in pred.groupby("experiment"):
        top = group[group["prediction"] >= group["prediction"].quantile(0.80)]
        bottom = group[group["prediction"] <= group["prediction"].quantile(0.20)]
        baseline = ((group["ranking_target"] - group["ranking_target"].mean()) ** 2).sum()
        residual = ((group["ranking_target"] - group["prediction"]) ** 2).sum()
        rows.append({
            "experiment": experiment,
            "target_type": group["target_type"].iloc[0],
            "observations": len(group),
            "oos_r2_on_training_target": 1 - residual / baseline if baseline > 0 else np.nan,
            "raw_return_spearman": group["prediction"].rank().corr(group["target_return"].rank()),
            "target_spearman": group["prediction"].rank().corr(group["ranking_target"].rank()),
            "top_quintile_mean_return": top["target_return"].mean(),
            "top_quintile_median_return": top["target_return"].median(),
            "top_quintile_hit_rate": (top["target_return"] > 0).mean(),
            "bottom_quintile_mean_return": bottom["target_return"].mean(),
            "top_minus_bottom_mean": top["target_return"].mean() - bottom["target_return"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["raw_return_spearman", "top_quintile_mean_return"], ascending=False)


def _portfolio_backtest(pred: pd.DataFrame) -> pd.DataFrame:
    # Event-level portfolio proxy: average realized return of top quintile.
    rows = []
    for experiment, group in pred.groupby("experiment"):
        top = group[group["prediction"] >= group["prediction"].quantile(0.80)]
        rows.append({
            "experiment": experiment,
            "event_top_quintile_mean_return": top["target_return"].mean(),
            "event_top_quintile_median_return": top["target_return"].median(),
            "event_top_quintile_hit_rate": (top["target_return"] > 0).mean(),
            "event_count": len(top),
        })
    return pd.DataFrame(rows).sort_values("event_top_quintile_mean_return", ascending=False)


def _write_report(perf: pd.DataFrame, portfolio: pd.DataFrame) -> None:
    lines = [
        "# Fundamental Upgrade Ablation",
        "",
        "Tests proposed upgrades: valuation-style features, excess-return targets, rank/classification targets, sector/macro features, and combined variants.",
        "",
        "## Predictive performance",
        "",
        _markdown_table(perf),
        "",
        "## Event top-quintile proxy",
        "",
        _markdown_table(portfolio),
        "",
        "## Interpretation",
        "",
        "- This suite evaluates whether changes improve the actual raw-return rank and top-quintile returns.",
        "- Excess-return/rank/classification targets are only useful if raw forward-return portfolio outcomes improve.",
    ]
    (OUTPUT_DIR / "FUNDAMENTAL_UPGRADE_ABLATION.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "event_count"}:
            if any(token in col for token in ["r2", "spearman", "return", "rate"]):
                display[col] = display[col].map(lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
