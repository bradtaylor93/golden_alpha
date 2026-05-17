"""DoltHub aligned replication of the Yahoo S&P annual model.

This script makes the Dolt validation closer to the Yahoo/S&P model:

- uses local Dolt income/balance/cash-flow fundamentals,
- uses the same broad feature names where possible,
- attaches current market-cap buckets from the Yahoo S&P run,
- compares price-only, income-only, financial-only, financial+price,
- runs a prediction-weighted top-quintile portfolio from Dolt predictions.

It still uses current S&P constituents and Yahoo adjusted prices, so it is not
fully point-in-time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research.metrics import summarize_returns
import run_dolthub_full_fundamental_validation as dolt


OUTPUT_DIR = Path("ath_reversion_research/reports/dolthub_aligned_replication")
CAPS = Path("ath_reversion_research/reports/sp500_annual_fundamental_regression/current_market_caps.csv")
YAHOO_PRICE_EXPORT = Path("ath_reversion_research/reports/dolthub_full_fundamental_validation/price_subset_yahoo.csv")

FINANCIAL_FEATURES = [
    "revenue_growth_yoy",
    "revenue_growth_accel",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "ebitda_margin",
    "cfo_margin",
    "fcf_margin",
    "capex_to_revenue",
    "debt_to_assets",
    "debt_to_equity",
    "cash_to_assets",
    "current_ratio",
    "asset_turnover",
]
PRICE_FEATURES = [
    "trailing_3m_return",
    "trailing_6m_return",
    "trailing_12m_return",
    "relative_6m_vs_spy",
    "realized_vol_3m",
    "drawdown_12m",
    "growth_x_trailing_6m",
    "growth_x_fcf_margin",
]
FEATURE_SETS = {
    "price_only": PRICE_FEATURES,
    "income_only": ["revenue_growth_yoy", "revenue_growth_accel", "gross_margin", "operating_margin", "net_margin", "ebitda_margin"],
    "financial_only": FINANCIAL_FEATURES,
    "financial_plus_price": FINANCIAL_FEATURES + PRICE_FEATURES,
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = _load_or_build_events()
    caps = pd.read_csv(CAPS)
    events = events.drop(columns=[c for c in ["current_market_cap", "market_cap_bucket"] if c in events], errors="ignore")
    events = events.merge(caps.loc[:, ["symbol", "current_market_cap", "market_cap_bucket"]], on="symbol", how="left")
    events.to_csv(OUTPUT_DIR / "aligned_events.csv", index=False)
    predictions = []
    for name, features in FEATURE_SETS.items():
        predictions.append(_walk_forward(events, name, features))
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    perf = _performance(pred)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    portfolio_summary, portfolio_returns = _portfolio_backtest(pred)
    portfolio_summary.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False)
    portfolio_returns.to_csv(OUTPUT_DIR / "daily_portfolio_returns.csv", index_label="date")
    _write_report(perf, portfolio_summary)
    print(perf.to_string(index=False))
    print("\nPortfolio")
    print(portfolio_summary.to_string(index=False))
    return 0


def _load_or_build_events() -> pd.DataFrame:
    cached = Path("ath_reversion_research/reports/dolthub_full_fundamental_validation/dolt_full_feature_events.csv")
    if cached.exists():
        return pd.read_csv(cached, parse_dates=["trade_date", "fwd_3m_return_end_date", "fwd_6m_return_end_date", "fwd_12m_return_end_date"])
    symbols = pd.read_csv("ath_reversion_research/reports/dolthub_full_fundamental_validation/sp500_symbols.csv")["symbol"].tolist()
    fundamentals = dolt._load_fundamentals(symbols)
    events = dolt._build_events(fundamentals)
    prices = dolt.download_yahoo_ohlcv(symbols + ["SPY"], start="2012-01-01", chunk_size=25)
    return dolt._attach_prices(events, prices)


def _walk_forward(events: pd.DataFrame, feature_set: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for target in ["fwd_3m_return", "fwd_6m_return", "fwd_12m_return"]:
        end_col = f"{target}_end_date"
        for bucket in ["all", "large_cap", "mid_cap"]:
            subset = events.copy() if bucket == "all" else events[events["market_cap_bucket"] == bucket].copy()
            subset = subset.dropna(subset=[target, end_col, "revenue_growth_yoy"])
            for year in sorted(subset["trade_date"].dt.year.unique()):
                start = pd.Timestamp(f"{int(year)}-01-01")
                test = subset[subset["trade_date"].dt.year == year].copy()
                train = subset[(subset["trade_date"] < start) & (subset[end_col] < start)].copy()
                if len(train) < 100 or test.empty:
                    continue
                test["prediction"] = _fit_predict(train, test, target, features)
                test["target_return"] = test[target]
                test["feature_set"] = feature_set
                test["forward_window"] = target
                test["regression_bucket"] = bucket
                test["test_year"] = int(year)
                rows.append(test)
    return pd.concat(rows, ignore_index=True)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str]) -> np.ndarray:
    usable = [f for f in features if f in train and train[f].notna().mean() >= 0.50 and train[f].std(skipna=True) > 0]
    low = train[usable].quantile(0.01)
    high = train[usable].quantile(0.99)
    med = train[usable].median()
    x_train = train[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    x_test = test[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    mean = x_train.mean()
    std = x_train.std().replace(0, 1)
    x = ((x_train - mean) / std).fillna(0).to_numpy(float)
    t = ((x_test - mean) / std).fillna(0).to_numpy(float)
    y = train[target].to_numpy(float)
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
    for keys, group in pred.groupby(["feature_set", "forward_window", "regression_bucket"]):
        feature_set, target, bucket = keys
        baseline = ((group["target_return"] - group["target_return"].mean()) ** 2).sum()
        residual = ((group["target_return"] - group["prediction"]) ** 2).sum()
        top = group[group["prediction"] >= group["prediction"].quantile(0.80)]
        bottom = group[group["prediction"] <= group["prediction"].quantile(0.20)]
        rows.append({
            "feature_set": feature_set,
            "forward_window": target,
            "regression_bucket": bucket,
            "observations": len(group),
            "test_years": group["test_year"].nunique(),
            "oos_r2": 1 - residual / baseline if baseline > 0 else np.nan,
            "pearson_corr": group["prediction"].corr(group["target_return"]),
            "spearman_corr": group["prediction"].rank().corr(group["target_return"].rank()),
            "top_quintile_mean_return": top["target_return"].mean(),
            "top_quintile_median_return": top["target_return"].median(),
            "top_quintile_hit_rate": (top["target_return"] > 0).mean(),
            "bottom_quintile_mean_return": bottom["target_return"].mean(),
            "top_minus_bottom_mean": top["target_return"].mean() - bottom["target_return"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["forward_window", "regression_bucket", "feature_set"])


def _portfolio_backtest(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = pred[
        (pred["feature_set"] == "financial_plus_price")
        & (pred["forward_window"] == "fwd_12m_return")
        & (pred["regression_bucket"] == "large_cap")
    ].copy()
    if subset.empty or not YAHOO_PRICE_EXPORT.exists():
        return pd.DataFrame(), pd.DataFrame()
    prices = pd.read_csv(YAHOO_PRICE_EXPORT, parse_dates=["date"])
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    symbols = sorted(subset["symbol"].dropna().astype(str).unique())
    target = pd.DataFrame(0.0, index=close.index, columns=symbols)
    for trade_date, group in subset.groupby("trade_date"):
        group = group.sort_values("prediction", ascending=False)
        top = group.head(max(1, int(np.ceil(len(group) * 0.20))))
        weights = pd.Series(1.0 / len(top), index=top["symbol"].astype(str))
        start = target.index.searchsorted(pd.Timestamp(trade_date), side="left")
        end = target.index.searchsorted(pd.Timestamp(group["fwd_12m_return_end_date"].max()), side="right")
        target.loc[target.index[start:end], weights.index] = target.loc[target.index[start:end], weights.index].add(weights, axis=1)
    gross = target.abs().sum(axis=1).replace(0, np.nan)
    target = target.div(gross, axis=0).fillna(0.0)
    returns = close.pct_change().reindex_like(target).fillna(0.0)
    executed = target.shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    net = (executed * returns).sum(axis=1) - turnover * 0.001
    active = target.abs().sum(axis=1) > 0
    start = active[active].index.min()
    summary = summarize_returns(net.loc[start:], turnover.loc[start:]).as_dict()
    summary["portfolio"] = "dolthub_aligned_largecap_top_quintile_equal"
    return pd.DataFrame([summary]), pd.DataFrame({"daily_return": net})


def _write_report(perf: pd.DataFrame, portfolio: pd.DataFrame) -> None:
    lines = [
        "# DoltHub Aligned Replication",
        "",
        "Aligns the DoltHub fundamentals validation more closely to the Yahoo S&P annual model and compares feature sets by market-cap bucket.",
        "",
        "## Performance",
        "",
        _markdown_table(perf),
        "",
        "## Prediction-weighted portfolio",
        "",
        _markdown_table(portfolio),
        "",
        "## Interpretation",
        "",
        "- This is the apples-to-apples Dolt check requested after noticing price-only strength.",
        "- Feature engineering is closer to the Yahoo model, and large-cap filtering uses the same current market-cap buckets from the Yahoo S&P run.",
        "- If price-only remains close to financial+price, fundamentals should be viewed as a filter/anchor rather than the sole driver.",
    ]
    (OUTPUT_DIR / "DOLTHUB_ALIGNED_REPLICATION.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "test_years"}:
            if any(token in col for token in ["r2", "corr", "return", "rate", "drawdown", "std"]):
                display[col] = display[col].map(lambda v: f"{float(v) * 100:.2f}%" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
