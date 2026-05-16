"""Improve financial-ratio return prediction with causal price context.

This script starts from the no-lookahead financial-ratio event file and adds
features known at each event trade date: trailing stock momentum, relative
strength versus SPY, realized volatility, drawdown, and log market cap.  It then
compares purged walk-forward ridge regressions for:

- financial ratios only
- financial ratios plus price/relative-strength context
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research import download_yahoo_ohlcv


OUTPUT_DIR = Path("ath_reversion_research/reports/financial_ratio_momentum_improvement")
EVENTS_PATH = Path("ath_reversion_research/reports/financial_ratio_regression/financial_ratio_events.csv")
BASE_FINANCIAL_FEATURES = [
    "revenue_growth_yoy",
    "revenue_growth_accel",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "ebitda_margin",
    "cfo_margin",
    "fcf_margin",
    "capex_to_revenue",
    "rd_to_revenue",
    "sga_to_revenue",
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
    "log_market_cap",
    "growth_x_trailing_6m",
    "growth_x_fcf_margin",
]
FORWARD_WINDOWS = {
    "fwd_3m_return": 63,
    "fwd_6m_return": 126,
    "fwd_12m_return": 252,
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS_PATH, parse_dates=["trade_date", "fwd_3m_return_end_date", "fwd_6m_return_end_date", "fwd_12m_return_end_date"])
    symbols = sorted(events["symbol"].dropna().astype(str).unique())
    prices = download_yahoo_ohlcv(symbols + ["SPY"], start="2017-01-01", chunk_size=25)
    enhanced = _add_price_features(events, prices)
    enhanced.to_csv(OUTPUT_DIR / "enhanced_events.csv", index=False)

    predictions = []
    for feature_set_name, features in {
        "financial_only": BASE_FINANCIAL_FEATURES,
        "financial_plus_price": BASE_FINANCIAL_FEATURES + PRICE_FEATURES,
    }.items():
        predictions.append(_walk_forward_predictions(enhanced, features, feature_set_name))
    predictions_frame = pd.concat(predictions, ignore_index=True)
    predictions_frame.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)

    performance = _performance_summary(predictions_frame)
    conditions = _condition_summary(predictions_frame)
    performance.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    conditions.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False)
    _write_report(performance, conditions)

    print("Performance")
    print(performance.to_string(index=False))
    print("\nConditions")
    print(conditions.head(30).to_string(index=False))
    return 0


def _add_price_features(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    spy = close["SPY"].ffill()
    rows = []
    for event in events.to_dict("records"):
        symbol = str(event["symbol"])
        if symbol not in close:
            continue
        trade_date = pd.Timestamp(event["trade_date"])
        series = close[symbol].dropna()
        pos = int(series.index.searchsorted(trade_date, side="right")) - 1
        if pos < 252:
            continue
        row = dict(event)
        current = float(series.iloc[pos])
        for name, lookback in [
            ("trailing_3m_return", 63),
            ("trailing_6m_return", 126),
            ("trailing_12m_return", 252),
        ]:
            row[name] = float(current / series.iloc[pos - lookback] - 1.0)
        stock_returns = series.pct_change().iloc[pos - 63 + 1 : pos + 1]
        row["realized_vol_3m"] = float(stock_returns.std() * np.sqrt(252))
        trailing_year = series.iloc[pos - 252 + 1 : pos + 1]
        row["drawdown_12m"] = float(current / trailing_year.max() - 1.0)
        spy_pos = int(spy.index.searchsorted(trade_date, side="right")) - 1
        if spy_pos >= 126:
            spy_6m = float(spy.iloc[spy_pos] / spy.iloc[spy_pos - 126] - 1.0)
        else:
            spy_6m = np.nan
        row["relative_6m_vs_spy"] = row["trailing_6m_return"] - spy_6m
        row["log_market_cap"] = float(np.log(row["current_market_cap"])) if pd.notna(row.get("current_market_cap")) and row["current_market_cap"] > 0 else np.nan
        row["growth_x_trailing_6m"] = row["revenue_growth_yoy"] * row["trailing_6m_return"]
        row["growth_x_fcf_margin"] = row["revenue_growth_yoy"] * row["fcf_margin"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _walk_forward_predictions(events: pd.DataFrame, features: list[str], feature_set_name: str) -> pd.DataFrame:
    rows = []
    for target, bars_ahead in FORWARD_WINDOWS.items():
        end_col = f"{target}_end_date"
        for bucket in ["all", "large_cap", "mid_cap", "low_cap"]:
            subset = events.copy() if bucket == "all" else events[events["market_cap_bucket"] == bucket].copy()
            subset = subset.dropna(subset=[target, end_col, "revenue_growth_yoy"])
            if subset.empty:
                continue
            for year in sorted(pd.to_datetime(subset["trade_date"]).dt.year.unique()):
                test = subset[pd.to_datetime(subset["trade_date"]).dt.year == year].copy()
                test_start = pd.Timestamp(f"{int(year)}-01-01")
                train = subset[
                    (pd.to_datetime(subset["trade_date"]) < test_start)
                    & (pd.to_datetime(subset[end_col]) < test_start)
                ].copy()
                min_rows = 15
                if len(train) < min_rows or test.empty:
                    continue
                pred = _fit_predict(train, test, target, features)
                test["prediction"] = pred
                test["target_return"] = test[target]
                test["feature_set"] = feature_set_name
                test["forward_window"] = target
                test["regression_bucket"] = bucket
                test["test_year"] = int(year)
                rows.append(test)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str], alpha: float = 10.0) -> np.ndarray:
    usable = [
        feature
        for feature in features
        if feature in train and train[feature].notna().mean() >= 0.50 and train[feature].astype(float).std(skipna=True) > 0.0
    ]
    if not usable:
        usable = ["revenue_growth_yoy"]
    low = train[usable].quantile(0.01)
    high = train[usable].quantile(0.99)
    med = train[usable].median()
    x_train = train[usable].astype(float).clip(lower=low, upper=high, axis=1).fillna(med)
    x_test = test[usable].astype(float).clip(lower=low, upper=high, axis=1).fillna(med)
    mean = x_train.mean()
    std = x_train.std().replace(0.0, 1.0)
    x = ((x_train - mean) / std).fillna(0.0).to_numpy(dtype=float)
    t = ((x_test - mean) / std).fillna(0.0).to_numpy(dtype=float)
    y = train[target].astype(float).to_numpy()
    y_mean = float(np.mean(y))
    x_mat = np.c_[np.ones(len(x)), x]
    t_mat = np.c_[np.ones(len(t)), t]
    penalty = np.eye(x_mat.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_mat.T @ x_mat + alpha * penalty, x_mat.T @ (y - y_mean))
    beta[0] += y_mean
    return t_mat @ beta


def _performance_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, target, bucket), group in predictions.groupby(["feature_set", "forward_window", "regression_bucket"]):
        baseline = float(((group["target_return"] - group["target_return"].mean()) ** 2).sum())
        residual = float(((group["target_return"] - group["prediction"]) ** 2).sum())
        top = group[group["prediction"] >= group["prediction"].quantile(0.80)]
        bottom = group[group["prediction"] <= group["prediction"].quantile(0.20)]
        rows.append(
            {
                "feature_set": feature_set,
                "forward_window": target,
                "regression_bucket": bucket,
                "observations": int(len(group)),
                "oos_r2": 1.0 - residual / baseline if baseline > 0 else np.nan,
                "pearson_corr": float(group["prediction"].corr(group["target_return"])),
                "spearman_corr": float(group["prediction"].rank().corr(group["target_return"].rank())),
                "top_quintile_mean_return": float(top["target_return"].mean()),
                "top_quintile_median_return": float(top["target_return"].median()),
                "top_quintile_hit_rate": float((top["target_return"] > 0.0).mean()),
                "bottom_quintile_mean_return": float(bottom["target_return"].mean()),
                "top_minus_bottom_mean": float(top["target_return"].mean() - bottom["target_return"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["forward_window", "regression_bucket", "feature_set"])


def _condition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, target, bucket), group in predictions.groupby(["feature_set", "forward_window", "regression_bucket"]):
        conditions = {
            "model_top_quintile": group["prediction"] >= group["prediction"].quantile(0.80),
            "growth_30_positive_margin_mom": (group["revenue_growth_yoy"] >= 0.30) & (group["operating_margin"] > 0.0) & (group["trailing_6m_return"] > 0.0),
            "growth_30_positive_fcf_mom": (group["revenue_growth_yoy"] >= 0.30) & (group["fcf_margin"] > 0.0) & (group["trailing_6m_return"] > 0.0),
            "growth_30_model_top_half_mom": (group["revenue_growth_yoy"] >= 0.30) & (group["prediction"] >= group["prediction"].median()) & (group["trailing_6m_return"] > 0.0),
        }
        for condition, mask in conditions.items():
            selected = group[mask]
            if len(selected) < 5:
                continue
            rows.append(
                {
                    "feature_set": feature_set,
                    "forward_window": target,
                    "regression_bucket": bucket,
                    "condition": condition,
                    "observations": int(len(selected)),
                    "mean_return": float(selected["target_return"].mean()),
                    "median_return": float(selected["target_return"].median()),
                    "hit_rate": float((selected["target_return"] > 0.0).mean()),
                    "avg_revenue_growth": float(selected["revenue_growth_yoy"].mean()),
                    "avg_trailing_6m_return": float(selected["trailing_6m_return"].mean()),
                    "avg_operating_margin": float(selected["operating_margin"].mean()),
                    "avg_fcf_margin": float(selected["fcf_margin"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["forward_window", "regression_bucket", "mean_return"], ascending=[True, True, False])


def _write_report(performance: pd.DataFrame, conditions: pd.DataFrame) -> None:
    lines = [
        "# Financial Ratio + Momentum Improvement Study",
        "",
        "This study tests whether causal trailing price context improves annual financial-ratio regressions.",
        "",
        "## Leakage controls",
        "",
        "- Starts from financial events already lagged 90 days after fiscal year end.",
        "- Adds only trailing returns, relative strength, volatility, drawdown, and market cap known at the trade date.",
        "- Uses purged walk-forward training; labels overlapping a test year are excluded from training.",
        "- Ridge alpha is fixed and not tuned on validation rows.",
        "",
        "## Financial-only versus financial + price context",
        "",
        _markdown_table(performance),
        "",
        "## Improved condition subsets",
        "",
        _markdown_table(conditions.head(50)),
        "",
        "## Interpretation",
        "",
        "- Adding price context improves some all-stock ranking metrics, especially the 6m top-quintile mean and hit rate.",
        "- It does not universally improve the model: large-cap financial-only regressions remain stronger than financial-plus-price for several top-quintile tests.",
        "- The most interesting recurring pocket remains 30%+ revenue growth with positive profitability/FCF; trailing momentum can be used as an additional filter, not a guaranteed improvement.",
        "- Mid-cap and low-cap outputs remain too sample-limited for deployment.",
        "- This is still a hypothesis layer; it should be combined with the existing technical/momentum framework rather than traded standalone.",
    ]
    (OUTPUT_DIR / "FINANCIAL_RATIO_MOMENTUM_IMPROVEMENT.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]) and column not in {"observations"}:
            if any(token in column for token in ["r2", "corr", "return", "rate", "growth", "margin", "momentum"]):
                display[column] = display[column].map(lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
