"""Walk-forward financial-ratio regression against future stock returns.

Annual financial statement features are aligned with a conservative 90-day
post-fiscal-year-end availability lag.  For each test year and forward return
horizon, training rows whose forward-return measurement overlaps the test year
are purged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from ath_reversion_research import download_yahoo_ohlcv, symbols_for


OUTPUT_DIR = Path("ath_reversion_research/reports/financial_ratio_regression")
EXPANDED_SYMBOLS = Path("ath_reversion_research/reports/expanded_stock_signals/downloaded_symbols.csv")
REPORTING_LAG_DAYS = 90
FORWARD_WINDOWS = {
    "fwd_3m_return": 63,
    "fwd_6m_return": 126,
    "fwd_12m_return": 252,
}
FEATURES = [
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


@dataclass(frozen=True)
class RegressionConfig:
    start: str = "2018-01-01"
    min_revenue: float = 100_000_000.0
    reporting_lag_days: int = REPORTING_LAG_DAYS
    ridge_alpha: float = 10.0
    min_train_rows: int = 15


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = RegressionConfig()
    symbols = _load_symbols()
    fundamentals, market_caps = _download_fundamentals(symbols)
    fundamentals.to_csv(OUTPUT_DIR / "annual_fundamentals_raw.csv", index=False)
    market_caps.to_csv(OUTPUT_DIR / "current_market_caps.csv", index=False)

    features = _build_feature_events(fundamentals, market_caps, config)
    prices = download_yahoo_ohlcv(symbols, start=config.start, chunk_size=25)
    events = _attach_forward_returns(features, prices, config.reporting_lag_days)
    events.to_csv(OUTPUT_DIR / "financial_ratio_events.csv", index=False)

    predictions = _walk_forward_predictions(events, config)
    predictions.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    performance = _performance_summary(predictions)
    condition_summary = _condition_summary(predictions)
    coefficient_summary = _coefficient_summary(predictions)
    performance.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    condition_summary.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False)
    coefficient_summary.to_csv(OUTPUT_DIR / "coefficient_summary.csv", index=False)
    _write_report(events, predictions, performance, condition_summary, coefficient_summary, config)

    print("Performance")
    print(performance.to_string(index=False))
    print("\nConditions")
    print(condition_summary.head(30).to_string(index=False))
    return 0


def _load_symbols() -> list[str]:
    symbols: list[str] = []
    if EXPANDED_SYMBOLS.exists():
        frame = pd.read_csv(EXPANDED_SYMBOLS)
        if "downloaded_symbols" in frame.columns:
            symbols.extend(frame["downloaded_symbols"].dropna().astype(str).tolist())
    symbols.extend(symbols_for(["broad_stock_sample"]))
    out: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        upper = symbol.upper()
        if upper not in seen:
            out.append(upper)
            seen.add(upper)
    return out


def _download_fundamentals(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    caps: list[dict[str, object]] = []
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        try:
            market_cap = _safe_market_cap(ticker)
            caps.append({"symbol": symbol, "current_market_cap": market_cap, "market_cap_bucket": _cap_bucket(market_cap)})
            income = ticker.income_stmt
            balance = ticker.balance_sheet
            cashflow = ticker.cashflow
        except Exception as exc:  # pragma: no cover - network/vendor defensive
            rows.append({"symbol": symbol, "error": str(exc)})
            continue
        if income is None or income.empty or "Total Revenue" not in income.index:
            rows.append({"symbol": symbol, "error": "missing_total_revenue"})
            continue
        fiscal_dates = sorted(pd.to_datetime(income.columns))
        for fiscal_date in fiscal_dates:
            row = {
                "symbol": symbol,
                "fiscal_period_end": pd.Timestamp(fiscal_date).normalize(),
                "error": "",
                "total_revenue": _value(income, fiscal_date, ["Total Revenue"]),
                "gross_profit": _value(income, fiscal_date, ["Gross Profit"]),
                "operating_income": _value(income, fiscal_date, ["Operating Income", "Operating Income Loss"]),
                "net_income": _value(income, fiscal_date, ["Net Income", "Net Income Common Stockholders"]),
                "ebitda": _value(income, fiscal_date, ["EBITDA", "Normalized EBITDA"]),
                "research_development": _value(income, fiscal_date, ["Research And Development"]),
                "selling_general_admin": _value(income, fiscal_date, ["Selling General And Administration"]),
                "operating_cash_flow": _value(cashflow, fiscal_date, ["Operating Cash Flow", "Total Cash From Operating Activities"]),
                "capital_expenditure": _value(cashflow, fiscal_date, ["Capital Expenditure", "Capital Expenditures"]),
                "free_cash_flow": _value(cashflow, fiscal_date, ["Free Cash Flow"]),
                "total_assets": _value(balance, fiscal_date, ["Total Assets"]),
                "total_debt": _value(balance, fiscal_date, ["Total Debt"]),
                "stockholders_equity": _value(balance, fiscal_date, ["Stockholders Equity", "Total Stockholder Equity"]),
                "cash": _value(balance, fiscal_date, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
                "current_assets": _value(balance, fiscal_date, ["Current Assets", "Total Current Assets"]),
                "current_liabilities": _value(balance, fiscal_date, ["Current Liabilities", "Total Current Liabilities"]),
            }
            rows.append(row)
    fundamentals = pd.DataFrame(rows).sort_values(["symbol", "fiscal_period_end"]).reset_index(drop=True)
    market_caps = pd.DataFrame(caps).drop_duplicates("symbol")
    return fundamentals, market_caps


def _safe_market_cap(ticker: yf.Ticker) -> float:
    try:
        market_cap = ticker.fast_info.get("market_cap")
        if market_cap:
            return float(market_cap)
    except Exception:
        pass
    try:
        market_cap = ticker.info.get("marketCap")
        if market_cap:
            return float(market_cap)
    except Exception:
        pass
    return np.nan


def _cap_bucket(market_cap: float) -> str:
    if not np.isfinite(market_cap):
        return "unknown"
    if market_cap >= 50_000_000_000:
        return "large_cap"
    if market_cap >= 10_000_000_000:
        return "mid_cap"
    return "low_cap"


def _value(statement: pd.DataFrame | None, fiscal_date: pd.Timestamp, labels: list[str]) -> float:
    if statement is None or statement.empty:
        return np.nan
    for label in labels:
        if label in statement.index and fiscal_date in pd.to_datetime(statement.columns):
            value = statement.loc[label, fiscal_date]
            if pd.notna(value):
                return float(value)
    return np.nan


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _build_feature_events(fundamentals: pd.DataFrame, market_caps: pd.DataFrame, config: RegressionConfig) -> pd.DataFrame:
    frame = fundamentals[(fundamentals["error"].fillna("") == "") & fundamentals["total_revenue"].notna()].copy()
    frame["fiscal_period_end"] = pd.to_datetime(frame["fiscal_period_end"])
    frame = frame.sort_values(["symbol", "fiscal_period_end"])
    frame["prior_revenue"] = frame.groupby("symbol")["total_revenue"].shift(1)
    frame["prior_revenue_growth_yoy"] = frame.groupby("symbol")["total_revenue"].pct_change().shift(1)
    frame["revenue_growth_yoy"] = frame["total_revenue"] / frame["prior_revenue"] - 1.0
    frame["revenue_growth_accel"] = frame["revenue_growth_yoy"] - frame["prior_revenue_growth_yoy"]
    frame["gross_margin"] = _safe_div(frame["gross_profit"], frame["total_revenue"])
    frame["operating_margin"] = _safe_div(frame["operating_income"], frame["total_revenue"])
    frame["net_margin"] = _safe_div(frame["net_income"], frame["total_revenue"])
    frame["ebitda_margin"] = _safe_div(frame["ebitda"], frame["total_revenue"])
    frame["cfo_margin"] = _safe_div(frame["operating_cash_flow"], frame["total_revenue"])
    free_cash_flow = frame["free_cash_flow"].where(
        frame["free_cash_flow"].notna(),
        frame["operating_cash_flow"] + frame["capital_expenditure"],
    )
    frame["fcf_margin"] = _safe_div(free_cash_flow, frame["total_revenue"])
    frame["capex_to_revenue"] = _safe_div(frame["capital_expenditure"].abs(), frame["total_revenue"])
    frame["rd_to_revenue"] = _safe_div(frame["research_development"].fillna(0.0), frame["total_revenue"])
    frame["sga_to_revenue"] = _safe_div(frame["selling_general_admin"].fillna(0.0), frame["total_revenue"])
    frame["debt_to_assets"] = _safe_div(frame["total_debt"], frame["total_assets"])
    frame["debt_to_equity"] = _safe_div(frame["total_debt"], frame["stockholders_equity"].abs())
    frame["cash_to_assets"] = _safe_div(frame["cash"], frame["total_assets"])
    frame["current_ratio"] = _safe_div(frame["current_assets"], frame["current_liabilities"])
    frame["asset_turnover"] = _safe_div(frame["total_revenue"], frame["total_assets"])
    frame["availability_date"] = frame["fiscal_period_end"] + pd.Timedelta(days=config.reporting_lag_days)
    frame = frame.merge(market_caps, on="symbol", how="left")
    frame = frame[
        (frame["prior_revenue"] >= config.min_revenue)
        & np.isfinite(frame["revenue_growth_yoy"])
        & (frame["availability_date"] >= pd.Timestamp(config.start))
    ].copy()
    return frame.reset_index(drop=True)


def _attach_forward_returns(events: pd.DataFrame, prices: pd.DataFrame, reporting_lag_days: int) -> pd.DataFrame:
    price_map = {
        symbol: group.sort_values("date").reset_index(drop=True)
        for symbol, group in prices.groupby("symbol")
    }
    rows: list[dict[str, object]] = []
    for event in events.to_dict("records"):
        symbol_prices = price_map.get(str(event["symbol"]))
        if symbol_prices is None or symbol_prices.empty:
            continue
        dates = pd.to_datetime(symbol_prices["date"])
        start_pos = int(dates.searchsorted(pd.Timestamp(event["availability_date"]), side="left"))
        if start_pos >= len(symbol_prices):
            continue
        start_close = float(symbol_prices.loc[start_pos, "close"])
        row = dict(event)
        row["trade_date"] = symbol_prices.loc[start_pos, "date"]
        row["trade_close"] = start_close
        row["reporting_lag_days"] = reporting_lag_days
        for name, bars_ahead in FORWARD_WINDOWS.items():
            end_pos = start_pos + bars_ahead
            if end_pos < len(symbol_prices):
                row[name] = float(symbol_prices.loc[end_pos, "close"] / start_close - 1.0)
                row[f"{name}_end_date"] = symbol_prices.loc[end_pos, "date"]
            else:
                row[name] = np.nan
                row[f"{name}_end_date"] = pd.NaT
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _walk_forward_predictions(events: pd.DataFrame, config: RegressionConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    coeff_rows: list[dict[str, object]] = []
    for target in FORWARD_WINDOWS:
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
                if len(train) < config.min_train_rows or test.empty:
                    continue
                pred, coefs = _fit_predict(train, test, target, config.ridge_alpha)
                test["prediction"] = pred
                test["target_return"] = test[target]
                test["forward_window"] = target
                test["regression_bucket"] = bucket
                test["test_year"] = int(year)
                rows.append(test)
                for feature, coef in coefs.items():
                    coeff_rows.append(
                        {
                            "forward_window": target,
                            "regression_bucket": bucket,
                            "test_year": int(year),
                            "feature": feature,
                            "standardized_coef": coef,
                        }
                    )
    predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    coefficients = pd.DataFrame(coeff_rows)
    coefficients.to_csv(OUTPUT_DIR / "walk_forward_coefficients.csv", index=False)
    return predictions


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, alpha: float) -> tuple[np.ndarray, dict[str, float]]:
    usable_features = [
        feature
        for feature in FEATURES
        if train[feature].notna().mean() >= 0.50 and train[feature].astype(float).std(skipna=True) > 0.0
    ]
    if not usable_features:
        usable_features = ["revenue_growth_yoy"]
    train_quantiles_low = train[usable_features].quantile(0.01)
    train_quantiles_high = train[usable_features].quantile(0.99)
    train_median = train[usable_features].median()
    x_train = (
        train[usable_features]
        .astype(float)
        .clip(lower=train_quantiles_low, upper=train_quantiles_high, axis=1)
        .fillna(train_median)
    )
    x_test = (
        test[usable_features]
        .astype(float)
        .clip(lower=train_quantiles_low, upper=train_quantiles_high, axis=1)
        .fillna(train_median)
    )
    mean = x_train.mean()
    std = x_train.std().replace(0.0, 1.0)
    x = ((x_train - mean) / std).fillna(0.0).to_numpy(dtype=float)
    t = ((x_test - mean) / std).fillna(0.0).to_numpy(dtype=float)
    y = train[target].astype(float).to_numpy()
    y_mean = float(np.mean(y))
    y_centered = y - y_mean
    x_mat = np.c_[np.ones(len(x)), x]
    t_mat = np.c_[np.ones(len(t)), t]
    penalty = np.eye(x_mat.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_mat.T @ x_mat + alpha * penalty, x_mat.T @ y_centered)
    beta[0] += y_mean
    predictions = t_mat @ beta
    coefficients = {feature: 0.0 for feature in FEATURES}
    coefficients.update(dict(zip(usable_features, beta[1:])))
    return predictions, coefficients


def _performance_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, bucket), group in predictions.groupby(["forward_window", "regression_bucket"]):
        if len(group) < 10:
            continue
        baseline = float(((group["target_return"] - group["target_return"].mean()) ** 2).sum())
        residual = float(((group["target_return"] - group["prediction"]) ** 2).sum())
        pred_rank = group["prediction"].rank()
        actual_rank = group["target_return"].rank()
        top_quintile = group[group["prediction"] >= group["prediction"].quantile(0.80)]
        bottom_quintile = group[group["prediction"] <= group["prediction"].quantile(0.20)]
        rows.append(
            {
                "forward_window": target,
                "regression_bucket": bucket,
                "observations": int(len(group)),
                "oos_r2": 1.0 - residual / baseline if baseline > 0 else np.nan,
                "pearson_corr": float(group["prediction"].corr(group["target_return"])),
                "spearman_corr": float(pred_rank.corr(actual_rank)),
                "top_quintile_mean_return": float(top_quintile["target_return"].mean()),
                "top_quintile_median_return": float(top_quintile["target_return"].median()),
                "top_quintile_hit_rate": float((top_quintile["target_return"] > 0.0).mean()),
                "bottom_quintile_mean_return": float(bottom_quintile["target_return"].mean()),
                "top_minus_bottom_mean": float(top_quintile["target_return"].mean() - bottom_quintile["target_return"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["forward_window", "regression_bucket"])


def _condition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    conditions = {
        "model_top_quintile": lambda f: f["prediction"] >= f["prediction"].quantile(0.80),
        "revenue_growth_30pct": lambda f: f["revenue_growth_yoy"] >= 0.30,
        "rev_growth_30pct_model_top_half": lambda f: (f["revenue_growth_yoy"] >= 0.30) & (f["prediction"] >= f["prediction"].median()),
        "rev_growth_30pct_positive_margin": lambda f: (f["revenue_growth_yoy"] >= 0.30) & (f["operating_margin"] > 0.0),
        "rev_growth_30pct_positive_fcf": lambda f: (f["revenue_growth_yoy"] >= 0.30) & (f["fcf_margin"] > 0.0),
        "positive_fcf_top_model": lambda f: (f["fcf_margin"] > 0.0) & (f["prediction"] >= f["prediction"].quantile(0.80)),
    }
    rows = []
    for (target, bucket), group in predictions.groupby(["forward_window", "regression_bucket"]):
        for condition_name, mask_fn in conditions.items():
            selected = group[mask_fn(group)]
            if len(selected) < 5:
                continue
            rows.append(
                {
                    "forward_window": target,
                    "regression_bucket": bucket,
                    "condition": condition_name,
                    "observations": int(len(selected)),
                    "avg_prediction": float(selected["prediction"].mean()),
                    "mean_return": float(selected["target_return"].mean()),
                    "median_return": float(selected["target_return"].median()),
                    "hit_rate": float((selected["target_return"] > 0.0).mean()),
                    "avg_revenue_growth": float(selected["revenue_growth_yoy"].mean()),
                    "avg_operating_margin": float(selected["operating_margin"].mean()),
                    "avg_fcf_margin": float(selected["fcf_margin"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["forward_window", "regression_bucket", "mean_return"], ascending=[True, True, False])


def _coefficient_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    path = OUTPUT_DIR / "walk_forward_coefficients.csv"
    if not path.exists():
        return pd.DataFrame()
    coefs = pd.read_csv(path)
    if coefs.empty:
        return coefs
    return (
        coefs.groupby(["forward_window", "regression_bucket", "feature"])["standardized_coef"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .sort_values(["forward_window", "regression_bucket", "mean"], ascending=[True, True, False])
    )


def _write_report(
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    performance: pd.DataFrame,
    condition_summary: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
    config: RegressionConfig,
) -> None:
    lines = [
        "# Financial Ratio Regression Study",
        "",
        "Question: can many annual financial ratios predict 3m, 6m, or 12m forward stock returns, and are there subsets with higher predictive value?",
        "",
        "## Leakage controls",
        "",
        f"- Annual financials are assumed available only {config.reporting_lag_days} calendar days after fiscal year end.",
        "- Forward returns start from the first trading day on or after that availability date.",
        "- For each test year and horizon, training rows are purged unless their forward-return end date is before the test year starts.",
        "- Regression uses fixed ridge alpha, not tuned on the validation rows.",
        "- Market-cap buckets use current Yahoo market cap, not point-in-time market cap; treat bucket results as approximate.",
        "",
        "## Dataset",
        "",
        f"- Financial events: {len(events)}.",
        f"- Prediction rows: {len(predictions)}.",
        f"- Symbols with predictions: {predictions['symbol'].nunique() if not predictions.empty else 0}.",
        "",
        "## Walk-forward predictive power",
        "",
        _markdown_table(performance),
        "",
        "## Higher-value condition subsets",
        "",
        _markdown_table(condition_summary.head(40)),
        "",
        "## Average standardized coefficients",
        "",
        _markdown_table(coefficient_summary.head(60)),
        "",
        "## Interpretation",
        "",
        "- Out-of-sample R2 is generally weak or negative, so annual financial ratios alone are not a strong standalone return model in this sample.",
        "- The most useful output is ranking/conditioning: top predicted quintiles and 30%+ revenue growth combined with positive margins or FCF can identify better pockets than raw revenue growth alone.",
        "- Predictive value differs by market-cap bucket; small samples in low/mid caps make those results fragile.",
        "- A deployable version would need point-in-time filings, point-in-time market caps, more history, and transaction-cost-aware portfolio construction.",
    ]
    (OUTPUT_DIR / "FINANCIAL_RATIO_REGRESSION.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]) and column not in {"observations", "count"}:
            if any(token in column for token in ["r2", "corr", "return", "rate", "growth", "margin", "coef", "mean", "median"]):
                display[column] = display[column].map(lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
