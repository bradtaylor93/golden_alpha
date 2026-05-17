"""Quarterly fundamentals regression with annual context.

This expands the annual fundamentals experiment with quarterly statements:

- quarterly raw ratios,
- QoQ and YoY quarterly deltas,
- latest available annual raw/delta context,
- purged walk-forward tests by calendar quarter.

Yahoo Finance exposes only recent quarterly history for many names, so this is
mainly a cross-sectional sample-size expansion rather than a deep-history test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from ath_reversion_research import download_yahoo_ohlcv, symbols_for


OUTPUT_DIR = Path("ath_reversion_research/reports/quarterly_fundamental_regression")
EXPANDED_SYMBOLS = Path("ath_reversion_research/reports/expanded_stock_signals/downloaded_symbols.csv")
ANNUAL_EVENTS = Path("ath_reversion_research/reports/financial_ratio_regression/financial_ratio_events.csv")
QUARTERLY_LAG_DAYS = 45
FORWARD_WINDOWS = {
    "fwd_1m_return": 21,
    "fwd_3m_return": 63,
    "fwd_6m_return": 126,
    "fwd_12m_return": 252,
}

RAW_FEATURES = [
    "q_revenue_growth_yoy",
    "q_revenue_growth_qoq",
    "q_revenue_growth_accel",
    "q_gross_margin",
    "q_operating_margin",
    "q_net_margin",
    "q_ebitda_margin",
    "q_cfo_margin",
    "q_fcf_margin",
    "q_capex_to_revenue",
    "q_rd_to_revenue",
    "q_sga_to_revenue",
    "q_debt_to_assets",
    "q_debt_to_equity",
    "q_cash_to_assets",
    "q_current_ratio",
    "q_asset_turnover",
]
DELTA_FEATURES = [f"{feature}_qoq_delta" for feature in RAW_FEATURES if feature.startswith("q_") and "growth" not in feature]
ANNUAL_CONTEXT_FEATURES = [
    "annual_revenue_growth_yoy",
    "annual_revenue_growth_accel",
    "annual_gross_margin",
    "annual_operating_margin",
    "annual_net_margin",
    "annual_fcf_margin",
    "annual_debt_to_assets",
    "annual_asset_turnover",
]
FEATURE_SETS = {
    "quarterly_raw": RAW_FEATURES,
    "quarterly_raw_delta": RAW_FEATURES + DELTA_FEATURES,
    "quarterly_plus_annual": RAW_FEATURES + DELTA_FEATURES + ANNUAL_CONTEXT_FEATURES,
}


@dataclass(frozen=True)
class Config:
    start: str = "2023-01-01"
    quarterly_lag_days: int = QUARTERLY_LAG_DAYS
    min_revenue: float = 25_000_000.0
    min_train_rows: int = 60
    ridge_alpha: float = 10.0


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = Config()
    symbols = _load_symbols()
    fundamentals, caps = _download_quarterly_fundamentals(symbols)
    fundamentals.to_csv(OUTPUT_DIR / "quarterly_fundamentals_raw.csv", index=False)
    caps.to_csv(OUTPUT_DIR / "current_market_caps.csv", index=False)

    features = _build_quarterly_features(fundamentals, caps, config)
    if ANNUAL_EVENTS.exists():
        features = _attach_annual_context(features, pd.read_csv(ANNUAL_EVENTS, parse_dates=["availability_date"]))
    prices = download_yahoo_ohlcv(symbols, start=config.start, chunk_size=25)
    events = _attach_forward_returns(features, prices, config.quarterly_lag_days)
    events.to_csv(OUTPUT_DIR / "quarterly_feature_events.csv", index=False)

    preds = _walk_forward_predictions(events, config)
    preds.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    perf = _performance_summary(preds)
    cond = _condition_summary(preds)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    cond.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False)
    _write_report(events, preds, perf, cond, config)

    print("Performance")
    print(perf.to_string(index=False))
    print("\nConditions")
    print(cond.head(30).to_string(index=False))
    return 0


def _load_symbols() -> list[str]:
    symbols: list[str] = []
    if EXPANDED_SYMBOLS.exists():
        frame = pd.read_csv(EXPANDED_SYMBOLS)
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


def _download_quarterly_fundamentals(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    caps: list[dict[str, object]] = []
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        try:
            market_cap = _safe_market_cap(ticker)
            caps.append({"symbol": symbol, "current_market_cap": market_cap, "market_cap_bucket": _cap_bucket(market_cap)})
            income = ticker.quarterly_income_stmt
            balance = ticker.quarterly_balance_sheet
            cashflow = ticker.quarterly_cashflow
        except Exception as exc:
            rows.append({"symbol": symbol, "error": str(exc)})
            continue
        if income is None or income.empty or "Total Revenue" not in income.index:
            rows.append({"symbol": symbol, "error": "missing_quarterly_revenue"})
            continue
        fiscal_dates = sorted(pd.to_datetime(income.columns))
        for fiscal_date in fiscal_dates:
            rows.append(
                {
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
            )
    return pd.DataFrame(rows).sort_values(["symbol", "fiscal_period_end"]).reset_index(drop=True), pd.DataFrame(caps).drop_duplicates("symbol")


def _safe_market_cap(ticker: yf.Ticker) -> float:
    try:
        value = ticker.fast_info.get("market_cap")
        if value:
            return float(value)
    except Exception:
        pass
    try:
        value = ticker.info.get("marketCap")
        if value:
            return float(value)
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
    columns = pd.to_datetime(statement.columns)
    for label in labels:
        if label in statement.index and fiscal_date in columns:
            value = statement.loc[label, fiscal_date]
            if pd.notna(value):
                return float(value)
    return np.nan


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _build_quarterly_features(fundamentals: pd.DataFrame, caps: pd.DataFrame, config: Config) -> pd.DataFrame:
    frame = fundamentals[(fundamentals["error"].fillna("") == "") & fundamentals["total_revenue"].notna()].copy()
    frame["fiscal_period_end"] = pd.to_datetime(frame["fiscal_period_end"])
    frame = frame.sort_values(["symbol", "fiscal_period_end"])
    grouped = frame.groupby("symbol")
    frame["prior_q_revenue"] = grouped["total_revenue"].shift(1)
    frame["prior_y_revenue"] = grouped["total_revenue"].shift(4)
    frame["prior_y_growth"] = grouped["total_revenue"].pct_change(4).shift(1)
    frame["q_revenue_growth_qoq"] = frame["total_revenue"] / frame["prior_q_revenue"] - 1.0
    frame["q_revenue_growth_yoy"] = frame["total_revenue"] / frame["prior_y_revenue"] - 1.0
    frame["q_revenue_growth_accel"] = frame["q_revenue_growth_yoy"] - frame["prior_y_growth"]
    frame["q_gross_margin"] = _safe_div(frame["gross_profit"], frame["total_revenue"])
    frame["q_operating_margin"] = _safe_div(frame["operating_income"], frame["total_revenue"])
    frame["q_net_margin"] = _safe_div(frame["net_income"], frame["total_revenue"])
    frame["q_ebitda_margin"] = _safe_div(frame["ebitda"], frame["total_revenue"])
    frame["q_cfo_margin"] = _safe_div(frame["operating_cash_flow"], frame["total_revenue"])
    fcf = frame["free_cash_flow"].where(frame["free_cash_flow"].notna(), frame["operating_cash_flow"] + frame["capital_expenditure"])
    frame["q_fcf_margin"] = _safe_div(fcf, frame["total_revenue"])
    frame["q_capex_to_revenue"] = _safe_div(frame["capital_expenditure"].abs(), frame["total_revenue"])
    frame["q_rd_to_revenue"] = _safe_div(frame["research_development"].fillna(0.0), frame["total_revenue"])
    frame["q_sga_to_revenue"] = _safe_div(frame["selling_general_admin"].fillna(0.0), frame["total_revenue"])
    frame["q_debt_to_assets"] = _safe_div(frame["total_debt"], frame["total_assets"])
    frame["q_debt_to_equity"] = _safe_div(frame["total_debt"], frame["stockholders_equity"].abs())
    frame["q_cash_to_assets"] = _safe_div(frame["cash"], frame["total_assets"])
    frame["q_current_ratio"] = _safe_div(frame["current_assets"], frame["current_liabilities"])
    frame["q_asset_turnover"] = _safe_div(frame["total_revenue"], frame["total_assets"])
    for feature in [f for f in RAW_FEATURES if "growth" not in f]:
        frame[f"{feature}_qoq_delta"] = grouped[feature].diff()
    frame["availability_date"] = frame["fiscal_period_end"] + pd.Timedelta(days=config.quarterly_lag_days)
    frame = frame.merge(caps, on="symbol", how="left")
    frame = frame[(frame["total_revenue"] >= config.min_revenue) & (frame["availability_date"] >= pd.Timestamp(config.start))].copy()
    return frame.reset_index(drop=True)


def _attach_annual_context(quarterly: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "revenue_growth_yoy": "annual_revenue_growth_yoy",
        "revenue_growth_accel": "annual_revenue_growth_accel",
        "gross_margin": "annual_gross_margin",
        "operating_margin": "annual_operating_margin",
        "net_margin": "annual_net_margin",
        "fcf_margin": "annual_fcf_margin",
        "debt_to_assets": "annual_debt_to_assets",
        "asset_turnover": "annual_asset_turnover",
    }
    annual_cols = ["symbol", "availability_date", *mapping.keys()]
    annual_context = annual[annual_cols].rename(columns=mapping).sort_values(["symbol", "availability_date"])
    pieces = []
    for symbol, group in quarterly.sort_values(["symbol", "availability_date"]).groupby("symbol"):
        annual_group = annual_context[annual_context["symbol"] == symbol].drop(columns=["symbol"])
        if annual_group.empty:
            pieces.append(group)
            continue
        merged = pd.merge_asof(
            group.sort_values("availability_date"),
            annual_group.sort_values("availability_date"),
            on="availability_date",
            direction="backward",
        )
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def _attach_forward_returns(events: pd.DataFrame, prices: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    price_map = {symbol: group.sort_values("date").reset_index(drop=True) for symbol, group in prices.groupby("symbol")}
    rows = []
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
        row["reporting_lag_days"] = lag_days
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


def _walk_forward_predictions(events: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows = []
    for feature_set, features in FEATURE_SETS.items():
        for target in FORWARD_WINDOWS:
            end_col = f"{target}_end_date"
            for bucket in ["all", "large_cap", "mid_cap", "low_cap"]:
                subset = events.copy() if bucket == "all" else events[events["market_cap_bucket"] == bucket].copy()
                subset = subset.dropna(subset=[target, end_col, "q_revenue_growth_qoq"])
                if subset.empty:
                    continue
                for test_period in sorted(pd.PeriodIndex(pd.to_datetime(subset["trade_date"]).dt.to_period("Q")).unique()):
                    period_start = test_period.start_time
                    period_end = test_period.end_time
                    test = subset[(pd.to_datetime(subset["trade_date"]) >= period_start) & (pd.to_datetime(subset["trade_date"]) <= period_end)].copy()
                    train = subset[(pd.to_datetime(subset["trade_date"]) < period_start) & (pd.to_datetime(subset[end_col]) < period_start)].copy()
                    if len(train) < config.min_train_rows or test.empty:
                        continue
                    pred = _fit_predict(train, test, target, features, config.ridge_alpha)
                    test["prediction"] = pred
                    test["target_return"] = test[target]
                    test["feature_set"] = feature_set
                    test["forward_window"] = target
                    test["regression_bucket"] = bucket
                    test["test_period"] = str(test_period)
                    rows.append(test)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str], alpha: float) -> np.ndarray:
    usable = [
        feature
        for feature in features
        if feature in train and train[feature].notna().mean() >= 0.40 and train[feature].astype(float).std(skipna=True) > 0.0
    ]
    if not usable:
        usable = ["q_revenue_growth_qoq"]
    low = train[usable].quantile(0.01)
    high = train[usable].quantile(0.99)
    median = train[usable].median()
    x_train = train[usable].astype(float).clip(lower=low, upper=high, axis=1).fillna(median)
    x_test = test[usable].astype(float).clip(lower=low, upper=high, axis=1).fillna(median)
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
                "test_periods": int(group["test_period"].nunique()),
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
            "q_yoy_growth_30pct": group["q_revenue_growth_yoy"] >= 0.30,
            "q_yoy_growth_30pct_positive_margin": (group["q_revenue_growth_yoy"] >= 0.30) & (group["q_operating_margin"] > 0),
            "q_yoy_growth_30pct_model_top_half": (group["q_revenue_growth_yoy"] >= 0.30) & (group["prediction"] >= group["prediction"].median()),
            "annual_growth_30pct_q_positive": (group.get("annual_revenue_growth_yoy", pd.Series(index=group.index, dtype=float)) >= 0.30) & (group["q_revenue_growth_yoy"] > 0),
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
                    "hit_rate": float((selected["target_return"] > 0).mean()),
                    "avg_q_yoy_growth": float(selected["q_revenue_growth_yoy"].mean()),
                    "avg_q_qoq_growth": float(selected["q_revenue_growth_qoq"].mean()),
                    "avg_q_operating_margin": float(selected["q_operating_margin"].mean()),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_set",
                "forward_window",
                "regression_bucket",
                "condition",
                "observations",
                "mean_return",
                "median_return",
                "hit_rate",
                "avg_q_yoy_growth",
                "avg_q_qoq_growth",
                "avg_q_operating_margin",
            ]
        )
    return pd.DataFrame(rows).sort_values(["forward_window", "regression_bucket", "mean_return"], ascending=[True, True, False])


def _write_report(events: pd.DataFrame, predictions: pd.DataFrame, perf: pd.DataFrame, cond: pd.DataFrame, config: Config) -> None:
    lines = [
        "# Quarterly Fundamental Regression Study",
        "",
        "This expands the fundamentals sample with quarterly statements and tests raw quarterly ratios, QoQ/YoY deltas, and latest annual context.",
        "",
        "## Leakage controls",
        "",
        f"- Quarterly fundamentals are assumed available only {config.quarterly_lag_days} calendar days after quarter end.",
        "- Forward returns start from the first trading day on or after that availability date.",
        "- Tests are run by calendar quarter; training rows are purged unless their forward-return end date is before the test quarter starts.",
        "- Model hyperparameters are fixed.",
        "",
        "## Dataset",
        "",
        f"- Quarterly events: {len(events)}.",
        f"- Prediction rows: {len(predictions)}.",
        f"- Symbols with predictions: {predictions['symbol'].nunique() if not predictions.empty else 0}.",
        f"- Test periods with predictions: {predictions['test_period'].nunique() if not predictions.empty else 0}.",
        "- Yahoo Finance quarterly history is shallow; 3m/6m/12m quarterly tests may have few completed test periods.",
        "",
        "## Predictive power by feature set",
        "",
        _markdown_table(perf),
        "",
        "## Conditional subsets",
        "",
        _markdown_table(cond.head(60)),
        "",
        "## Interpretation",
        "",
        "- Quarterly data increases event count and creates more test periods, but Yahoo's quarterly history is still shallow.",
        "- The quarterly-plus-annual feature set should be judged by OOS rank/correlation and top-minus-bottom spreads, not R2 alone.",
        "- Any low/mid-cap results with small row counts remain fragile.",
    ]
    (OUTPUT_DIR / "QUARTERLY_FUNDAMENTAL_REGRESSION.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]) and column not in {"observations", "test_periods"}:
            if any(token in column for token in ["r2", "corr", "return", "rate", "growth", "margin", "mean", "median"]):
                display[column] = display[column].map(lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
