"""DoltHub annual income-statement validation.

This is a first DoltHub validation pass.  It uses:

- DoltHub `post-no-preference/earnings.income_statement` for longer annual
  income-statement history,
- Yahoo adjusted prices for returns because long DoltHub OHLCV API queries time
  out in this environment.

The study keeps the no-lookahead mechanics: annual statements are assumed
available 90 days after fiscal year end and training rows are purged if their
forward-return window overlaps the test year.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ath_reversion_research import download_yahoo_ohlcv


OUTPUT_DIR = Path("ath_reversion_research/reports/dolthub_annual_income_validation")
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DOLTHUB_EARNINGS_API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/earnings/master"
REPORTING_LAG_DAYS = 90
FORWARD_WINDOWS = {"fwd_3m_return": 63, "fwd_6m_return": 126, "fwd_12m_return": 252}
FEATURES = [
    "sales_growth_yoy",
    "sales_growth_accel",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "ebitda_margin",
    "trailing_3m_return",
    "trailing_6m_return",
    "trailing_12m_return",
    "relative_6m_vs_spy",
    "realized_vol_3m",
    "drawdown_12m",
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = _download_sp500_symbols()
    fundamentals = _download_income_statements(symbols)
    fundamentals.to_csv(OUTPUT_DIR / "dolthub_income_statement_raw.csv", index=False)
    events = _build_events(fundamentals)
    prices = download_yahoo_ohlcv(sorted(set(symbols) | {"SPY"}), start="2012-01-01", chunk_size=25)
    events = _attach_price_features_and_returns(events, prices)
    events.to_csv(OUTPUT_DIR / "dolthub_income_events.csv", index=False)
    preds = _walk_forward(events)
    preds.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    perf = _performance(preds)
    cond = _conditions(preds)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    cond.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False)
    _write_report(symbols, events, preds, perf, cond)
    print(perf.to_string(index=False))
    return 0


def _download_sp500_symbols() -> list[str]:
    response = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text))[0]
    return sorted(set(table["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper()))


def _query_dolthub(query: str) -> pd.DataFrame:
    response = requests.get(DOLTHUB_EARNINGS_API, params={"q": query}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("rows", [])
    if payload.get("query_execution_status") not in {"Success", "RowLimit"}:
        message = payload.get("query_execution_message")
        if rows:
            print(f"DoltHub returned partial rows with status={payload.get('query_execution_status')}: {message}")
        else:
            raise RuntimeError(f"DoltHub query failed: {message}")
    return pd.DataFrame(rows)


def _download_income_statements(symbols: list[str], chunk_size: int = 50) -> pd.DataFrame:
    frames = []
    columns = [
        "act_symbol",
        "date",
        "period",
        "sales",
        "gross_profit",
        "income_after_depreciation_and_amortization",
        "net_income",
        "income_before_depreciation_and_amortization",
        "diluted_net_eps",
    ]
    for start in range(0, len(symbols), chunk_size):
        chunk = symbols[start : start + chunk_size]
        in_list = ", ".join(f"'{symbol}'" for symbol in chunk)
        query = (
            "SELECT "
            + ", ".join(columns)
            + " FROM income_statement "
            + f"WHERE period='Year' AND act_symbol IN ({in_list}) "
            + "ORDER BY act_symbol, date;"
        )
        frame = _query_dolthub(query)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No DoltHub income statement rows returned")
    out = pd.concat(frames, ignore_index=True)
    for column in columns:
        if column not in out:
            out[column] = np.nan
    return out.loc[:, columns].sort_values(["act_symbol", "date"]).reset_index(drop=True)


def _build_events(fundamentals: pd.DataFrame) -> pd.DataFrame:
    frame = fundamentals.rename(columns={"act_symbol": "symbol", "date": "fiscal_period_end"}).copy()
    frame["fiscal_period_end"] = pd.to_datetime(frame["fiscal_period_end"])
    numeric_cols = [c for c in frame.columns if c not in {"symbol", "fiscal_period_end", "period"}]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["symbol", "fiscal_period_end"])
    grouped = frame.groupby("symbol")
    frame["prior_sales"] = grouped["sales"].shift(1)
    frame["prior_sales_growth_yoy"] = grouped["sales"].pct_change().shift(1)
    frame["sales_growth_yoy"] = frame["sales"] / frame["prior_sales"] - 1.0
    frame["sales_growth_accel"] = frame["sales_growth_yoy"] - frame["prior_sales_growth_yoy"]
    frame["gross_margin"] = frame["gross_profit"] / frame["sales"].replace(0, np.nan)
    frame["operating_margin"] = frame["income_after_depreciation_and_amortization"] / frame["sales"].replace(0, np.nan)
    frame["net_margin"] = frame["net_income"] / frame["sales"].replace(0, np.nan)
    frame["ebitda_margin"] = frame["income_before_depreciation_and_amortization"] / frame["sales"].replace(0, np.nan)
    frame["availability_date"] = frame["fiscal_period_end"] + pd.Timedelta(days=REPORTING_LAG_DAYS)
    frame = frame[(frame["prior_sales"] > 100_000_000) & frame["sales_growth_yoy"].notna()].copy()
    return frame.reset_index(drop=True)


def _attach_price_features_and_returns(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    spy = close["SPY"]
    rows = []
    for event in events.to_dict("records"):
        symbol = str(event["symbol"])
        if symbol not in close:
            continue
        trade_date = pd.Timestamp(event["availability_date"])
        series = close[symbol].dropna()
        pos = int(series.index.searchsorted(trade_date, side="left"))
        if pos >= len(series) or pos < 252:
            continue
        spy_pos = int(spy.index.searchsorted(series.index[pos], side="right")) - 1
        if spy_pos < 126:
            continue
        row = dict(event)
        current = float(series.iloc[pos])
        row["trade_date"] = series.index[pos]
        row["trade_close"] = current
        row["trailing_3m_return"] = float(current / series.iloc[pos - 63] - 1.0)
        row["trailing_6m_return"] = float(current / series.iloc[pos - 126] - 1.0)
        row["trailing_12m_return"] = float(current / series.iloc[pos - 252] - 1.0)
        spy_6m = float(spy.iloc[spy_pos] / spy.iloc[spy_pos - 126] - 1.0)
        row["relative_6m_vs_spy"] = row["trailing_6m_return"] - spy_6m
        row["realized_vol_3m"] = float(series.pct_change().iloc[pos - 62 : pos + 1].std() * np.sqrt(252))
        row["drawdown_12m"] = float(current / series.iloc[pos - 251 : pos + 1].max() - 1.0)
        for name, bars in FORWARD_WINDOWS.items():
            end_pos = pos + bars
            if end_pos < len(series):
                row[name] = float(series.iloc[end_pos] / current - 1.0)
                row[f"{name}_end_date"] = series.index[end_pos]
            else:
                row[name] = np.nan
                row[f"{name}_end_date"] = pd.NaT
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _walk_forward(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in FORWARD_WINDOWS:
        end_col = f"{target}_end_date"
        subset = events.dropna(subset=[target, end_col, "sales_growth_yoy"]).copy()
        for year in sorted(subset["trade_date"].dt.year.unique()):
            test_start = pd.Timestamp(f"{int(year)}-01-01")
            test = subset[subset["trade_date"].dt.year == year].copy()
            train = subset[(subset["trade_date"] < test_start) & (subset[end_col] < test_start)].copy()
            if len(train) < 100 or test.empty:
                continue
            test["prediction"] = _fit_predict(train, test, target)
            test["target_return"] = test[target]
            test["forward_window"] = target
            test["test_year"] = int(year)
            rows.append(test)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, alpha: float = 10.0) -> np.ndarray:
    usable = [f for f in FEATURES if train[f].notna().mean() >= 0.50 and train[f].std(skipna=True) > 0]
    lo = train[usable].quantile(0.01)
    hi = train[usable].quantile(0.99)
    med = train[usable].median()
    x_train = train[usable].clip(lower=lo, upper=hi, axis=1).fillna(med)
    x_test = test[usable].clip(lower=lo, upper=hi, axis=1).fillna(med)
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
    beta = np.linalg.solve(x_mat.T @ x_mat + alpha * penalty, x_mat.T @ (y - y_mean))
    beta[0] += y_mean
    return t_mat @ beta


def _performance(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in predictions.groupby("forward_window"):
        baseline = ((group["target_return"] - group["target_return"].mean()) ** 2).sum()
        residual = ((group["target_return"] - group["prediction"]) ** 2).sum()
        top = group[group["prediction"] >= group["prediction"].quantile(0.80)]
        bottom = group[group["prediction"] <= group["prediction"].quantile(0.20)]
        rows.append(
            {
                "forward_window": target,
                "observations": int(len(group)),
                "test_years": int(group["test_year"].nunique()),
                "oos_r2": float(1 - residual / baseline) if baseline > 0 else np.nan,
                "pearson_corr": float(group["prediction"].corr(group["target_return"])),
                "spearman_corr": float(group["prediction"].rank().corr(group["target_return"].rank())),
                "top_quintile_mean_return": float(top["target_return"].mean()),
                "top_quintile_median_return": float(top["target_return"].median()),
                "top_quintile_hit_rate": float((top["target_return"] > 0).mean()),
                "bottom_quintile_mean_return": float(bottom["target_return"].mean()),
                "top_minus_bottom_mean": float(top["target_return"].mean() - bottom["target_return"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("forward_window")


def _conditions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in predictions.groupby("forward_window"):
        for condition, mask in {
            "model_top_quintile": group["prediction"] >= group["prediction"].quantile(0.80),
            "sales_growth_30pct": group["sales_growth_yoy"] >= 0.30,
            "sales_growth_30pct_top_half": (group["sales_growth_yoy"] >= 0.30) & (group["prediction"] >= group["prediction"].median()),
            "sales_growth_30pct_positive_margin": (group["sales_growth_yoy"] >= 0.30) & (group["operating_margin"] > 0),
        }.items():
            selected = group[mask]
            if len(selected) < 10:
                continue
            rows.append(
                {
                    "forward_window": target,
                    "condition": condition,
                    "observations": int(len(selected)),
                    "mean_return": float(selected["target_return"].mean()),
                    "median_return": float(selected["target_return"].median()),
                    "hit_rate": float((selected["target_return"] > 0).mean()),
                    "avg_sales_growth": float(selected["sales_growth_yoy"].mean()),
                    "avg_operating_margin": float(selected["operating_margin"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["forward_window", "mean_return"], ascending=[True, False])


def _write_report(symbols: list[str], events: pd.DataFrame, preds: pd.DataFrame, perf: pd.DataFrame, cond: pd.DataFrame) -> None:
    lines = [
        "# DoltHub Annual Income Validation",
        "",
        "First validation pass using DoltHub annual income statements and Yahoo adjusted prices.",
        "",
        "## Data",
        "",
        f"- Income-statement events: {len(events)}.",
        f"- Prediction rows: {len(preds)}.",
        f"- Symbols with predictions: {preds['symbol'].nunique() if not preds.empty else 0}.",
        f"- Test years: {sorted(preds['test_year'].unique().tolist()) if not preds.empty else []}.",
        "",
        "## Leakage controls",
        "",
        "- Annual income statements are assumed available 90 days after fiscal year end.",
        "- Forward returns start after that availability date.",
        "- Training rows are purged unless their forward-return end date is before the test year starts.",
        "- DoltHub is used for fundamentals; Yahoo is still used for prices in this first pass due DoltHub OHLCV API timeouts.",
        "",
        "## Performance",
        "",
        _markdown_table(perf),
        "",
        "## Conditions",
        "",
        _markdown_table(cond),
        "",
        "## Interpretation",
        "",
        "- DoltHub materially increases annual statement history versus Yahoo.",
        "- This first pass uses income-statement features only, so it is not apples-to-apples with the richer Yahoo financial+price model.",
        "- If predictive power is positive here, the next step is cloning Dolt locally or using a PIT vendor to add balance sheet/cash-flow fields and faster price access.",
    ]
    (OUTPUT_DIR / "DOLTHUB_ANNUAL_INCOME_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "test_years"}:
            if any(token in col for token in ["r2", "corr", "return", "rate", "growth", "margin"]):
                display[col] = display[col].map(lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
