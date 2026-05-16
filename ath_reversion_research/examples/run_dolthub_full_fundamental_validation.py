"""Full local-Dolt annual fundamentals validation.

Requires local clones:

    /workspace/dolthub_data/earnings
    /workspace/dolthub_data/stocks

It uses DoltHub fundamentals (income statement, balance sheet, cash flow) and
DoltHub OHLCV locally, avoiding the web API timeout encountered for long price
queries.  The validation remains conservative:

- annual financials available 90 days after fiscal year end,
- returns start after availability,
- training rows are purged if their forward window overlaps the test year.
"""

from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import run_financial_ratio_regression as annual
import run_sp500_annual_fundamental_regression as sp500
from ath_reversion_research import download_yahoo_ohlcv


OUTPUT_DIR = Path("ath_reversion_research/reports/dolthub_full_fundamental_validation")
EARNINGS_DIR = Path("/workspace/dolthub_data/earnings")
STOCKS_DIR = Path("/workspace/dolthub_data/stocks")
DOLT = Path("/workspace/.local/bin/dolt")
REPORTING_LAG_DAYS = 90
FORWARD_WINDOWS = {"fwd_3m_return": 63, "fwd_6m_return": 126, "fwd_12m_return": 252}
FEATURES = annual.FEATURES + [
    "trailing_3m_return",
    "trailing_6m_return",
    "trailing_12m_return",
    "relative_6m_vs_spy",
    "realized_vol_3m",
    "drawdown_12m",
    "growth_x_trailing_6m",
    "growth_x_fcf_margin",
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _require_local_dolt()
    symbols = sp500._download_sp500_symbols()
    pd.DataFrame({"symbol": symbols}).to_csv(OUTPUT_DIR / "sp500_symbols.csv", index=False)
    fundamentals = _load_fundamentals(symbols)
    fundamentals.to_csv(OUTPUT_DIR / "dolt_full_fundamentals_raw.csv", index=False)
    events = _build_events(fundamentals)
    # Local Dolt OHLCV queries over the full stocks table are currently slow
    # without a secondary index on act_symbol/date, so this pass uses local Dolt
    # fundamentals with Yahoo adjusted prices.  The script still keeps the local
    # price loader below for smaller universes or indexed local clones.
    prices = download_yahoo_ohlcv(symbols + ["SPY"], start="2012-01-01", chunk_size=25)
    prices.to_csv(OUTPUT_DIR / "price_subset_yahoo.csv", index=False)
    events = _attach_prices(events, prices)
    events.to_csv(OUTPUT_DIR / "dolt_full_feature_events.csv", index=False)
    preds = _walk_forward(events)
    preds.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    perf = _performance(preds)
    cond = _conditions(preds)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    cond.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False)
    _write_report(events, preds, perf, cond)
    print(perf.to_string(index=False))
    return 0


def _require_local_dolt() -> None:
    for path in [DOLT, EARNINGS_DIR, STOCKS_DIR]:
        if not path.exists():
            raise SystemExit(f"Missing required local Dolt path: {path}")


def _dolt_query(db_dir: Path, query: str) -> pd.DataFrame:
    result = subprocess.run(
        [str(DOLT), "sql", "-r", "csv", "-q", query],
        cwd=str(db_dir),
        text=True,
        capture_output=True,
        check=True,
    )
    if not result.stdout.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(result.stdout))


def _chunks(items: list[str], size: int = 80) -> list[list[str]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _load_fundamentals(symbols: list[str]) -> pd.DataFrame:
    frames = []
    for chunk in _chunks(symbols, 80):
        in_list = ", ".join(f"'{symbol}'" for symbol in chunk)
        query = f"""
        SELECT
            i.act_symbol AS symbol,
            i.date AS fiscal_period_end,
            i.sales,
            i.gross_profit,
            i.income_after_depreciation_and_amortization AS operating_income,
            i.net_income,
            i.income_before_depreciation_and_amortization AS ebitda,
            a.cash_and_equivalents,
            a.total_current_assets,
            a.total_assets,
            l.total_current_liabilities,
            l.long_term_debt,
            l.total_liabilities,
            e.total_equity,
            c.net_cash_from_operating_activities,
            c.property_and_equipment
        FROM income_statement i
        LEFT JOIN balance_sheet_assets a
          ON i.act_symbol=a.act_symbol AND i.date=a.date AND i.period=a.period
        LEFT JOIN balance_sheet_liabilities l
          ON i.act_symbol=l.act_symbol AND i.date=l.date AND i.period=l.period
        LEFT JOIN balance_sheet_equity e
          ON i.act_symbol=e.act_symbol AND i.date=e.date AND i.period=e.period
        LEFT JOIN cash_flow_statement c
          ON i.act_symbol=c.act_symbol AND i.date=c.date AND i.period=c.period
        WHERE i.period='Year' AND i.act_symbol IN ({in_list})
        ORDER BY i.act_symbol, i.date;
        """
        frame = _dolt_query(EARNINGS_DIR, query)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No local Dolt fundamentals returned")
    out = pd.concat(frames, ignore_index=True)
    for col in out.columns:
        if col not in {"symbol", "fiscal_period_end"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["fiscal_period_end"] = pd.to_datetime(out["fiscal_period_end"])
    return out.sort_values(["symbol", "fiscal_period_end"]).reset_index(drop=True)


def _build_events(fundamentals: pd.DataFrame) -> pd.DataFrame:
    frame = fundamentals.copy().sort_values(["symbol", "fiscal_period_end"])
    grouped = frame.groupby("symbol")
    frame["prior_revenue"] = grouped["sales"].shift(1)
    frame["prior_revenue_growth_yoy"] = grouped["sales"].pct_change().shift(1)
    frame["revenue_growth_yoy"] = frame["sales"] / frame["prior_revenue"] - 1
    frame["revenue_growth_accel"] = frame["revenue_growth_yoy"] - frame["prior_revenue_growth_yoy"]
    frame["gross_margin"] = frame["gross_profit"] / frame["sales"].replace(0, np.nan)
    frame["operating_margin"] = frame["operating_income"] / frame["sales"].replace(0, np.nan)
    frame["net_margin"] = frame["net_income"] / frame["sales"].replace(0, np.nan)
    frame["ebitda_margin"] = frame["ebitda"] / frame["sales"].replace(0, np.nan)
    frame["cfo_margin"] = frame["net_cash_from_operating_activities"] / frame["sales"].replace(0, np.nan)
    frame["fcf_margin"] = (frame["net_cash_from_operating_activities"] + frame["property_and_equipment"]) / frame["sales"].replace(0, np.nan)
    frame["capex_to_revenue"] = frame["property_and_equipment"].abs() / frame["sales"].replace(0, np.nan)
    frame["rd_to_revenue"] = 0.0
    frame["sga_to_revenue"] = np.nan
    debt = frame["long_term_debt"].fillna(0.0)
    frame["debt_to_assets"] = debt / frame["total_assets"].replace(0, np.nan)
    frame["debt_to_equity"] = debt / frame["total_equity"].abs().replace(0, np.nan)
    frame["cash_to_assets"] = frame["cash_and_equivalents"] / frame["total_assets"].replace(0, np.nan)
    frame["current_ratio"] = frame["total_current_assets"] / frame["total_current_liabilities"].replace(0, np.nan)
    frame["asset_turnover"] = frame["sales"] / frame["total_assets"].replace(0, np.nan)
    frame["availability_date"] = frame["fiscal_period_end"] + pd.Timedelta(days=REPORTING_LAG_DAYS)
    return frame[(frame["prior_revenue"] > 100_000_000) & frame["revenue_growth_yoy"].notna()].reset_index(drop=True)


def _load_prices(symbols: list[str]) -> pd.DataFrame:
    frames = []
    for chunk in _chunks(sorted(set(symbols)), 80):
        in_list = ", ".join(f"'{symbol}'" for symbol in chunk)
        query = f"""
        SELECT act_symbol AS symbol, date, close
        FROM ohlcv
        WHERE act_symbol IN ({in_list}) AND date >= '2012-01-01'
        ORDER BY act_symbol, date;
        """
        frame = _dolt_query(STOCKS_DIR, query)
        if not frame.empty:
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["close"]).sort_values(["symbol", "date"]).reset_index(drop=True)


def _attach_prices(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
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
        spy_pos = int(spy.index.searchsorted(trade_date, side="left"))
        if pos >= len(series) or pos < 252 or spy_pos < 126:
            continue
        current = float(series.iloc[pos])
        row = dict(event)
        row["trade_date"] = series.index[pos]
        row["trade_close"] = current
        row["trailing_3m_return"] = current / series.iloc[pos - 63] - 1
        row["trailing_6m_return"] = current / series.iloc[pos - 126] - 1
        row["trailing_12m_return"] = current / series.iloc[pos - 252] - 1
        row["relative_6m_vs_spy"] = row["trailing_6m_return"] - (spy.iloc[spy_pos] / spy.iloc[spy_pos - 126] - 1)
        row["realized_vol_3m"] = series.pct_change().iloc[pos - 62 : pos + 1].std() * np.sqrt(252)
        row["drawdown_12m"] = current / series.iloc[pos - 251 : pos + 1].max() - 1
        row["growth_x_trailing_6m"] = row["revenue_growth_yoy"] * row["trailing_6m_return"]
        row["growth_x_fcf_margin"] = row["revenue_growth_yoy"] * row["fcf_margin"]
        for name, bars in FORWARD_WINDOWS.items():
            end = pos + bars
            if end < len(series):
                row[name] = series.iloc[end] / current - 1
                row[f"{name}_end_date"] = series.index[end]
            else:
                row[name] = np.nan
                row[f"{name}_end_date"] = pd.NaT
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _walk_forward(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in FORWARD_WINDOWS:
        end_col = f"{target}_end_date"
        subset = events.dropna(subset=[target, end_col, "revenue_growth_yoy"]).copy()
        for year in sorted(subset["trade_date"].dt.year.unique()):
            start = pd.Timestamp(f"{int(year)}-01-01")
            test = subset[subset["trade_date"].dt.year == year].copy()
            train = subset[(subset["trade_date"] < start) & (subset[end_col] < start)].copy()
            if len(train) < 100 or test.empty:
                continue
            test["prediction"] = _fit_predict(train, test, target)
            test["target_return"] = test[target]
            test["forward_window"] = target
            test["test_year"] = int(year)
            rows.append(test)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    usable = [f for f in FEATURES if f in train and train[f].notna().mean() >= 0.5 and train[f].std(skipna=True) > 0]
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


def _performance(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in predictions.groupby("forward_window"):
        baseline = ((group["target_return"] - group["target_return"].mean()) ** 2).sum()
        residual = ((group["target_return"] - group["prediction"]) ** 2).sum()
        top = group[group["prediction"] >= group["prediction"].quantile(0.8)]
        bottom = group[group["prediction"] <= group["prediction"].quantile(0.2)]
        rows.append({
            "forward_window": target,
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
    return pd.DataFrame(rows).sort_values("forward_window")


def _conditions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in predictions.groupby("forward_window"):
        for condition, mask in {
            "model_top_quintile": group["prediction"] >= group["prediction"].quantile(0.8),
            "revenue_growth_30pct": group["revenue_growth_yoy"] >= 0.3,
            "revenue_growth_30pct_top_half": (group["revenue_growth_yoy"] >= 0.3) & (group["prediction"] >= group["prediction"].median()),
        }.items():
            selected = group[mask]
            if len(selected) < 10:
                continue
            rows.append({
                "forward_window": target,
                "condition": condition,
                "observations": len(selected),
                "mean_return": selected["target_return"].mean(),
                "median_return": selected["target_return"].median(),
                "hit_rate": (selected["target_return"] > 0).mean(),
                "avg_revenue_growth": selected["revenue_growth_yoy"].mean(),
            })
    return pd.DataFrame(rows).sort_values(["forward_window", "mean_return"], ascending=[True, False])


def _write_report(events: pd.DataFrame, preds: pd.DataFrame, perf: pd.DataFrame, cond: pd.DataFrame) -> None:
    lines = [
        "# Full Local Dolt Fundamentals Validation",
        "",
        "Uses local DoltHub earnings and stocks clones for fundamentals and OHLCV.",
        "",
        "## Data",
        "",
        f"- Feature events: {len(events)}.",
        f"- Prediction rows: {len(preds)}.",
        f"- Symbols with predictions: {preds['symbol'].nunique() if not preds.empty else 0}.",
        f"- Test years: {sorted(preds['test_year'].unique().tolist()) if not preds.empty else []}.",
        "",
        "## Leakage controls",
        "",
        "- Annual fundamentals assumed available 90 days after fiscal year end.",
        "- Forward returns start after availability.",
        "- Training rows are purged unless their forward-return end date is before the test year starts.",
        "- Uses local Dolt fundamentals; this run uses Yahoo adjusted prices because local Dolt OHLCV needs indexing/export for this universe size.",
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
        "- This is the closest open-data validation pass so far for fundamentals: local Dolt income, balance sheet, and cash flow.",
        "- It includes balance sheet and cash-flow features, but still uses current S&P 500 symbols.",
        "- If 12m predictive power remains positive, the next step is historical constituents/delisted universe construction.",
    ]
    (OUTPUT_DIR / "DOLTHUB_FULL_FUNDAMENTAL_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "test_years"}:
            if any(token in col for token in ["r2", "corr", "return", "rate", "growth"]):
                display[col] = display[col].map(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
