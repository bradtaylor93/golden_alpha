"""DoltHub quarterly/TTM fundamental trend model.

Tests whether richer time-series fundamentals improve prediction:

- QoQ and YoY quarterly revenue trends,
- TTM revenue/margin/FCF trends,
- margin and cash-flow rate-of-change,
- trailing price context.

Uses local Dolt `earnings` fundamentals and Yahoo adjusted prices from the
existing local cache/export where available.  Training is purged by test quarter.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research import download_yahoo_ohlcv
from ath_reversion_research.metrics import summarize_returns
import run_dolthub_full_fundamental_validation as dolt_full
import run_sp500_annual_fundamental_regression as sp500


OUTPUT_DIR = Path("ath_reversion_research/reports/dolthub_quarterly_trend_model")
PRICE_CACHE = Path("ath_reversion_research/reports/dolthub_full_fundamental_validation/price_subset_yahoo.csv")
REPORTING_LAG_DAYS = 45
FORWARD_WINDOWS = {"fwd_3m_return": 63, "fwd_6m_return": 126, "fwd_12m_return": 252}

QUARTERLY_FEATURES = [
    "q_sales_qoq",
    "q_sales_yoy",
    "q_sales_yoy_accel",
    "ttm_sales_yoy",
    "ttm_sales_yoy_accel",
    "q_gross_margin",
    "q_operating_margin",
    "q_net_margin",
    "q_cfo_margin",
    "q_fcf_margin",
    "ttm_gross_margin",
    "ttm_operating_margin",
    "ttm_net_margin",
    "ttm_cfo_margin",
    "ttm_fcf_margin",
    "q_operating_margin_yoy_change",
    "q_fcf_margin_yoy_change",
    "ttm_operating_margin_yoy_change",
    "ttm_fcf_margin_yoy_change",
    "q_cfo_to_net_income",
    "q_fcf_to_net_income",
    "q_accruals_to_assets",
    "debt_to_assets",
    "debt_to_equity",
    "cash_to_assets",
    "current_ratio",
    "asset_turnover_q",
]
PRICE_FEATURES = [
    "trailing_3m_return",
    "trailing_6m_return",
    "trailing_12m_return",
    "relative_6m_vs_spy",
    "realized_vol_3m",
    "drawdown_12m",
]
ANNUAL_CONTEXT_FEATURES = [
    "annual_revenue_growth_yoy",
    "annual_revenue_growth_accel",
    "annual_gross_margin",
    "annual_operating_margin",
    "annual_net_margin",
    "annual_ebitda_margin",
    "annual_cfo_margin",
    "annual_fcf_margin",
    "annual_debt_to_assets",
    "annual_cash_to_assets",
    "annual_asset_turnover",
]
FEATURE_SETS = {
    "quarterly_trend_only": QUARTERLY_FEATURES,
    "price_only": PRICE_FEATURES,
    "quarterly_trend_price": QUARTERLY_FEATURES + PRICE_FEATURES,
    "quarterly_annual_price": QUARTERLY_FEATURES + ANNUAL_CONTEXT_FEATURES + PRICE_FEATURES,
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sp500._download_sp500_symbols()
    fundamentals = _load_quarterly_fundamentals(symbols)
    fundamentals.to_csv(OUTPUT_DIR / "quarterly_fundamentals_raw.csv", index=False)
    features = _build_features(fundamentals)
    features = _attach_annual_context(features)
    prices = _load_prices(symbols + ["SPY"])
    events = _attach_prices(features, prices)
    events.to_csv(OUTPUT_DIR / "quarterly_trend_events.csv", index=False)
    predictions = []
    for name, feature_list in FEATURE_SETS.items():
        predictions.append(_walk_forward(events, name, feature_list))
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    perf = _performance(pred)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    _write_report(events, pred, perf)
    print(perf.to_string(index=False))
    return 0


def _load_quarterly_fundamentals(symbols: list[str]) -> pd.DataFrame:
    frames = []
    for chunk in dolt_full._chunks(symbols, 80):
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
        WHERE i.period='Quarter' AND i.act_symbol IN ({in_list})
        ORDER BY i.act_symbol, i.date;
        """
        frame = dolt_full._dolt_query(dolt_full.EARNINGS_DIR, query)
        if not frame.empty:
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    for col in out.columns:
        if col not in {"symbol", "fiscal_period_end"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["fiscal_period_end"] = pd.to_datetime(out["fiscal_period_end"])
    return out.sort_values(["symbol", "fiscal_period_end"]).reset_index(drop=True)


def _build_features(fundamentals: pd.DataFrame) -> pd.DataFrame:
    frame = fundamentals.sort_values(["symbol", "fiscal_period_end"]).copy()
    grouped = frame.groupby("symbol")
    sales = frame["sales"].replace(0, np.nan)
    frame["q_sales_qoq"] = grouped["sales"].pct_change(1)
    frame["q_sales_yoy"] = grouped["sales"].pct_change(4)
    frame["q_sales_yoy_accel"] = frame["q_sales_yoy"] - grouped["q_sales_yoy"].shift(1)
    for col, raw in [
        ("q_gross_margin", frame["gross_profit"] / sales),
        ("q_operating_margin", frame["operating_income"] / sales),
        ("q_net_margin", frame["net_income"] / sales),
        ("q_cfo_margin", frame["net_cash_from_operating_activities"] / sales),
        ("q_fcf_margin", (frame["net_cash_from_operating_activities"] + frame["property_and_equipment"]) / sales),
    ]:
        frame[col] = raw
        frame[f"{col}_yoy_change"] = grouped[col].diff(4)
    frame["ttm_sales"] = grouped["sales"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
    frame["ttm_sales_yoy"] = frame["ttm_sales"] / grouped["ttm_sales"].shift(4) - 1
    frame["ttm_sales_yoy_accel"] = frame["ttm_sales_yoy"] - grouped["ttm_sales_yoy"].shift(1)
    for source, target in [
        ("gross_profit", "ttm_gross_margin"),
        ("operating_income", "ttm_operating_margin"),
        ("net_income", "ttm_net_margin"),
        ("net_cash_from_operating_activities", "ttm_cfo_margin"),
    ]:
        ttm_value = grouped[source].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        frame[target] = ttm_value / frame["ttm_sales"].replace(0, np.nan)
        frame[f"{target}_yoy_change"] = grouped[target].diff(4)
    ttm_fcf = grouped["net_cash_from_operating_activities"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True) + grouped["property_and_equipment"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
    frame["ttm_fcf_margin"] = ttm_fcf / frame["ttm_sales"].replace(0, np.nan)
    frame["ttm_fcf_margin_yoy_change"] = grouped["ttm_fcf_margin"].diff(4)
    frame["q_cfo_to_net_income"] = frame["net_cash_from_operating_activities"] / frame["net_income"].replace(0, np.nan)
    frame["q_fcf_to_net_income"] = (frame["net_cash_from_operating_activities"] + frame["property_and_equipment"]) / frame["net_income"].replace(0, np.nan)
    frame["q_accruals_to_assets"] = (frame["net_income"] - frame["net_cash_from_operating_activities"]) / frame["total_assets"].replace(0, np.nan)
    debt = frame["long_term_debt"].fillna(0)
    frame["debt_to_assets"] = debt / frame["total_assets"].replace(0, np.nan)
    frame["debt_to_equity"] = debt / frame["total_equity"].abs().replace(0, np.nan)
    frame["cash_to_assets"] = frame["cash_and_equivalents"] / frame["total_assets"].replace(0, np.nan)
    frame["current_ratio"] = frame["total_current_assets"] / frame["total_current_liabilities"].replace(0, np.nan)
    frame["asset_turnover_q"] = frame["sales"] / frame["total_assets"].replace(0, np.nan)
    frame["availability_date"] = frame["fiscal_period_end"] + pd.Timedelta(days=45)
    return frame[frame["sales"] > 25_000_000].reset_index(drop=True)


def _attach_annual_context(quarterly: pd.DataFrame) -> pd.DataFrame:
    annual_path = Path("ath_reversion_research/reports/dolthub_aligned_replication/aligned_events.csv")
    if not annual_path.exists():
        return quarterly
    annual = pd.read_csv(annual_path, parse_dates=["availability_date"])
    mapping = {
        "revenue_growth_yoy": "annual_revenue_growth_yoy",
        "revenue_growth_accel": "annual_revenue_growth_accel",
        "gross_margin": "annual_gross_margin",
        "operating_margin": "annual_operating_margin",
        "net_margin": "annual_net_margin",
        "ebitda_margin": "annual_ebitda_margin",
        "cfo_margin": "annual_cfo_margin",
        "fcf_margin": "annual_fcf_margin",
        "debt_to_assets": "annual_debt_to_assets",
        "cash_to_assets": "annual_cash_to_assets",
        "asset_turnover": "annual_asset_turnover",
    }
    annual_context = annual[["symbol", "availability_date", *mapping.keys()]].rename(columns=mapping)
    quarterly = quarterly.copy()
    quarterly["availability_date"] = pd.to_datetime(quarterly["availability_date"])
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


def _load_prices(symbols: list[str]) -> pd.DataFrame:
    cache = Path("ath_reversion_research/reports/dolthub_full_fundamental_validation/price_subset_yahoo.csv")
    if cache.exists():
        prices = pd.read_csv(cache, parse_dates=["date"])
        missing = sorted(set(symbols) - set(prices["symbol"].astype(str)))
        if missing:
            prices = pd.concat([prices, download_yahoo_ohlcv(missing, start="2015-01-01", chunk_size=25)], ignore_index=True)
        return prices[prices["symbol"].isin(symbols)].copy()
    return download_yahoo_ohlcv(symbols, start="2015-01-01", chunk_size=25)


def _attach_prices(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    spy = close["SPY"]
    rows = []
    for row in events.to_dict("records"):
        symbol = str(row["symbol"])
        if symbol not in close:
            continue
        series = close[symbol].dropna()
        trade_date = pd.Timestamp(row["availability_date"])
        pos = series.index.searchsorted(trade_date, side="left")
        spy_pos = spy.index.searchsorted(trade_date, side="left")
        if pos >= len(series) or pos < 252 or spy_pos < 126:
            continue
        out = dict(row)
        current = float(series.iloc[pos])
        out["trade_date"] = series.index[pos]
        out["trade_close"] = current
        out["trailing_3m_return"] = current / series.iloc[pos - 63] - 1
        out["trailing_6m_return"] = current / series.iloc[pos - 126] - 1
        out["trailing_12m_return"] = current / series.iloc[pos - 252] - 1
        out["relative_6m_vs_spy"] = out["trailing_6m_return"] - (spy.iloc[spy_pos] / spy.iloc[spy_pos - 126] - 1)
        out["realized_vol_3m"] = series.pct_change().iloc[pos - 62 : pos + 1].std() * np.sqrt(252)
        out["drawdown_12m"] = current / series.iloc[pos - 251 : pos + 1].max() - 1
        for name, bars in FORWARD_WINDOWS.items():
            end = pos + bars
            if end < len(series):
                out[name] = series.iloc[end] / current - 1
                out[f"{name}_end_date"] = series.index[end]
            else:
                out[name] = np.nan
                out[f"{name}_end_date"] = pd.NaT
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _walk_forward(events: pd.DataFrame, feature_set: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for target in FORWARD_WINDOWS:
        end_col = f"{target}_end_date"
        subset = events.dropna(subset=[target, end_col, "q_sales_qoq"]).copy()
        for period in sorted(subset["trade_date"].dt.to_period("Q").unique()):
            start = period.start_time
            end = period.end_time
            test = subset[(subset["trade_date"] >= start) & (subset["trade_date"] <= end)].copy()
            train = subset[(subset["trade_date"] < start) & (subset[end_col] < start)].copy()
            if len(train) < 100 or test.empty:
                continue
            test["prediction"] = _fit_predict(train, test, target, features)
            test["target_return"] = test[target]
            test["feature_set"] = feature_set
            test["forward_window"] = target
            test["test_period"] = str(period)
            rows.append(test)
    return pd.concat(rows, ignore_index=True)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str]) -> np.ndarray:
    usable = [f for f in features if f in train and train[f].notna().mean() >= 0.4 and train[f].std(skipna=True) > 0]
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
    for (feature_set, target), group in pred.groupby(["feature_set", "forward_window"]):
        top = group[group["prediction"] >= group["prediction"].quantile(0.8)]
        bottom = group[group["prediction"] <= group["prediction"].quantile(0.2)]
        baseline = ((group["target_return"] - group["target_return"].mean()) ** 2).sum()
        residual = ((group["target_return"] - group["prediction"]) ** 2).sum()
        rows.append({
            "feature_set": feature_set,
            "forward_window": target,
            "observations": len(group),
            "test_periods": group["test_period"].nunique(),
            "oos_r2": 1 - residual / baseline if baseline > 0 else np.nan,
            "pearson_corr": group["prediction"].corr(group["target_return"]),
            "spearman_corr": group["prediction"].rank().corr(group["target_return"].rank()),
            "top_quintile_mean_return": top["target_return"].mean(),
            "top_quintile_median_return": top["target_return"].median(),
            "top_quintile_hit_rate": (top["target_return"] > 0).mean(),
            "bottom_quintile_mean_return": bottom["target_return"].mean(),
            "top_minus_bottom_mean": top["target_return"].mean() - bottom["target_return"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["forward_window", "spearman_corr"], ascending=[True, False])


def _write_report(events: pd.DataFrame, pred: pd.DataFrame, perf: pd.DataFrame) -> None:
    lines = [
        "# DoltHub Quarterly Trend Model",
        "",
        "Tests TTM, QoQ, YoY, acceleration, margin-change, cash-flow-quality, and trailing price features from Dolt quarterly statements.",
        "",
        "## Dataset",
        "",
        f"- Quarterly events: {len(events)}.",
        f"- Prediction rows: {len(pred)}.",
        f"- Symbols with predictions: {pred['symbol'].nunique() if not pred.empty else 0}.",
        f"- Test periods: {pred['test_period'].nunique() if not pred.empty else 0}.",
        "",
        "## Performance",
        "",
        _markdown_table(perf),
        "",
        "## Interpretation",
        "",
        "- This is the requested time-series/rate-of-change fundamentals test.",
        "- Quarterly features are useful only if they beat the annual/price baselines on rank and top-quintile realized returns.",
    ]
    (OUTPUT_DIR / "DOLTHUB_QUARTERLY_TREND_MODEL.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "test_periods"}:
            if any(token in col for token in ["r2", "corr", "return", "rate"]):
                display[col] = display[col].map(lambda v: f"{float(v) * 100:.2f}%" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
