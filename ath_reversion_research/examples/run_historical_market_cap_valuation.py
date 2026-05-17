"""Historical market-cap valuation features at trade date.

The previous valuation ablations used current market cap. This script replaces
those valuation features with an estimated market cap at the trade date:

    historical_market_cap ~= annual shares outstanding * trade-date close

Shares outstanding comes from local Dolt annual balance_sheet_equity, with
income_statement average_shares as a fallback. Trade-date close comes from the
already aligned event file. This is still approximate because share counts and
adjusted prices may not share exactly the same split/restatement convention.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import run_dolthub_aligned_replication as aligned
import run_dolthub_full_fundamental_validation as dolt_full


OUTPUT_DIR = Path("ath_reversion_research/reports/historical_market_cap_valuation")
ABLATION_EVENTS = Path("ath_reversion_research/reports/dolthub_upgrade_ablation/ablation_events.csv")
SAVE_ROW_LEVEL = os.environ.get("SAVE_ROW_LEVEL_HIST_VALUATION", "").lower() in {"1", "true", "yes"}

BASE_FEATURES = aligned.FINANCIAL_FEATURES + aligned.PRICE_FEATURES
CURRENT_VALUATION_FEATURES = [
    "sales_to_market_cap",
    "earnings_yield",
    "fcf_yield",
    "gross_profit_to_market_cap",
    "growth_to_sales_multiple",
]
HIST_VALUATION_FEATURES = [
    "hist_sales_to_market_cap",
    "hist_earnings_yield",
    "hist_fcf_yield",
    "hist_gross_profit_to_market_cap",
    "hist_ebitda_to_market_cap",
    "hist_growth_to_sales_multiple",
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(
        ABLATION_EVENTS,
        parse_dates=["trade_date", "fwd_12m_return_end_date", "fiscal_period_end"],
    )
    shares = _load_shares(sorted(events["symbol"].dropna().astype(str).unique()))
    events = events.merge(shares, on=["symbol", "fiscal_period_end"], how="left")
    events["shares_for_market_cap"] = events["shares_outstanding"].where(
        events["shares_outstanding"].notna(), events["average_shares"]
    )
    events["historical_market_cap"] = events["shares_for_market_cap"] * events["trade_close"]
    events = _add_hist_valuation(events)
    events = events.replace([np.inf, -np.inf], np.nan)
    if SAVE_ROW_LEVEL:
        events.to_csv(OUTPUT_DIR / "events_with_historical_valuation.csv", index=False)

    experiments = [
        ("baseline_no_valuation", "raw_return", BASE_FEATURES),
        ("current_cap_valuation", "raw_return", BASE_FEATURES + CURRENT_VALUATION_FEATURES),
        ("hist_cap_valuation", "raw_return", BASE_FEATURES + HIST_VALUATION_FEATURES),
        (
            "hist_cap_valuation_sector_excess",
            "sector_excess_return",
            BASE_FEATURES + HIST_VALUATION_FEATURES,
        ),
        (
            "hist_and_current_valuation",
            "raw_return",
            BASE_FEATURES + CURRENT_VALUATION_FEATURES + HIST_VALUATION_FEATURES,
        ),
    ]
    predictions = [_walk_forward(events, name, target_type, features) for name, target_type, features in experiments]
    pred = pd.concat(predictions, ignore_index=True)
    if SAVE_ROW_LEVEL:
        pred.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    perf = _performance(pred)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    _write_report(perf, events)
    print(perf.to_string(index=False))
    return 0


def _load_shares(symbols: list[str]) -> pd.DataFrame:
    frames = []
    for chunk in dolt_full._chunks(symbols, 100):
        in_list = ", ".join(f"'{symbol}'" for symbol in chunk)
        query = f"""
        SELECT
            i.act_symbol AS symbol,
            i.date AS fiscal_period_end,
            e.shares_outstanding,
            i.average_shares
        FROM income_statement i
        LEFT JOIN balance_sheet_equity e
          ON i.act_symbol=e.act_symbol AND i.date=e.date AND i.period=e.period
        WHERE i.period='Year' AND i.act_symbol IN ({in_list})
        ORDER BY i.act_symbol, i.date;
        """
        frame = dolt_full._dolt_query(dolt_full.EARNINGS_DIR, query)
        if not frame.empty:
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["fiscal_period_end"] = pd.to_datetime(out["fiscal_period_end"])
    for col in ["shares_outstanding", "average_shares"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop_duplicates(["symbol", "fiscal_period_end"])


def _add_hist_valuation(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    hist_cap = out["historical_market_cap"].replace(0, np.nan)
    revenue = out["sales"].replace(0, np.nan)
    fcf = out["fcf_margin"] * revenue
    out["hist_sales_to_market_cap"] = revenue / hist_cap
    out["hist_earnings_yield"] = out["net_income"] / hist_cap
    out["hist_fcf_yield"] = fcf / hist_cap
    out["hist_gross_profit_to_market_cap"] = out["gross_profit"] / hist_cap
    out["hist_ebitda_to_market_cap"] = out["ebitda"] / hist_cap
    sales_multiple = hist_cap / revenue
    out["hist_growth_to_sales_multiple"] = out["revenue_growth_yoy"] / sales_multiple.replace(0, np.nan)
    return out


def _target_column(target_type: str) -> str:
    if target_type == "sector_excess_return":
        return "fwd_12m_sector_excess_return"
    return "fwd_12m_return"


def _walk_forward(events: pd.DataFrame, experiment: str, target_type: str, features: list[str]) -> pd.DataFrame:
    rows = []
    target_col = _target_column(target_type)
    for bucket in ["all", "large_cap", "mid_cap"]:
        subset = events.copy() if bucket == "all" else events[events["market_cap_bucket"] == bucket].copy()
        subset = subset.dropna(subset=[target_col, "fwd_12m_return", "fwd_12m_return_end_date"])
        for year in sorted(subset["trade_date"].dt.year.unique()):
            start = pd.Timestamp(f"{int(year)}-01-01")
            test = subset[subset["trade_date"].dt.year == year].copy()
            train = subset[(subset["trade_date"] < start) & (subset["fwd_12m_return_end_date"] < start)].copy()
            if len(train) < 100 or test.empty:
                continue
            test["prediction"] = _fit_predict(train, test, target_col, features)
            test["target_return"] = test["fwd_12m_return"]
            test["ranking_target"] = test[target_col]
            test["experiment"] = experiment
            test["target_type"] = target_type
            test["regression_bucket"] = bucket
            test["test_year"] = int(year)
            rows.append(test)
    return pd.concat(rows, ignore_index=True)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, target_col: str, features: list[str]) -> np.ndarray:
    usable = [
        f
        for f in features
        if f in train and train[f].notna().mean() >= 0.50 and train[f].std(skipna=True) > 0
    ]
    low = train[usable].quantile(0.01)
    high = train[usable].quantile(0.99)
    med = train[usable].median()
    x_train = train[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    x_test = test[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    mean = x_train.mean()
    std = x_train.std().replace(0, 1)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    y_train = train[target_col].astype(float)
    alpha = 10.0
    x = x_train.to_numpy(dtype=float)
    y = y_train.to_numpy(dtype=float)
    beta = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    return x_test.to_numpy(dtype=float) @ beta


def _spearman(a: pd.Series, b: pd.Series) -> float:
    tmp = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(tmp) < 3:
        return np.nan
    return float(tmp["a"].rank().corr(tmp["b"].rank()))


def _top_minus_bottom(group: pd.DataFrame) -> float:
    ranked = group.dropna(subset=["prediction", "target_return"]).copy()
    if len(ranked) < 10:
        return np.nan
    ranked["rank_pct"] = ranked["prediction"].rank(pct=True)
    top = ranked[ranked["rank_pct"] >= 0.80]["target_return"].mean()
    bottom = ranked[ranked["rank_pct"] <= 0.20]["target_return"].mean()
    return float(top - bottom)


def _performance(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in pred.groupby(["experiment", "target_type", "regression_bucket"]):
        experiment, target_type, bucket = keys
        top = group[group["prediction"].rank(pct=True) >= 0.80]["target_return"]
        yearly = group.groupby("test_year").apply(_top_minus_bottom)
        rows.append(
            {
                "experiment": experiment,
                "target_type": target_type,
                "regression_bucket": bucket,
                "rows": len(group),
                "years": group["test_year"].nunique(),
                "spearman": _spearman(group["prediction"], group["target_return"]),
                "target_spearman": _spearman(group["prediction"], group["ranking_target"]),
                "top_quintile_mean_return": top.mean(),
                "top_quintile_hit_rate": (top > 0).mean(),
                "mean_yearly_top_minus_bottom": yearly.mean(),
                "positive_yearly_spread_rate": (yearly > 0).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["regression_bucket", "spearman"], ascending=[True, False])


def _coverage(events: pd.DataFrame) -> pd.DataFrame:
    cols = ["historical_market_cap", "shares_outstanding", "average_shares"] + HIST_VALUATION_FEATURES
    return pd.DataFrame(
        [{"field": col, "coverage": events[col].notna().mean(), "median": events[col].median()} for col in cols]
    )


def _write_report(perf: pd.DataFrame, events: pd.DataFrame) -> None:
    coverage = _coverage(events)
    with (OUTPUT_DIR / "HISTORICAL_MARKET_CAP_VALUATION.md").open("w", encoding="utf-8") as f:
        f.write("# Historical market-cap valuation feature test\n\n")
        f.write(
            "This test replaces current market-cap valuation ratios with ratios computed from "
            "`shares_outstanding * trade_close` at the trade date. Average shares is used only "
            "when ending shares outstanding is missing.\n\n"
        )
        f.write("## No-leakage setup\n\n")
        f.write(
            "- Fundamental rows are annual Dolt rows already shifted to their reporting/trade date.\n"
            "- Market cap uses the trade-date close and annual share count from the same fiscal row.\n"
            "- Walk-forward training only uses observations whose 12-month forward window ended before the test year.\n"
            "- The calculation remains approximate if Dolt share counts and Yahoo adjusted closes differ in split convention.\n\n"
        )
        f.write(
            "Row-level event/prediction CSVs are skipped by default to avoid large generated files. "
            "Set `SAVE_ROW_LEVEL_HIST_VALUATION=1` when running the script to export them locally.\n\n"
        )
        f.write("## Feature coverage\n\n")
        f.write(_markdown_table(coverage))
        f.write("\n\n## Performance summary\n\n")
        f.write(_markdown_table(perf))
        f.write("\n")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + body)


if __name__ == "__main__":
    raise SystemExit(main())
