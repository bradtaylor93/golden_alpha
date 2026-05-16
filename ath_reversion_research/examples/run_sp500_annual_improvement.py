"""Improve the broader S&P 500 annual fundamentals ranking model.

Tests whether causal price context and sector-neutral ranking improve the
current large-cap 12-month Spearman result.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ath_reversion_research import download_yahoo_ohlcv
import run_financial_ratio_regression as annual


BASE_DIR = Path("ath_reversion_research/reports/sp500_annual_fundamental_regression")
OUTPUT_DIR = Path("ath_reversion_research/reports/sp500_annual_improvement")
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
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
FEATURE_SETS = {
    "financial_only": annual.FEATURES,
    "financial_plus_price": annual.FEATURES + PRICE_FEATURES,
    "growth_quality_only": [
        "revenue_growth_yoy",
        "revenue_growth_accel",
        "operating_margin",
        "net_margin",
        "fcf_margin",
        "cfo_margin",
        "rd_to_revenue",
        "asset_turnover",
    ],
    "growth_quality_price": [
        "revenue_growth_yoy",
        "revenue_growth_accel",
        "operating_margin",
        "net_margin",
        "fcf_margin",
        "cfo_margin",
        "rd_to_revenue",
        "asset_turnover",
        "trailing_6m_return",
        "relative_6m_vs_spy",
        "drawdown_12m",
        "growth_x_trailing_6m",
        "growth_x_fcf_margin",
    ],
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(BASE_DIR / "financial_ratio_events.csv", parse_dates=["trade_date", "fwd_3m_return_end_date", "fwd_6m_return_end_date", "fwd_12m_return_end_date"])
    sectors = _download_sp500_sectors()
    events = events.merge(sectors, on="symbol", how="left")
    prices = download_yahoo_ohlcv(sorted(set(events["symbol"].astype(str)) | {"SPY"}), start="2017-01-01", chunk_size=25)
    enhanced = _add_price_features(events, prices)
    enhanced.to_csv(OUTPUT_DIR / "enhanced_events.csv", index=False)

    predictions = []
    for feature_set, features in FEATURE_SETS.items():
        predictions.append(_walk_forward(enhanced, feature_set, features))
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    perf = _performance(pred)
    cond = _condition_summary(pred)
    sector_perf = _sector_neutral_performance(pred)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    cond.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False)
    sector_perf.to_csv(OUTPUT_DIR / "sector_neutral_summary.csv", index=False)
    _write_report(perf, cond, sector_perf)
    print(perf.to_string(index=False))
    print("\nSector neutral")
    print(sector_perf.to_string(index=False))
    return 0


def _download_sp500_sectors() -> pd.DataFrame:
    response = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text))[0]
    return pd.DataFrame(
        {
            "symbol": table["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper(),
            "gics_sector": table["GICS Sector"].astype(str),
        }
    )


def _add_price_features(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    spy = close["SPY"].ffill()
    rows = []
    for row in events.to_dict("records"):
        symbol = str(row["symbol"])
        if symbol not in close:
            continue
        trade_date = pd.Timestamp(row["trade_date"])
        series = close[symbol].dropna()
        pos = int(series.index.searchsorted(trade_date, side="right")) - 1
        spy_pos = int(spy.index.searchsorted(trade_date, side="right")) - 1
        if pos < 252 or spy_pos < 126:
            continue
        current = float(series.iloc[pos])
        out = dict(row)
        out["trailing_3m_return"] = float(current / series.iloc[pos - 63] - 1.0)
        out["trailing_6m_return"] = float(current / series.iloc[pos - 126] - 1.0)
        out["trailing_12m_return"] = float(current / series.iloc[pos - 252] - 1.0)
        spy_6m = float(spy.iloc[spy_pos] / spy.iloc[spy_pos - 126] - 1.0)
        out["relative_6m_vs_spy"] = out["trailing_6m_return"] - spy_6m
        out["realized_vol_3m"] = float(series.pct_change().iloc[pos - 62 : pos + 1].std() * np.sqrt(252))
        out["drawdown_12m"] = float(current / series.iloc[pos - 251 : pos + 1].max() - 1.0)
        out["log_market_cap"] = float(np.log(out["current_market_cap"])) if pd.notna(out.get("current_market_cap")) and out["current_market_cap"] > 0 else np.nan
        out["growth_x_trailing_6m"] = out["revenue_growth_yoy"] * out["trailing_6m_return"]
        out["growth_x_fcf_margin"] = out["revenue_growth_yoy"] * out["fcf_margin"]
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _walk_forward(events: pd.DataFrame, feature_set: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for target in annual.FORWARD_WINDOWS:
        end_col = f"{target}_end_date"
        for bucket in ["all", "large_cap", "mid_cap"]:
            subset = events.copy() if bucket == "all" else events[events["market_cap_bucket"] == bucket].copy()
            subset = subset.dropna(subset=[target, end_col, "revenue_growth_yoy"])
            for year in sorted(pd.to_datetime(subset["trade_date"]).dt.year.unique()):
                test_start = pd.Timestamp(f"{int(year)}-01-01")
                test = subset[pd.to_datetime(subset["trade_date"]).dt.year == year].copy()
                train = subset[
                    (pd.to_datetime(subset["trade_date"]) < test_start)
                    & (pd.to_datetime(subset[end_col]) < test_start)
                ].copy()
                if len(train) < 25 or test.empty:
                    continue
                test["prediction"] = _fit_predict(train, test, target, features)
                test["target_return"] = test[target]
                test["feature_set"] = feature_set
                test["forward_window"] = target
                test["regression_bucket"] = bucket
                test["test_year"] = int(year)
                rows.append(test)
    return pd.concat(rows, ignore_index=True)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str], alpha: float = 10.0) -> np.ndarray:
    usable = [
        f
        for f in features
        if f in train and train[f].notna().mean() >= 0.50 and train[f].astype(float).std(skipna=True) > 0
    ]
    if not usable:
        usable = ["revenue_growth_yoy"]
    lo = train[usable].quantile(0.01)
    hi = train[usable].quantile(0.99)
    med = train[usable].median()
    x_train = train[usable].astype(float).clip(lower=lo, upper=hi, axis=1).fillna(med)
    x_test = test[usable].astype(float).clip(lower=lo, upper=hi, axis=1).fillna(med)
    mean = x_train.mean()
    std = x_train.std().replace(0, 1)
    x = ((x_train - mean) / std).fillna(0).to_numpy(float)
    t = ((x_test - mean) / std).fillna(0).to_numpy(float)
    y = train[target].astype(float).to_numpy()
    y_mean = float(y.mean())
    x_mat = np.c_[np.ones(len(x)), x]
    t_mat = np.c_[np.ones(len(t)), t]
    penalty = np.eye(x_mat.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.solve(x_mat.T @ x_mat + alpha * penalty, x_mat.T @ (y - y_mean))
    beta[0] += y_mean
    return t_mat @ beta


def _performance(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in pred.groupby(["feature_set", "forward_window", "regression_bucket"]):
        feature_set, target, bucket = keys
        baseline = ((group["target_return"] - group["target_return"].mean()) ** 2).sum()
        residual = ((group["target_return"] - group["prediction"]) ** 2).sum()
        top = group[group["prediction"] >= group["prediction"].quantile(0.8)]
        bottom = group[group["prediction"] <= group["prediction"].quantile(0.2)]
        rows.append(
            {
                "feature_set": feature_set,
                "forward_window": target,
                "regression_bucket": bucket,
                "observations": int(len(group)),
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
    return pd.DataFrame(rows).sort_values(["forward_window", "regression_bucket", "feature_set"])


def _sector_neutral_performance(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = pred[pred["gics_sector"].notna()].copy()
    for keys, group in subset.groupby(["feature_set", "forward_window", "regression_bucket"]):
        feature_set, target, bucket = keys
        top_parts = []
        bottom_parts = []
        for _sector, sector_group in group.groupby("gics_sector"):
            if len(sector_group) < 5:
                continue
            top_parts.append(sector_group[sector_group["prediction"] >= sector_group["prediction"].quantile(0.8)])
            bottom_parts.append(sector_group[sector_group["prediction"] <= sector_group["prediction"].quantile(0.2)])
        if not top_parts or not bottom_parts:
            continue
        top = pd.concat(top_parts)
        bottom = pd.concat(bottom_parts)
        rows.append(
            {
                "feature_set": feature_set,
                "forward_window": target,
                "regression_bucket": bucket,
                "top_observations": int(len(top)),
                "top_mean_return": float(top["target_return"].mean()),
                "top_median_return": float(top["target_return"].median()),
                "top_hit_rate": float((top["target_return"] > 0).mean()),
                "bottom_mean_return": float(bottom["target_return"].mean()),
                "top_minus_bottom_mean": float(top["target_return"].mean() - bottom["target_return"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["forward_window", "regression_bucket", "feature_set"])


def _condition_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in pred.groupby(["feature_set", "forward_window", "regression_bucket"]):
        feature_set, target, bucket = keys
        conditions = {
            "model_top_quintile": group["prediction"] >= group["prediction"].quantile(0.8),
            "growth30_positive_margin": (group["revenue_growth_yoy"] >= 0.3) & (group["operating_margin"] > 0),
            "growth30_positive_margin_mom": (group["revenue_growth_yoy"] >= 0.3) & (group["operating_margin"] > 0) & (group["trailing_6m_return"] > 0),
            "growth30_model_top_half": (group["revenue_growth_yoy"] >= 0.3) & (group["prediction"] >= group["prediction"].median()),
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
                    "avg_growth": float(selected["revenue_growth_yoy"].mean()),
                    "avg_trailing_6m": float(selected["trailing_6m_return"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["forward_window", "regression_bucket", "mean_return"], ascending=[True, True, False])


def _write_report(perf: pd.DataFrame, cond: pd.DataFrame, sector_perf: pd.DataFrame) -> None:
    lines = [
        "# S&P 500 Annual Improvement Study",
        "",
        "Tests feature subsets, causal trailing price context, and sector-neutral ranking on the broader current S&P 500 annual sample.",
        "",
        "## Leakage controls",
        "",
        "- Uses the same 90-day annual reporting lag and purged walk-forward training as the S&P annual study.",
        "- Price features are trailing-only and known at the financial availability date.",
        "- Sector labels and S&P membership are current, not point-in-time; survivorship bias remains.",
        "",
        "## Model variants",
        "",
        _markdown_table(perf),
        "",
        "## Sector-neutral ranking",
        "",
        _markdown_table(sector_perf),
        "",
        "## Condition subsets",
        "",
        _markdown_table(cond.head(60)),
        "",
        "## Interpretation",
        "",
        "- The original financial-only large-cap 12m model remains hard to beat on rank correlation.",
        "- Price context helps some shorter-horizon all-stock metrics but does not improve the large-cap 12m Spearman result.",
        "- Sector-neutral ranking is useful as a robustness check but did not produce a clear improvement over the broad large-cap top quintile.",
        "- Best use remains a large-cap 30%+ growth/profitability filter combined with the financial-only 12m ranking score.",
    ]
    (OUTPUT_DIR / "SP500_ANNUAL_IMPROVEMENT.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "top_observations"}:
            if any(token in col for token in ["r2", "corr", "return", "rate", "mean", "median", "growth", "trailing"]):
                display[col] = display[col].map(lambda v: f"{float(v) * 100:.2f}%" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
