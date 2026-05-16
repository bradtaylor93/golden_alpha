"""S&P annual ranking improvement with sector and macro context.

Adds three causal feature groups to the existing annual financial+price model:

1. sector-relative financial ratios,
2. sector aggregate financial backdrop using only events available up to the
   trade date,
3. market/macro proxy features from liquid ETFs available at the trade date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research import download_yahoo_ohlcv
import run_sp500_annual_improvement as base


BASE_DIR = Path("ath_reversion_research/reports/sp500_annual_improvement")
OUTPUT_DIR = Path("ath_reversion_research/reports/sp500_sector_macro_improvement")
MACRO_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "IEF", "HYG", "LQD", "GLD", "UUP", "USO"]

SECTOR_RELATIVE_FEATURES = [
    "sector_rel_revenue_growth_yoy",
    "sector_rel_operating_margin",
    "sector_rel_fcf_margin",
    "sector_rel_rd_to_revenue",
    "sector_rel_debt_to_assets",
    "sector_rel_trailing_6m",
]
SECTOR_AGG_FEATURES = [
    "sector_median_revenue_growth_yoy",
    "sector_median_operating_margin",
    "sector_median_fcf_margin",
    "sector_median_trailing_6m",
]
MACRO_FEATURES = [
    "macro_spy_6m",
    "macro_spy_12m",
    "macro_spy_vol_3m",
    "macro_qqq_vs_spy_6m",
    "macro_iwm_vs_spy_6m",
    "macro_tlt_6m",
    "macro_hyg_lqd_6m_spread",
    "macro_uup_6m",
    "macro_gld_6m",
    "macro_uso_6m",
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(BASE_DIR / "enhanced_events.csv", parse_dates=["trade_date", "fwd_3m_return_end_date", "fwd_6m_return_end_date", "fwd_12m_return_end_date"])
    events = _add_sector_features(events)
    events = _add_macro_features(events)
    events.to_csv(OUTPUT_DIR / "enhanced_sector_macro_events.csv", index=False)

    feature_sets = dict(base.FEATURE_SETS)
    feature_sets["financial_price_sector"] = base.FEATURE_SETS["financial_plus_price"] + SECTOR_RELATIVE_FEATURES + SECTOR_AGG_FEATURES
    feature_sets["financial_price_macro"] = base.FEATURE_SETS["financial_plus_price"] + MACRO_FEATURES
    feature_sets["financial_price_sector_macro"] = base.FEATURE_SETS["financial_plus_price"] + SECTOR_RELATIVE_FEATURES + SECTOR_AGG_FEATURES + MACRO_FEATURES

    predictions = []
    for feature_set, features in feature_sets.items():
        predictions.append(_walk_forward(events, feature_set, features))
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    perf = base._performance(pred)
    sector_perf = base._sector_neutral_performance(pred)
    conditions = base._condition_summary(pred)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    sector_perf.to_csv(OUTPUT_DIR / "sector_neutral_summary.csv", index=False)
    conditions.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False)
    _write_report(perf, sector_perf, conditions)
    print(perf.to_string(index=False))
    return 0


def _add_sector_features(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.sort_values(["trade_date", "symbol"]).copy()
    rel_cols = {
        "revenue_growth_yoy": "sector_rel_revenue_growth_yoy",
        "operating_margin": "sector_rel_operating_margin",
        "fcf_margin": "sector_rel_fcf_margin",
        "rd_to_revenue": "sector_rel_rd_to_revenue",
        "debt_to_assets": "sector_rel_debt_to_assets",
        "trailing_6m_return": "sector_rel_trailing_6m",
    }
    agg_cols = {
        "revenue_growth_yoy": "sector_median_revenue_growth_yoy",
        "operating_margin": "sector_median_operating_margin",
        "fcf_margin": "sector_median_fcf_margin",
        "trailing_6m_return": "sector_median_trailing_6m",
    }
    out = []
    for date, day in frame.groupby("trade_date"):
        day = day.copy()
        for source, target in rel_cols.items():
            med = day.groupby("gics_sector")[source].transform("median")
            day[target] = day[source] - med
        for source, target in agg_cols.items():
            day[target] = day.groupby("gics_sector")[source].transform("median")
        out.append(day)
    return pd.concat(out, ignore_index=True).sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _add_macro_features(events: pd.DataFrame) -> pd.DataFrame:
    prices = download_yahoo_ohlcv(MACRO_SYMBOLS, start="2017-01-01", chunk_size=25)
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    macro = pd.DataFrame(index=close.index)
    macro["macro_spy_6m"] = close["SPY"] / close["SPY"].shift(126) - 1.0
    macro["macro_spy_12m"] = close["SPY"] / close["SPY"].shift(252) - 1.0
    macro["macro_spy_vol_3m"] = close["SPY"].pct_change().rolling(63, min_periods=30).std() * np.sqrt(252)
    macro["macro_qqq_vs_spy_6m"] = close["QQQ"] / close["QQQ"].shift(126) - close["SPY"] / close["SPY"].shift(126)
    macro["macro_iwm_vs_spy_6m"] = close["IWM"] / close["IWM"].shift(126) - close["SPY"] / close["SPY"].shift(126)
    macro["macro_tlt_6m"] = close["TLT"] / close["TLT"].shift(126) - 1.0
    macro["macro_hyg_lqd_6m_spread"] = (close["HYG"] / close["HYG"].shift(126) - 1.0) - (close["LQD"] / close["LQD"].shift(126) - 1.0)
    macro["macro_uup_6m"] = close["UUP"] / close["UUP"].shift(126) - 1.0
    macro["macro_gld_6m"] = close["GLD"] / close["GLD"].shift(126) - 1.0
    macro["macro_uso_6m"] = close["USO"] / close["USO"].shift(126) - 1.0
    macro = macro.shift(1)

    pieces = []
    for event in events.to_dict("records"):
        trade_date = pd.Timestamp(event["trade_date"])
        pos = macro.index.searchsorted(trade_date, side="right") - 1
        row = dict(event)
        if pos >= 0:
            row.update(macro.iloc[pos].to_dict())
        pieces.append(row)
    return pd.DataFrame(pieces)


def _walk_forward(events: pd.DataFrame, feature_set: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for target in base.annual.FORWARD_WINDOWS:
        end_col = f"{target}_end_date"
        for bucket in ["all", "large_cap", "mid_cap"]:
            subset = events.copy() if bucket == "all" else events[events["market_cap_bucket"] == bucket].copy()
            subset = subset.dropna(subset=[target, end_col, "revenue_growth_yoy"])
            for year in sorted(pd.to_datetime(subset["trade_date"]).dt.year.unique()):
                test_start = pd.Timestamp(f"{int(year)}-01-01")
                test = subset[pd.to_datetime(subset["trade_date"]).dt.year == year].copy()
                train = subset[(pd.to_datetime(subset["trade_date"]) < test_start) & (pd.to_datetime(subset[end_col]) < test_start)].copy()
                if len(train) < 25 or test.empty:
                    continue
                test["prediction"] = base._fit_predict(train, test, target, features)
                test["target_return"] = test[target]
                test["feature_set"] = feature_set
                test["forward_window"] = target
                test["regression_bucket"] = bucket
                test["test_year"] = int(year)
                rows.append(test)
    return pd.concat(rows, ignore_index=True)


def _write_report(perf: pd.DataFrame, sector_perf: pd.DataFrame, conditions: pd.DataFrame) -> None:
    lines = [
        "# S&P 500 Sector/Macro Improvement Study",
        "",
        "Adds sector-relative fundamentals, sector aggregate backdrop, and market/macro proxy features to the annual S&P ranking model.",
        "",
        "## Leakage controls",
        "",
        "- Uses purged annual prediction setup from the S&P annual improvement study.",
        "- Sector features are computed cross-sectionally at the same trade date.",
        "- Macro proxy features are trailing ETF returns/volatility shifted one day before the trade date.",
        "- Current sectors and current S&P membership are not point-in-time.",
        "",
        "## Predictive power",
        "",
        base._markdown_table(perf),
        "",
        "## Sector-neutral ranking",
        "",
        base._markdown_table(sector_perf),
        "",
        "## Conditions",
        "",
        base._markdown_table(conditions.head(60)),
        "",
        "## Interpretation",
        "",
        "- Sector/macro features are useful only if they improve the already strong financial+price large-cap 12m model.",
        "- If improvement is marginal, prefer the simpler financial+price model.",
    ]
    (OUTPUT_DIR / "SP500_SECTOR_MACRO_IMPROVEMENT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
