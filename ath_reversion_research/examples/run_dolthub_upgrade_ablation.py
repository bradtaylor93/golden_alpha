"""DoltHub long-history ablation suite for ranking model upgrades.

Runs the proposed improvements across the longer Dolt aligned annual sample:

- raw return vs SPY/sector excess-return targets,
- raw return regression vs rank / top-quintile targets,
- valuation-style features,
- combined feature sets.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ath_reversion_research import download_yahoo_ohlcv
import run_dolthub_aligned_replication as aligned


OUTPUT_DIR = Path("ath_reversion_research/reports/dolthub_upgrade_ablation")
EVENTS = Path("ath_reversion_research/reports/dolthub_aligned_replication/aligned_events.csv")
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SECTOR_ETF = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

VALUATION_FEATURES = [
    "sales_to_market_cap",
    "earnings_yield",
    "fcf_yield",
    "gross_profit_to_market_cap",
    "growth_to_sales_multiple",
]
BASE_FEATURES = aligned.FINANCIAL_FEATURES + aligned.PRICE_FEATURES
SECTOR_DUMMY_FEATURES = [f"sector_dummy_{idx}" for idx in range(len(SECTOR_ETF))]
SECTOR_INTERACTION_FEATURES = [
    f"{feature}_x_{dummy}"
    for dummy in SECTOR_DUMMY_FEATURES
    for feature in ["revenue_growth_yoy", "operating_margin", "fcf_margin", "debt_to_assets", "trailing_6m_return"]
]

EXPERIMENTS = [
    ("baseline_raw_return", "raw_return", BASE_FEATURES),
    ("valuation_raw_return", "raw_return", BASE_FEATURES + VALUATION_FEATURES),
    ("baseline_rank_target", "rank_target", BASE_FEATURES),
    ("valuation_rank_target", "rank_target", BASE_FEATURES + VALUATION_FEATURES),
    ("baseline_top_quintile_target", "top_quintile_target", BASE_FEATURES),
    ("valuation_top_quintile_target", "top_quintile_target", BASE_FEATURES + VALUATION_FEATURES),
    ("spy_excess_raw_return", "spy_excess_return", BASE_FEATURES + VALUATION_FEATURES),
    ("sector_excess_raw_return", "sector_excess_return", BASE_FEATURES + VALUATION_FEATURES),
    ("sector_excess_rank_target", "sector_excess_rank_target", BASE_FEATURES + VALUATION_FEATURES),
    ("sector_dummy_raw_return", "raw_return", BASE_FEATURES + VALUATION_FEATURES + SECTOR_DUMMY_FEATURES),
    ("sector_dummy_excess_return", "sector_excess_return", BASE_FEATURES + VALUATION_FEATURES + SECTOR_DUMMY_FEATURES),
    ("sector_dummy_interactions", "sector_excess_return", BASE_FEATURES + VALUATION_FEATURES + SECTOR_DUMMY_FEATURES + SECTOR_INTERACTION_FEATURES),
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS, parse_dates=["trade_date", "fwd_3m_return_end_date", "fwd_6m_return_end_date", "fwd_12m_return_end_date"])
    events = _add_sectors(events)
    events = _add_valuation_features(events)
    events = _add_sector_dummy_features(events)
    events = _add_excess_return_targets(events)
    events.to_csv(OUTPUT_DIR / "ablation_events.csv", index=False)
    predictions = []
    for experiment, target_type, features in EXPERIMENTS:
        predictions.append(_walk_forward(events, experiment, target_type, features))
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    perf = _performance(pred)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    _write_report(perf)
    print(perf.to_string(index=False))
    return 0


def _add_sectors(events: pd.DataFrame) -> pd.DataFrame:
    response = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text))[0]
    sectors = pd.DataFrame({
        "symbol": table["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper(),
        "gics_sector": table["GICS Sector"].astype(str),
    })
    return events.drop(columns=["gics_sector"], errors="ignore").merge(sectors, on="symbol", how="left")


def _add_valuation_features(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    cap = frame["current_market_cap"].replace(0, np.nan)
    revenue = frame["sales"].replace(0, np.nan) if "sales" in frame else frame["total_revenue"].replace(0, np.nan)
    net_income = frame["net_income"].replace(0, np.nan)
    gross_profit = frame["gross_profit"].replace(0, np.nan)
    fcf = frame["fcf_margin"] * revenue
    frame["sales_to_market_cap"] = revenue / cap
    frame["earnings_yield"] = net_income / cap
    frame["fcf_yield"] = fcf / cap
    frame["gross_profit_to_market_cap"] = gross_profit / cap
    frame["growth_to_sales_multiple"] = frame["revenue_growth_yoy"] / (cap / revenue).replace(0, np.nan)
    return frame


def _add_sector_dummy_features(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    sectors = sorted(SECTOR_ETF)
    for idx, sector in enumerate(sectors):
        dummy = f"sector_dummy_{idx}"
        frame[dummy] = (frame["gics_sector"] == sector).astype(float)
        for feature in ["revenue_growth_yoy", "operating_margin", "fcf_margin", "debt_to_assets", "trailing_6m_return"]:
            frame[f"{feature}_x_{dummy}"] = frame[feature] * frame[dummy]
    return frame


def _add_excess_return_targets(events: pd.DataFrame) -> pd.DataFrame:
    symbols = sorted(set(["SPY", *SECTOR_ETF.values()]))
    prices = download_yahoo_ohlcv(symbols, start="2012-01-01", chunk_size=25)
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    rows = []
    for row in events.to_dict("records"):
        start = close.index.searchsorted(pd.Timestamp(row["trade_date"]), side="left")
        end = close.index.searchsorted(pd.Timestamp(row["fwd_12m_return_end_date"]), side="left")
        out = dict(row)
        if start < len(close) and end < len(close):
            spy_return = close["SPY"].iloc[end] / close["SPY"].iloc[start] - 1.0
            sector_symbol = SECTOR_ETF.get(row.get("gics_sector"))
            sector_return = close[sector_symbol].iloc[end] / close[sector_symbol].iloc[start] - 1.0 if sector_symbol in close else spy_return
            out["fwd_12m_spy_excess_return"] = out["fwd_12m_return"] - spy_return
            out["fwd_12m_sector_excess_return"] = out["fwd_12m_return"] - sector_return
        else:
            out["fwd_12m_spy_excess_return"] = np.nan
            out["fwd_12m_sector_excess_return"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _target_column(target_type: str) -> str:
    if target_type == "spy_excess_return":
        return "fwd_12m_spy_excess_return"
    if target_type in {"sector_excess_return", "sector_excess_rank_target"}:
        return "fwd_12m_sector_excess_return"
    return "fwd_12m_return"


def _build_target(frame: pd.DataFrame, target_type: str) -> pd.Series:
    raw = frame[_target_column(target_type)].astype(float)
    if target_type in {"rank_target", "sector_excess_rank_target"}:
        return raw.groupby(frame["trade_date"]).rank(pct=True)
    if target_type == "top_quintile_target":
        return (raw >= raw.groupby(frame["trade_date"]).transform(lambda s: s.quantile(0.80))).astype(float)
    return raw


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
            y_train = _build_target(train, target_type)
            test["prediction"] = _fit_predict(train, test, features, y_train)
            test["target_return"] = test["fwd_12m_return"]
            test["ranking_target"] = _build_target(test, target_type)
            test["experiment"] = experiment
            test["target_type"] = target_type
            test["regression_bucket"] = bucket
            test["test_year"] = int(year)
            rows.append(test)
    return pd.concat(rows, ignore_index=True)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], y_train: pd.Series) -> np.ndarray:
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
    y = y_train.to_numpy(float)
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
    for (experiment, bucket), group in pred.groupby(["experiment", "regression_bucket"]):
        top = group[group["prediction"] >= group["prediction"].quantile(0.80)]
        bottom = group[group["prediction"] <= group["prediction"].quantile(0.20)]
        base = ((group["ranking_target"] - group["ranking_target"].mean()) ** 2).sum()
        resid = ((group["ranking_target"] - group["prediction"]) ** 2).sum()
        rows.append({
            "experiment": experiment,
            "target_type": group["target_type"].iloc[0],
            "regression_bucket": bucket,
            "observations": len(group),
            "test_years": group["test_year"].nunique(),
            "oos_r2_on_training_target": 1 - resid / base if base > 0 else np.nan,
            "raw_return_spearman": group["prediction"].rank().corr(group["target_return"].rank()),
            "target_spearman": group["prediction"].rank().corr(group["ranking_target"].rank()),
            "top_quintile_mean_return": top["target_return"].mean(),
            "top_quintile_median_return": top["target_return"].median(),
            "top_quintile_hit_rate": (top["target_return"] > 0).mean(),
            "bottom_quintile_mean_return": bottom["target_return"].mean(),
            "top_minus_bottom_mean": top["target_return"].mean() - bottom["target_return"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["regression_bucket", "raw_return_spearman"], ascending=[True, False])


def _write_report(perf: pd.DataFrame) -> None:
    lines = [
        "# DoltHub Upgrade Ablation",
        "",
        "Runs proposed target/valuation/rank upgrades across the longer Dolt sample.",
        "",
        "## Performance",
        "",
        _markdown_table(perf),
        "",
        "## Interpretation",
        "",
        "- This is the long-history counterpart to the Yahoo/current-S&P ablation.",
        "- Upgrades are only useful if they improve realized raw-return ranking, not merely the transformed training target.",
    ]
    (OUTPUT_DIR / "DOLTHUB_UPGRADE_ABLATION.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "test_years"}:
            if any(token in col for token in ["r2", "spearman", "return", "rate"]):
                display[col] = display[col].map(lambda v: f"{float(v) * 100:.2f}%" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
