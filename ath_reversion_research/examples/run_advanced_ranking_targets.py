"""Advanced target/ranking experiments on the Dolt aligned sample.

Tests two proposed improvements:

1. risk-adjusted sector-excess target,
2. date-grouped LightGBM LambdaRank objective.

Evaluation is always on realized raw 12-month returns because the portfolio
ultimately needs raw P&L, not just a transformed target.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from ath_reversion_research import download_yahoo_ohlcv
import run_dolthub_aligned_replication as aligned


OUTPUT_DIR = Path("ath_reversion_research/reports/advanced_ranking_targets")
EVENTS = Path("ath_reversion_research/reports/dolthub_aligned_replication/aligned_events.csv")
PRICE_CACHE = Path("ath_reversion_research/reports/dolthub_full_fundamental_validation/price_subset_yahoo.csv")
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
FEATURES = aligned.FINANCIAL_FEATURES + aligned.PRICE_FEATURES + VALUATION_FEATURES


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS, parse_dates=["trade_date", "fwd_12m_return_end_date"])
    events = _add_sectors(events)
    events = _add_valuation_features(events)
    events = _add_forward_risk_targets(events)
    events.to_csv(OUTPUT_DIR / "advanced_events.csv", index=False)

    predictions = []
    for bucket in ["all", "large_cap", "mid_cap"]:
        predictions.append(_ridge_walk_forward(events, bucket, "raw_return", "fwd_12m_return", FEATURES))
        predictions.append(_ridge_walk_forward(events, bucket, "sector_excess", "fwd_12m_sector_excess_return", FEATURES))
        predictions.append(_ridge_walk_forward(events, bucket, "risk_adj_sector_excess", "fwd_12m_risk_adj_sector_excess", FEATURES))
        predictions.append(_lgbm_ranker_walk_forward(events, bucket, "lgbm_rank_raw", "fwd_12m_return", FEATURES))
        predictions.append(_lgbm_ranker_walk_forward(events, bucket, "lgbm_rank_sector_excess", "fwd_12m_sector_excess_return", FEATURES))
    pred = pd.concat([p for p in predictions if not p.empty], ignore_index=True)
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
    sectors = pd.DataFrame(
        {
            "symbol": table["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper(),
            "gics_sector": table["GICS Sector"].astype(str),
        }
    )
    return events.drop(columns=["gics_sector"], errors="ignore").merge(sectors, on="symbol", how="left")


def _add_valuation_features(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    cap = frame["current_market_cap"].replace(0, np.nan)
    revenue = frame["sales"].replace(0, np.nan) if "sales" in frame else frame["total_revenue"].replace(0, np.nan)
    frame["sales_to_market_cap"] = revenue / cap
    frame["earnings_yield"] = frame["net_income"] / cap
    frame["fcf_yield"] = (frame["fcf_margin"] * revenue) / cap
    frame["gross_profit_to_market_cap"] = frame["gross_profit"] / cap
    frame["growth_to_sales_multiple"] = frame["revenue_growth_yoy"] / (cap / revenue).replace(0, np.nan)
    return frame


def _load_close(symbols: list[str]) -> pd.DataFrame:
    if PRICE_CACHE.exists():
        prices = pd.read_csv(PRICE_CACHE, parse_dates=["date"])
        missing = sorted(set(symbols) - set(prices["symbol"].astype(str)))
        if missing:
            extra = download_yahoo_ohlcv(missing, start="2012-01-01", chunk_size=25)
            prices = pd.concat([prices, extra], ignore_index=True)
    else:
        prices = download_yahoo_ohlcv(symbols, start="2012-01-01", chunk_size=25)
    return prices[prices["symbol"].isin(symbols)].pivot(index="date", columns="symbol", values="close").sort_index().ffill()


def _add_forward_risk_targets(events: pd.DataFrame) -> pd.DataFrame:
    symbols = sorted(set(events["symbol"].astype(str)) | set(SECTOR_ETF.values()) | {"SPY"})
    close = _load_close(symbols)
    rows = []
    for row in events.to_dict("records"):
        start = close.index.searchsorted(pd.Timestamp(row["trade_date"]), side="left")
        end = close.index.searchsorted(pd.Timestamp(row["fwd_12m_return_end_date"]), side="left")
        out = dict(row)
        sector_symbol = SECTOR_ETF.get(row.get("gics_sector"))
        if start < len(close) and end < len(close) and sector_symbol in close and row["symbol"] in close:
            sector_return = close[sector_symbol].iloc[end] / close[sector_symbol].iloc[start] - 1.0
            out["fwd_12m_sector_excess_return"] = out["fwd_12m_return"] - sector_return
            daily = close[row["symbol"]].pct_change().iloc[start + 1 : end + 1]
            forward_vol = daily.std() * np.sqrt(252)
            if np.isfinite(forward_vol) and forward_vol > 0:
                out["fwd_12m_risk_adj_sector_excess"] = out["fwd_12m_sector_excess_return"] / forward_vol
            else:
                out["fwd_12m_risk_adj_sector_excess"] = np.nan
        else:
            out["fwd_12m_sector_excess_return"] = np.nan
            out["fwd_12m_risk_adj_sector_excess"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _subset(events: pd.DataFrame, bucket: str, target_col: str) -> pd.DataFrame:
    frame = events.copy() if bucket == "all" else events[events["market_cap_bucket"] == bucket].copy()
    return frame.dropna(subset=[target_col, "fwd_12m_return", "fwd_12m_return_end_date", "revenue_growth_yoy"])


def _ridge_walk_forward(events: pd.DataFrame, bucket: str, experiment: str, target_col: str, features: list[str]) -> pd.DataFrame:
    rows = []
    subset = _subset(events, bucket, target_col)
    for year in sorted(subset["trade_date"].dt.year.unique()):
        start = pd.Timestamp(f"{int(year)}-01-01")
        test = subset[subset["trade_date"].dt.year == year].copy()
        train = subset[(subset["trade_date"] < start) & (subset["fwd_12m_return_end_date"] < start)].copy()
        if len(train) < 100 or test.empty:
            continue
        test["prediction"] = _fit_ridge(train, test, target_col, features)
        test["target_return"] = test["fwd_12m_return"]
        test["ranking_target"] = test[target_col]
        test["experiment"] = experiment
        test["bucket"] = bucket
        test["test_year"] = int(year)
        rows.append(test)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fit_ridge(train: pd.DataFrame, test: pd.DataFrame, target_col: str, features: list[str]) -> np.ndarray:
    x_train, x_test = _prepare_features(train, test, features)
    y = train[target_col].to_numpy(float)
    y_mean = y.mean()
    x_mat = np.c_[np.ones(len(x_train)), x_train]
    t_mat = np.c_[np.ones(len(x_test)), x_test]
    penalty = np.eye(x_mat.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.solve(x_mat.T @ x_mat + 10.0 * penalty, x_mat.T @ (y - y_mean))
    beta[0] += y_mean
    return t_mat @ beta


def _lgbm_ranker_walk_forward(events: pd.DataFrame, bucket: str, experiment: str, target_col: str, features: list[str]) -> pd.DataFrame:
    try:
        import lightgbm as lgb
    except Exception:
        return pd.DataFrame()
    rows = []
    subset = _subset(events, bucket, target_col)
    for year in sorted(subset["trade_date"].dt.year.unique()):
        start = pd.Timestamp(f"{int(year)}-01-01")
        test = subset[subset["trade_date"].dt.year == year].copy()
        train = subset[(subset["trade_date"] < start) & (subset["fwd_12m_return_end_date"] < start)].copy()
        if len(train) < 100 or test.empty:
            continue
        x_train, x_test = _prepare_features(train, test, features)
        labels = (
            train[target_col]
            .groupby(train["trade_date"])
            .rank(pct=True, method="first")
            .mul(30)
            .round()
            .clip(lower=0, upper=30)
            .astype(int)
            .to_numpy()
        )
        group = train.groupby("trade_date").size().to_list()
        model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=60,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=3,
            min_child_samples=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            random_state=42,
            verbosity=-1,
        )
        model.fit(x_train, labels, group=group)
        test["prediction"] = model.predict(x_test)
        test["target_return"] = test["fwd_12m_return"]
        test["ranking_target"] = test[target_col]
        test["experiment"] = experiment
        test["bucket"] = bucket
        test["test_year"] = int(year)
        rows.append(test)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _prepare_features(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    usable = [f for f in features if f in train and train[f].notna().mean() >= 0.50 and train[f].std(skipna=True) > 0]
    low = train[usable].quantile(0.01)
    high = train[usable].quantile(0.99)
    med = train[usable].median()
    x_train = train[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    x_test = test[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    mean = x_train.mean()
    std = x_train.std().replace(0, 1)
    return ((x_train - mean) / std).fillna(0).to_numpy(float), ((x_test - mean) / std).fillna(0).to_numpy(float)


def _performance(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, bucket), group in pred.groupby(["experiment", "bucket"]):
        top = group[group["prediction"] >= group["prediction"].quantile(0.80)]
        bottom = group[group["prediction"] <= group["prediction"].quantile(0.20)]
        rows.append({
            "experiment": experiment,
            "bucket": bucket,
            "observations": len(group),
            "test_years": group["test_year"].nunique(),
            "raw_return_spearman": group["prediction"].rank().corr(group["target_return"].rank()),
            "target_spearman": group["prediction"].rank().corr(group["ranking_target"].rank()),
            "top_quintile_mean_return": top["target_return"].mean(),
            "top_quintile_median_return": top["target_return"].median(),
            "top_quintile_hit_rate": (top["target_return"] > 0).mean(),
            "bottom_quintile_mean_return": bottom["target_return"].mean(),
            "top_minus_bottom_mean": top["target_return"].mean() - bottom["target_return"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["bucket", "raw_return_spearman"], ascending=[True, False])


def _write_report(perf: pd.DataFrame) -> None:
    lines = [
        "# Advanced Ranking Targets",
        "",
        "Tests risk-adjusted sector-excess targets and grouped LightGBM ranking objectives on the Dolt aligned sample.",
        "",
        "## Results",
        "",
        _markdown_table(perf),
        "",
        "## Interpretation",
        "",
        "- A target is useful only if it improves realized raw 12m return ranking.",
        "- LightGBM ranker is useful only if it beats ridge on raw-return Spearman and top-quintile realized returns.",
    ]
    (OUTPUT_DIR / "ADVANCED_RANKING_TARGETS.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "test_years"}:
            if any(token in col for token in ["spearman", "return", "rate"]):
                display[col] = display[col].map(lambda v: f"{float(v) * 100:.2f}%" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
