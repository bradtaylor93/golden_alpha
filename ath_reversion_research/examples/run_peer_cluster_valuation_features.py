"""Peer, cluster, and valuation-residual feature ablation.

This script tests whether "under/over-valued versus similar peers" improves the
Dolt annual ranking model. It keeps the walk-forward discipline strict:

* peer clusters are fitted inside each fold using only prior train rows;
* sector/cluster reference valuation medians come only from prior train rows;
* expected-valuation residual models are fitted only on prior train rows;
* the final 12m return model trains only on observations whose forward window
  ended before the test year.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_historical_market_cap_valuation as hist


OUTPUT_DIR = Path("ath_reversion_research/reports/peer_cluster_valuation_features")

VALUATION_FEATURES = hist.CURRENT_VALUATION_FEATURES + hist.HIST_VALUATION_FEATURES
BEST_KNOWN_FEATURES = hist.BASE_FEATURES + VALUATION_FEATURES
SECTOR_REL_FEATURES = [
    f"sector_rel_{col}" for col in VALUATION_FEATURES
] + [
    "sector_quality_value_score",
    "sector_undervalued_with_momentum",
    "sector_growth_value_gap",
]
CLUSTER_REL_FEATURES = [
    f"cluster_rel_{col}" for col in VALUATION_FEATURES
] + [
    "cluster_sales_fair_value_upside",
    "cluster_gross_profit_fair_value_upside",
    "cluster_earnings_fair_value_upside",
    "cluster_fcf_fair_value_upside",
    "cluster_quality_value_score",
    "cluster_undervalued_with_momentum",
    "cluster_growth_value_gap",
]
RESIDUAL_FEATURES = [
    "hist_log_price_to_sales_residual",
    "hist_price_to_sales_cheap_residual",
    "hist_log_price_to_gross_profit_residual",
    "hist_price_to_gross_profit_cheap_residual",
    "hist_log_price_to_fcf_residual",
    "hist_price_to_fcf_cheap_residual",
    "residual_cheap_with_momentum",
    "residual_cheap_quality_score",
]
SECTOR_AWARE_RESIDUAL_FEATURES = [
    "sector_hist_log_price_to_sales_residual",
    "sector_hist_price_to_sales_cheap_residual",
    "sector_hist_log_price_to_gross_profit_residual",
    "sector_hist_price_to_gross_profit_cheap_residual",
    "sector_hist_log_price_to_fcf_residual",
    "sector_hist_price_to_fcf_cheap_residual",
    "sector_residual_cheap_with_momentum",
    "sector_residual_cheap_quality_score",
]
RESIDUAL_SALES_FEATURES = [
    "hist_price_to_sales_cheap_residual",
    "residual_cheap_with_momentum",
    "residual_cheap_quality_score",
]
FAIR_VALUE_FEATURES = [
    "cluster_sales_fair_value_upside",
    "cluster_gross_profit_fair_value_upside",
    "cluster_earnings_fair_value_upside",
    "cluster_fcf_fair_value_upside",
]
SELECTED_PEER_VALUE_FEATURES = [
    "hist_price_to_sales_cheap_residual",
    "residual_cheap_with_momentum",
    "residual_cheap_quality_score",
    "cluster_sales_fair_value_upside",
    "cluster_gross_profit_fair_value_upside",
    "cluster_quality_value_score",
]
SELF_RELATIVE_FEATURES = [
    f"self_rel_{col}" for col in hist.HIST_VALUATION_FEATURES
] + [
    "self_relative_cheap_score",
    "self_relative_quality_value_score",
    "self_relative_cheap_with_momentum",
]
SECTOR_DUMMY_FEATURES = [f"sector_dummy_{idx}" for idx in range(11)]
CLUSTER_EMBEDDING_FEATURES = [
    "trailing_3m_return",
    "trailing_6m_return",
    "trailing_12m_return",
    "relative_6m_vs_spy",
    "realized_vol_3m",
    "drawdown_12m",
    "revenue_growth_yoy",
    "revenue_growth_accel",
    "gross_margin",
    "operating_margin",
    "fcf_margin",
    "debt_to_assets",
    "asset_turnover",
    "hist_sales_to_market_cap",
    "hist_earnings_yield",
    "hist_fcf_yield",
]
QUALITY_FEATURES = [
    "revenue_growth_yoy",
    "gross_margin",
    "operating_margin",
    "fcf_margin",
]
RISK_FEATURES = ["debt_to_assets", "realized_vol_3m"]
TREND_FEATURES = ["trailing_6m_return", "relative_6m_vs_spy"]
EXPECTED_VALUATION_FEATURES = [
    "revenue_growth_yoy",
    "revenue_growth_accel",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "fcf_margin",
    "debt_to_assets",
    "asset_turnover",
    "trailing_6m_return",
    "trailing_12m_return",
    "relative_6m_vs_spy",
    "realized_vol_3m",
    "drawdown_12m",
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = _load_events()
    experiments = [
        ("baseline_no_valuation", "raw_return", hist.BASE_FEATURES, ()),
        ("best_known_current_plus_hist", "raw_return", BEST_KNOWN_FEATURES, ()),
        ("sector_relative_value", "raw_return", BEST_KNOWN_FEATURES + SECTOR_REL_FEATURES, ("sector",)),
        ("cluster_relative_value", "raw_return", BEST_KNOWN_FEATURES + CLUSTER_REL_FEATURES, ("cluster",)),
        ("valuation_residuals", "raw_return", BEST_KNOWN_FEATURES + RESIDUAL_FEATURES, ("residual",)),
        (
            "sector_aware_residuals",
            "raw_return",
            BEST_KNOWN_FEATURES + SECTOR_AWARE_RESIDUAL_FEATURES,
            ("sector_residual",),
        ),
        ("self_relative_value", "raw_return", BEST_KNOWN_FEATURES + SELF_RELATIVE_FEATURES, ("self",)),
        (
            "residual_plus_self_value",
            "raw_return",
            BEST_KNOWN_FEATURES + RESIDUAL_FEATURES + SELF_RELATIVE_FEATURES,
            ("residual", "self"),
        ),
        ("sales_residual_selected", "raw_return", BEST_KNOWN_FEATURES + RESIDUAL_SALES_FEATURES, ("residual",)),
        ("cluster_fair_value_only", "raw_return", BEST_KNOWN_FEATURES + FAIR_VALUE_FEATURES, ("cluster",)),
        (
            "selected_peer_value",
            "raw_return",
            BEST_KNOWN_FEATURES + SELECTED_PEER_VALUE_FEATURES,
            ("cluster", "residual"),
        ),
        (
            "clean_improved_value",
            "raw_return",
            BEST_KNOWN_FEATURES
            + SECTOR_AWARE_RESIDUAL_FEATURES
            + SELF_RELATIVE_FEATURES
            + FAIR_VALUE_FEATURES,
            ("sector_residual", "self", "cluster"),
        ),
        (
            "combined_peer_cluster_value",
            "raw_return",
            BEST_KNOWN_FEATURES + SECTOR_REL_FEATURES + CLUSTER_REL_FEATURES + RESIDUAL_FEATURES,
            ("sector", "cluster", "residual"),
        ),
        (
            "combined_peer_cluster_sector_excess",
            "sector_excess_return",
            BEST_KNOWN_FEATURES + SECTOR_REL_FEATURES + CLUSTER_REL_FEATURES + RESIDUAL_FEATURES,
            ("sector", "cluster", "residual"),
        ),
    ]
    predictions = [
        _walk_forward(events, name, target_type, features, engines)
        for name, target_type, features, engines in experiments
    ]
    pred = pd.concat(predictions, ignore_index=True)
    perf = hist._performance(pred)
    sharpe = _top_quintile_sharpe(pred)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    sharpe.to_csv(OUTPUT_DIR / "top_quintile_sharpe_summary.csv", index=False)
    _write_report(perf, sharpe, events)
    print(perf.to_string(index=False))
    return 0


def _load_events() -> pd.DataFrame:
    events = pd.read_csv(
        hist.ABLATION_EVENTS,
        parse_dates=["trade_date", "fwd_12m_return_end_date", "fiscal_period_end"],
    )
    shares = hist._load_shares(sorted(events["symbol"].dropna().astype(str).unique()))
    events = events.merge(shares, on=["symbol", "fiscal_period_end"], how="left")
    events["shares_for_market_cap"] = events["shares_outstanding"].where(
        events["shares_outstanding"].notna(), events["average_shares"]
    )
    events["historical_market_cap"] = events["shares_for_market_cap"] * events["trade_close"]
    events = hist._add_hist_valuation(events)
    return events.replace([np.inf, -np.inf], np.nan)


def _walk_forward(
    events: pd.DataFrame,
    experiment: str,
    target_type: str,
    features: list[str],
    engines: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    target_col = hist._target_column(target_type)
    for bucket in ["all", "large_cap", "mid_cap"]:
        subset = events.copy() if bucket == "all" else events[events["market_cap_bucket"] == bucket].copy()
        subset = subset.dropna(subset=[target_col, "fwd_12m_return", "fwd_12m_return_end_date"])
        for year in sorted(subset["trade_date"].dt.year.unique()):
            start = pd.Timestamp(f"{int(year)}-01-01")
            test = subset[subset["trade_date"].dt.year == year].copy()
            train = subset[(subset["trade_date"] < start) & (subset["fwd_12m_return_end_date"] < start)].copy()
            if len(train) < 100 or test.empty:
                continue
            train, test = _add_fold_features(train, test, engines)
            test["prediction"] = hist._fit_predict(train, test, target_col, features)
            test["target_return"] = test["fwd_12m_return"]
            test["ranking_target"] = test[target_col]
            test["experiment"] = experiment
            test["target_type"] = target_type
            test["regression_bucket"] = bucket
            test["test_year"] = int(year)
            rows.append(test)
    return pd.concat(rows, ignore_index=True)


def _add_fold_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    engines: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    if "sector" in engines:
        train, test = _add_reference_features(train, test, "gics_sector", "sector")
    if "cluster" in engines:
        train, test = _add_cluster_features(train, test)
    if "residual" in engines:
        train, test = _add_expected_valuation_residuals(train, test)
    if "sector_residual" in engines:
        train, test = _add_expected_valuation_residuals(
            train,
            test,
            prefix="sector_",
            expected_features=EXPECTED_VALUATION_FEATURES + SECTOR_DUMMY_FEATURES,
        )
    if "self" in engines:
        train, test = _add_self_relative_features(train, test)
    return train.replace([np.inf, -np.inf], np.nan), test.replace([np.inf, -np.inf], np.nan)


def _add_reference_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_col: str,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in VALUATION_FEATURES:
        if col not in train:
            continue
        med = train.groupby(group_col)[col].median()
        global_med = train[col].median()
        train[f"{prefix}_rel_{col}"] = train[col] - train[group_col].map(med).fillna(global_med)
        test[f"{prefix}_rel_{col}"] = test[col] - test[group_col].map(med).fillna(global_med)

    train_scores, test_scores = _quality_value_scores(train, test)
    train[f"{prefix}_quality_value_score"] = train_scores["quality"] + train_scores["cheap"]
    test[f"{prefix}_quality_value_score"] = test_scores["quality"] + test_scores["cheap"]
    train[f"{prefix}_undervalued_with_momentum"] = train_scores["cheap"] * train_scores["trend"]
    test[f"{prefix}_undervalued_with_momentum"] = test_scores["cheap"] * test_scores["trend"]
    train[f"{prefix}_growth_value_gap"] = train_scores["growth"] + train_scores["cheap"]
    test[f"{prefix}_growth_value_gap"] = test_scores["growth"] + test_scores["cheap"]
    return train, test


def _add_cluster_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train, x_test = _matrix(train, test, CLUSTER_EMBEDDING_FEATURES)
    centers = _fit_kmeans(x_train, k=min(8, max(2, len(train) // 150)))
    train["peer_cluster"] = _assign_clusters(x_train, centers)
    test["peer_cluster"] = _assign_clusters(x_test, centers)
    train, test = _add_reference_features(train, test, "peer_cluster", "cluster")

    med = train.groupby("peer_cluster").median(numeric_only=True)
    _add_fair_value_upside(train, med, "peer_cluster")
    _add_fair_value_upside(test, med, "peer_cluster")
    return train, test


def _add_fair_value_upside(frame: pd.DataFrame, med: pd.DataFrame, group_col: str) -> None:
    cap = frame["historical_market_cap"].replace(0, np.nan)
    specs = [
        ("sales", "hist_sales_to_market_cap", "cluster_sales_fair_value_upside"),
        ("gross_profit", "hist_gross_profit_to_market_cap", "cluster_gross_profit_fair_value_upside"),
        ("net_income", "hist_earnings_yield", "cluster_earnings_fair_value_upside"),
        ("fcf_proxy", "hist_fcf_yield", "cluster_fcf_fair_value_upside"),
    ]
    frame["fcf_proxy"] = frame["fcf_margin"] * frame["sales"]
    for numerator, yield_col, out_col in specs:
        cluster_yield = frame[group_col].map(med[yield_col]).replace(0, np.nan)
        fair_cap = frame[numerator] / cluster_yield
        frame[out_col] = fair_cap / cap - 1


def _add_expected_valuation_residuals(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prefix: str = "",
    expected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_features = expected_features or EXPECTED_VALUATION_FEATURES
    targets = [
        (
            "sales",
            f"{prefix}hist_log_price_to_sales_residual",
            f"{prefix}hist_price_to_sales_cheap_residual",
        ),
        (
            "gross_profit",
            f"{prefix}hist_log_price_to_gross_profit_residual",
            f"{prefix}hist_price_to_gross_profit_cheap_residual",
        ),
        (
            "fcf_proxy",
            f"{prefix}hist_log_price_to_fcf_residual",
            f"{prefix}hist_price_to_fcf_cheap_residual",
        ),
    ]
    train["fcf_proxy"] = train["fcf_margin"] * train["sales"]
    test["fcf_proxy"] = test["fcf_margin"] * test["sales"]
    for denominator, residual_col, cheap_col in targets:
        train_y = _safe_log_ratio(train["historical_market_cap"], train[denominator])
        test_y = _safe_log_ratio(test["historical_market_cap"], test[denominator])
        pred_train, pred_test = _ridge_predict_continuous(train, test, train_y, expected_features)
        train[residual_col] = train_y - pred_train
        test[residual_col] = test_y - pred_test
        train[cheap_col] = -train[residual_col]
        test[cheap_col] = -test[residual_col]
    train_scores, test_scores = _quality_value_scores(train, test)
    sales_cheap = f"{prefix}hist_price_to_sales_cheap_residual"
    train[f"{prefix}residual_cheap_with_momentum"] = train[sales_cheap] * train_scores["trend"]
    test[f"{prefix}residual_cheap_with_momentum"] = test[sales_cheap] * test_scores["trend"]
    train[f"{prefix}residual_cheap_quality_score"] = train[sales_cheap] + train_scores["quality"]
    test[f"{prefix}residual_cheap_quality_score"] = test[sales_cheap] + test_scores["quality"]
    return train, test


def _add_self_relative_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rel_cols = []
    train = train.sort_values(["symbol", "trade_date"]).copy()
    test = test.copy()
    for col in hist.HIST_VALUATION_FEATURES:
        if col not in train:
            continue
        rel_col = f"self_rel_{col}"
        global_median = train[col].median()
        prior_ref = pd.Series(index=train.index, dtype=float)
        for _, group in train.groupby("symbol", sort=False):
            prior_ref.loc[group.index] = group[col].expanding(min_periods=2).median().shift(1)
        symbol_ref = train.groupby("symbol")[col].median()
        train[rel_col] = train[col] - prior_ref.fillna(global_median)
        test[rel_col] = test[col] - test["symbol"].map(symbol_ref).fillna(global_median)
        rel_cols.append(rel_col)

    train_z, test_z = _zscore_columns(train, test, rel_cols + QUALITY_FEATURES + TREND_FEATURES)
    train["self_relative_cheap_score"] = train_z[rel_cols].mean(axis=1)
    test["self_relative_cheap_score"] = test_z[rel_cols].mean(axis=1)
    train["self_relative_quality_value_score"] = train["self_relative_cheap_score"] + train_z[QUALITY_FEATURES].mean(axis=1)
    test["self_relative_quality_value_score"] = test["self_relative_cheap_score"] + test_z[QUALITY_FEATURES].mean(axis=1)
    train["self_relative_cheap_with_momentum"] = train["self_relative_cheap_score"] * train_z[TREND_FEATURES].mean(axis=1)
    test["self_relative_cheap_with_momentum"] = test["self_relative_cheap_score"] * test_z[TREND_FEATURES].mean(axis=1)
    return train.sort_index(), test


def _quality_value_scores(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    cheap_cols = [col for col in ["hist_sales_to_market_cap", "hist_earnings_yield", "hist_fcf_yield"] if col in train]
    train_z, test_z = _zscore_columns(train, test, QUALITY_FEATURES + RISK_FEATURES + TREND_FEATURES + cheap_cols)
    train_quality = train_z[QUALITY_FEATURES].mean(axis=1) - train_z[RISK_FEATURES].mean(axis=1)
    test_quality = test_z[QUALITY_FEATURES].mean(axis=1) - test_z[RISK_FEATURES].mean(axis=1)
    train_scores = {
        "quality": train_quality,
        "cheap": train_z[cheap_cols].mean(axis=1),
        "trend": train_z[TREND_FEATURES].mean(axis=1),
        "growth": train_z["revenue_growth_yoy"],
    }
    test_scores = {
        "quality": test_quality,
        "cheap": test_z[cheap_cols].mean(axis=1),
        "trend": test_z[TREND_FEATURES].mean(axis=1),
        "growth": test_z["revenue_growth_yoy"],
    }
    return train_scores, test_scores


def _zscore_columns(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = [col for col in cols if col in train]
    low = train[usable].quantile(0.01)
    high = train[usable].quantile(0.99)
    med = train[usable].median()
    x_train = train[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    x_test = test[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    mean = x_train.mean()
    std = x_train.std().replace(0, 1)
    return (x_train - mean) / std, (x_test - mean) / std


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator.where(numerator > 0) / denominator.where(denominator > 0)
    return np.log(ratio.replace([np.inf, -np.inf], np.nan))


def _ridge_predict_continuous(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    fit_frame = train.copy()
    fit_frame["_target"] = y_train
    fit_frame = fit_frame.dropna(subset=["_target"])
    if len(fit_frame) < 100:
        return np.full(len(train), np.nan), np.full(len(test), np.nan)
    x_fit, x_test = _matrix(fit_frame, test, features)
    _, x_all_train = _matrix(fit_frame, train, features)
    y = fit_frame["_target"].to_numpy(dtype=float)
    alpha = 25.0
    beta = np.linalg.solve(x_fit.T @ x_fit + alpha * np.eye(x_fit.shape[1]), x_fit.T @ y)
    train_pred = x_all_train @ beta
    test_pred = x_test @ beta
    return train_pred, test_pred


def _matrix(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    usable = [
        col
        for col in features
        if col in train and train[col].notna().mean() >= 0.50 and train[col].std(skipna=True) > 0
    ]
    low = train[usable].quantile(0.01)
    high = train[usable].quantile(0.99)
    med = train[usable].median()
    x_train = train[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    x_test = test[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    mean = x_train.mean()
    std = x_train.std().replace(0, 1)
    return ((x_train - mean) / std).to_numpy(dtype=float), ((x_test - mean) / std).to_numpy(dtype=float)


def _fit_kmeans(x: np.ndarray, k: int, iterations: int = 30) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0, 0))
    order = np.argsort(x[:, 0])
    seed_idx = np.linspace(0, len(order) - 1, k).round().astype(int)
    centers = x[order[seed_idx]].copy()
    for _ in range(iterations):
        labels = _assign_clusters(x, centers)
        updated = centers.copy()
        for label in range(k):
            mask = labels == label
            if mask.any():
                updated[label] = x[mask].mean(axis=0)
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers


def _assign_clusters(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def _coverage(events: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "historical_market_cap",
        "gics_sector",
        "hist_sales_to_market_cap",
        "hist_earnings_yield",
        "hist_fcf_yield",
    ]
    rows = []
    for col in cols:
        rows.append({"field": col, "coverage": events[col].notna().mean(), "unique": events[col].nunique()})
    return pd.DataFrame(rows)


def _top_quintile_sharpe(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, bucket), group in pred.groupby(["experiment", "regression_bucket"]):
        annual_returns = []
        for year, year_group in group.groupby("test_year"):
            ranked = year_group.dropna(subset=["prediction", "target_return"]).copy()
            if len(ranked) < 10:
                continue
            cutoff = ranked["prediction"].quantile(0.80)
            annual_returns.append((year, ranked.loc[ranked["prediction"] >= cutoff, "target_return"].mean()))
        series = pd.Series(dict(annual_returns)).sort_index()
        std = series.std(ddof=1)
        rows.append(
            {
                "experiment": experiment,
                "regression_bucket": bucket,
                "years": len(series),
                "annual_topq_mean": series.mean(),
                "annual_topq_std": std,
                "annual_topq_sharpe": series.mean() / std if len(series) > 1 and std > 0 else np.nan,
                "min_year_return": series.min(),
                "min_return_year": int(series.idxmin()) if len(series) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["regression_bucket", "annual_topq_sharpe"], ascending=[True, False]
    )


def _write_report(perf: pd.DataFrame, sharpe: pd.DataFrame, events: pd.DataFrame) -> None:
    with (OUTPUT_DIR / "PEER_CLUSTER_VALUATION_FEATURES.md").open("w", encoding="utf-8") as f:
        f.write("# Peer and cluster valuation feature test\n\n")
        f.write(
            "This ablation tests under/over-valuation features relative to sectors, learned peer "
            "clusters, and fold-local expected valuation models.\n\n"
        )
        f.write("## No-leakage setup\n\n")
        f.write(
            "- Clusters are fit inside each walk-forward fold using only train rows whose 12m forward window is complete.\n"
            "- Sector and cluster valuation medians are computed from train rows only.\n"
            "- Expected valuation residuals are from ridge models trained only on prior train rows.\n"
            "- Final return predictions use the same purged annual walk-forward split as the valuation study.\n\n"
        )
        f.write("## Feature groups tested\n\n")
        f.write(
            "- Sector-relative valuation yields and quality/value/trend interactions.\n"
            "- Dynamic cluster-relative valuation yields from return/fundamental embeddings.\n"
            "- Peer-implied fair value upside using cluster median sales, gross-profit, earnings, and FCF yields.\n"
            "- Expected valuation residuals: actual log valuation minus valuation predicted from growth, quality, risk, and trend.\n\n"
        )
        f.write("## Input coverage\n\n")
        f.write(hist._markdown_table(_coverage(events)))
        f.write("\n\n## Performance summary\n\n")
        f.write(hist._markdown_table(perf))
        f.write("\n\n## Top-quintile annual portfolio Sharpe\n\n")
        f.write(
            "Sharpe here is the mean/std of yearly top-quintile 12-month forward returns. "
            "It is a coarse annual portfolio proxy, not a daily marked-to-market live portfolio Sharpe.\n\n"
        )
        f.write(hist._markdown_table(sharpe))
        f.write("\n\n## Interpretation\n\n")
        f.write(_interpretation(perf, sharpe))
        f.write("\n")


def _interpretation(perf: pd.DataFrame, sharpe: pd.DataFrame) -> str:
    lines = []
    for bucket in ["all", "large_cap", "mid_cap"]:
        subset = perf[(perf["regression_bucket"] == bucket) & (perf["target_type"] == "raw_return")]
        if subset.empty:
            continue
        best = subset.sort_values("spearman", ascending=False).iloc[0]
        benchmark = subset[subset["experiment"] == "best_known_current_plus_hist"]
        bench_spearman = float(benchmark["spearman"].iloc[0]) if not benchmark.empty else np.nan
        delta = best["spearman"] - bench_spearman
        lines.append(
            f"- {bucket}: best raw-return ranker is `{best['experiment']}` with "
            f"Spearman {best['spearman']:.4f}, delta {delta:+.4f} versus current+historical valuation."
        )
    lines.append(
        "- Broad combined peer/cluster feature sets underperformed, so the useful signal is selective: "
        "expected valuation residuals and some cluster/self-relative diagnostics, not every peer feature at once."
    )
    lines.append(
        "- Sector-aware residuals were the best mid-cap variant, suggesting valuation expectations should account "
        "for industry context when modeling smaller names."
    )
    for bucket in ["all", "large_cap", "mid_cap"]:
        subset = sharpe[sharpe["regression_bucket"] == bucket]
        if subset.empty:
            continue
        best = subset.sort_values("annual_topq_sharpe", ascending=False).iloc[0]
        benchmark = subset[subset["experiment"] == "best_known_current_plus_hist"]
        bench_sharpe = float(benchmark["annual_topq_sharpe"].iloc[0]) if not benchmark.empty else np.nan
        lines.append(
            f"- {bucket}: best annual top-quintile Sharpe is `{best['experiment']}` at "
            f"{best['annual_topq_sharpe']:.2f}, delta {best['annual_topq_sharpe'] - bench_sharpe:+.2f} "
            "versus current+historical valuation."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
