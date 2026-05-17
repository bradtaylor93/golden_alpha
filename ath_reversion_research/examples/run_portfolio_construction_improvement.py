"""Portfolio construction tests for the annual fundamental ranker.

This script keeps the same walk-forward predictions as the peer/valuation tests
and only changes how selected names are weighted. It tests whether rank
confidence, volatility adjustment, sector caps, and cap-bucket sleeves improve
the annual top-quintile portfolio proxy.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import run_historical_market_cap_valuation as hist
import run_peer_cluster_valuation_features as peer


OUTPUT_DIR = Path("ath_reversion_research/reports/portfolio_construction_improvement")
SECTOR_CAP = 0.25
MAX_NAME_WEIGHT = 0.05
SAVE_ROW_LEVEL = os.environ.get("SAVE_ROW_LEVEL_PORTFOLIO_CONSTRUCTION", "").lower() in {"1", "true", "yes"}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = peer._load_events()
    pred = _build_prediction_set(events)
    if SAVE_ROW_LEVEL:
        pred.to_csv(OUTPUT_DIR / "selected_model_predictions.csv", index=False)
    annual, summary = _evaluate_portfolios(pred)
    annual.to_csv(OUTPUT_DIR / "annual_portfolio_returns.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False)
    _write_report(summary, annual)
    print(summary.to_string(index=False))
    return 0


def _build_prediction_set(events: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("all_best_known", peer.BEST_KNOWN_FEATURES, ()),
        (
            "all_sector_aware_residuals",
            peer.BEST_KNOWN_FEATURES + peer.SECTOR_AWARE_RESIDUAL_FEATURES,
            ("sector_residual",),
        ),
        ("large_best_known", peer.BEST_KNOWN_FEATURES, ()),
        (
            "mid_sector_aware_residuals",
            peer.BEST_KNOWN_FEATURES + peer.SECTOR_AWARE_RESIDUAL_FEATURES,
            ("sector_residual",),
        ),
    ]
    frames = []
    for name, features, engines in specs:
        frame = peer._walk_forward(events, name, "raw_return", features, engines)
        if name == "all_best_known":
            frame = frame[frame["regression_bucket"] == "all"]
        elif name == "all_sector_aware_residuals":
            frame = frame[frame["regression_bucket"] == "all"]
        elif name == "large_best_known":
            frame = frame[frame["regression_bucket"] == "large_cap"]
        elif name == "mid_sector_aware_residuals":
            frame = frame[frame["regression_bucket"] == "mid_cap"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _evaluate_portfolios(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_rows = []
    variants = [
        ("all_equal_top20", "all_best_known", "all", "equal"),
        ("all_confidence_top20", "all_best_known", "all", "confidence"),
        ("all_vol_conf_top20", "all_best_known", "all", "vol_conf"),
        ("all_vol_conf_sector_cap", "all_best_known", "all", "vol_conf_sector_cap"),
        ("all_residual_equal_top20", "all_sector_aware_residuals", "all", "equal"),
        ("all_residual_vol_conf_sector_cap", "all_sector_aware_residuals", "all", "vol_conf_sector_cap"),
    ]
    for portfolio, experiment, bucket, method in variants:
        subset = pred[(pred["experiment"] == experiment) & (pred["regression_bucket"] == bucket)]
        annual_rows.extend(_annual_portfolio_rows(subset, portfolio, method))

    annual_rows.extend(_bucket_sleeve_rows(pred, "bucket_equal_sleeves", "equal", dynamic=False))
    annual_rows.extend(_bucket_sleeve_rows(pred, "bucket_vol_conf_sector_cap", "vol_conf_sector_cap", dynamic=False))
    annual_rows.extend(_bucket_sleeve_rows(pred, "bucket_dynamic_sharpe_sleeves", "vol_conf_sector_cap", dynamic=True))
    annual = pd.DataFrame(annual_rows)
    summary = _summary(annual)
    return annual, summary


def _annual_portfolio_rows(subset: pd.DataFrame, portfolio: str, method: str) -> list[dict]:
    rows = []
    for year, group in subset.groupby("test_year"):
        weights = _weights(group, method)
        if weights.empty:
            continue
        aligned = group.loc[weights.index]
        rows.append(
            {
                "portfolio": portfolio,
                "test_year": int(year),
                "return": float((weights * aligned["target_return"]).sum()),
                "holdings": int((weights > 0).sum()),
                "max_weight": float(weights.max()),
                "sector_count": int(aligned.loc[weights > 0, "gics_sector"].nunique()),
                "large_weight": float(weights[aligned["market_cap_bucket"] == "large_cap"].sum()),
                "mid_weight": float(weights[aligned["market_cap_bucket"] == "mid_cap"].sum()),
            }
        )
    return rows


def _bucket_sleeve_rows(pred: pd.DataFrame, portfolio: str, method: str, dynamic: bool) -> list[dict]:
    large = pred[(pred["experiment"] == "large_best_known") & (pred["regression_bucket"] == "large_cap")]
    mid = pred[(pred["experiment"] == "mid_sector_aware_residuals") & (pred["regression_bucket"] == "mid_cap")]
    large_returns = _annual_return_series(large, method)
    mid_returns = _annual_return_series(mid, method)
    rows = []
    for year in sorted(set(large_returns.index).intersection(mid_returns.index)):
        if dynamic:
            large_alloc, mid_alloc = _dynamic_sleeve_weights(large_returns, mid_returns, year)
        else:
            large_alloc, mid_alloc = 0.5, 0.5
        large_weights = _weights(large[large["test_year"] == year], method) * large_alloc
        mid_weights = _weights(mid[mid["test_year"] == year], method) * mid_alloc
        large_group = large[large["test_year"] == year]
        mid_group = mid[mid["test_year"] == year]
        ret = large_returns.loc[year] * large_alloc + mid_returns.loc[year] * mid_alloc
        rows.append(
            {
                "portfolio": portfolio,
                "test_year": int(year),
                "return": float(ret),
                "holdings": int((large_weights > 0).sum() + (mid_weights > 0).sum()),
                "max_weight": float(max(large_weights.max(), mid_weights.max())),
                "sector_count": int(
                    pd.concat(
                        [
                            large_group.loc[large_weights.index[large_weights > 0], "gics_sector"],
                            mid_group.loc[mid_weights.index[mid_weights > 0], "gics_sector"],
                        ]
                    ).nunique()
                ),
                "large_weight": float(large_alloc),
                "mid_weight": float(mid_alloc),
            }
        )
    return rows


def _annual_return_series(subset: pd.DataFrame, method: str) -> pd.Series:
    rows = {}
    for year, group in subset.groupby("test_year"):
        weights = _weights(group, method)
        if not weights.empty:
            rows[int(year)] = float((weights * group.loc[weights.index, "target_return"]).sum())
    return pd.Series(rows).sort_index()


def _dynamic_sleeve_weights(large: pd.Series, mid: pd.Series, year: int) -> tuple[float, float]:
    hist_large = large[large.index < year].tail(3)
    hist_mid = mid[mid.index < year].tail(3)
    if len(hist_large) < 2 or len(hist_mid) < 2:
        return 0.5, 0.5
    large_score = max(hist_large.mean() / hist_large.std(ddof=1), 0.1)
    mid_score = max(hist_mid.mean() / hist_mid.std(ddof=1), 0.1)
    total = large_score + mid_score
    large_weight = float(np.clip(large_score / total, 0.30, 0.70))
    return large_weight, 1 - large_weight


def _weights(group: pd.DataFrame, method: str) -> pd.Series:
    ranked = group.dropna(subset=["prediction", "target_return"]).copy()
    if len(ranked) < 10:
        return pd.Series(dtype=float)
    ranked["rank_pct"] = ranked["prediction"].rank(pct=True)
    selected = ranked[ranked["rank_pct"] >= 0.80].copy()
    if selected.empty:
        return pd.Series(dtype=float)
    if method == "equal":
        raw = pd.Series(1.0, index=selected.index)
    elif method == "confidence":
        raw = (selected["rank_pct"] - 0.80).clip(lower=0.001)
    elif method in {"vol_conf", "vol_conf_sector_cap"}:
        vol = selected["realized_vol_3m"].replace(0, np.nan).fillna(selected["realized_vol_3m"].median())
        raw = (selected["rank_pct"] - 0.80).clip(lower=0.001) / vol.clip(lower=0.05)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    weights = raw / raw.sum()
    weights = _cap_names(weights, MAX_NAME_WEIGHT)
    if method == "vol_conf_sector_cap":
        weights = _cap_sectors(weights, selected["gics_sector"], SECTOR_CAP)
        weights = _cap_names(weights, MAX_NAME_WEIGHT)
    return weights / weights.sum()


def _cap_names(weights: pd.Series, cap: float) -> pd.Series:
    weights = weights.copy()
    for _ in range(20):
        over = weights > cap
        if not over.any():
            break
        excess = (weights[over] - cap).sum()
        weights[over] = cap
        under = ~over
        if not under.any() or weights[under].sum() <= 0:
            break
        weights[under] += excess * weights[under] / weights[under].sum()
    return weights / weights.sum()


def _cap_sectors(weights: pd.Series, sectors: pd.Series, cap: float) -> pd.Series:
    weights = weights.copy()
    for _ in range(20):
        sector_weights = weights.groupby(sectors.loc[weights.index]).sum()
        over_sectors = sector_weights[sector_weights > cap]
        if over_sectors.empty:
            break
        for sector, sector_weight in over_sectors.items():
            idx = weights.index[sectors.loc[weights.index] == sector]
            weights.loc[idx] *= cap / sector_weight
        shortfall = 1 - weights.sum()
        under_idx = weights.index[~sectors.loc[weights.index].isin(over_sectors.index)]
        if len(under_idx) == 0 or weights.loc[under_idx].sum() <= 0:
            break
        weights.loc[under_idx] += shortfall * weights.loc[under_idx] / weights.loc[under_idx].sum()
    return weights / weights.sum()


def _summary(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, group in annual.groupby("portfolio"):
        ret = group.sort_values("test_year")["return"]
        std = ret.std(ddof=1)
        rows.append(
            {
                "portfolio": portfolio,
                "years": len(ret),
                "mean_return": ret.mean(),
                "std_return": std,
                "sharpe": ret.mean() / std if len(ret) > 1 and std > 0 else np.nan,
                "min_return": ret.min(),
                "positive_year_rate": (ret > 0).mean(),
                "avg_holdings": group["holdings"].mean(),
                "avg_sector_count": group["sector_count"].mean(),
                "avg_large_weight": group["large_weight"].mean(),
                "avg_mid_weight": group["mid_weight"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def _write_report(summary: pd.DataFrame, annual: pd.DataFrame) -> None:
    with (OUTPUT_DIR / "PORTFOLIO_CONSTRUCTION_IMPROVEMENT.md").open("w", encoding="utf-8") as f:
        f.write("# Portfolio construction improvement test\n\n")
        f.write(
            "This study keeps the no-leakage annual model predictions fixed and tests weighting/portfolio "
            "construction variants on yearly top-quintile forward returns.\n\n"
        )
        f.write("## Variants\n\n")
        f.write(
            "- Equal-weight top 20% baseline.\n"
            "- Rank-confidence weighting.\n"
            "- Rank-confidence divided by trailing realized volatility.\n"
            "- Volatility-adjusted confidence weights with 5% name cap and 25% sector cap.\n"
            "- Bucket sleeves: large-cap prior-best model plus mid-cap sector-aware residual model.\n"
            "- Dynamic sleeve allocation based only on trailing realized sleeve Sharpe.\n\n"
        )
        f.write(
            "Row-level selected predictions are skipped by default to avoid large generated files. "
            "Set `SAVE_ROW_LEVEL_PORTFOLIO_CONSTRUCTION=1` to export them locally.\n\n"
        )
        f.write("## Summary\n\n")
        f.write(hist._markdown_table(summary))
        f.write("\n\n## Annual returns\n\n")
        f.write(hist._markdown_table(annual.sort_values(["portfolio", "test_year"])))
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
