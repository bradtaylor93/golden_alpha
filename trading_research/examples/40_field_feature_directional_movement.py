"""Directional movement trading test using engineered field features.

Goal:
- Use the existing research field feature pack.
- Train walk-forward directional classifiers (up/down next-day movement).
- Trade off model probabilities (long-short and long-only variants).
- Report overall + yearly performance tables with Sharpe/Sortino/returns.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trading_research.data.vendors.yahoo import YahooMarketDataVendor
from trading_research.features.research_pack import ResearchFeaturePackFamily

UNIVERSE_50 = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "TSLA",
    "BRK-B",
    "JPM",
    "JNJ",
    "V",
    "UNH",
    "HD",
    "PG",
    "MA",
    "XOM",
    "CVX",
    "ABBV",
    "BAC",
    "KO",
    "PEP",
    "COST",
    "AVGO",
    "TMO",
    "MRK",
    "WMT",
    "DIS",
    "ADBE",
    "CRM",
    "NFLX",
    "PFE",
    "CSCO",
    "ACN",
    "MCD",
    "ABT",
    "DHR",
    "LIN",
    "AMD",
    "ORCL",
    "INTC",
    "NKE",
    "WFC",
    "QCOM",
    "TXN",
    "UPS",
    "CAT",
    "GS",
    "HON",
    "LOW",
    "AMGN",
]


@dataclass(frozen=True)
class Config:
    period: str = "max"
    interval: str = "1d"
    start_date: str = "2013-01-01"
    prediction_horizon_days: int = 10
    min_train_years: int = 3
    gross_target: float = 1.0
    max_abs_weight_per_asset: float = 0.04
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    lookbacks: tuple[int, ...] = (5, 10, 20, 40, 80, 120, 200)
    sr_windows: tuple[int, ...] = (40, 80, 120, 200)
    bb_windows: tuple[int, ...] = (40, 80, 120)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    model_name: str  # "logit" | "hgbt"
    long_only: bool
    prob_threshold: float


def _strategy_grid() -> list[StrategySpec]:
    rows: list[StrategySpec] = []
    for model_name in ("logit", "hgbt"):
        for long_only in (False, True):
            for p in (0.52, 0.55, 0.60):
                side = "lo" if long_only else "ls"
                rows.append(
                    StrategySpec(
                        name=f"{model_name}_{side}_p{int(round(p * 100))}",
                        model_name=model_name,
                        long_only=long_only,
                        prob_threshold=float(p),
                    )
                )
    return rows


def _one_way_cost_return(cfg: Config) -> float:
    return (cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side) / 10_000.0


def _annualized_sharpe(net: pd.Series) -> float:
    x = pd.to_numeric(net, errors="coerce").dropna()
    if len(x) < 2:
        return float("nan")
    s = float(x.std(ddof=1))
    if s <= 0.0:
        return float("nan")
    return float((x.mean() / s) * math.sqrt(252.0))


def _annualized_sortino(net: pd.Series) -> float:
    x = pd.to_numeric(net, errors="coerce").dropna()
    if len(x) < 2:
        return float("nan")
    downside = x[x < 0.0]
    if len(downside) < 2:
        return float("nan")
    dd_std = float(downside.std(ddof=1))
    if dd_std <= 0.0:
        return float("nan")
    return float((x.mean() / dd_std) * math.sqrt(252.0))


def _max_drawdown(net: pd.Series) -> float:
    x = pd.to_numeric(net, errors="coerce").fillna(0.0)
    if x.empty:
        return float("nan")
    eq = (1.0 + x).cumprod()
    run = eq.cummax()
    return float((eq / run - 1.0).min())


def _portfolio_metrics(daily: pd.DataFrame, n_signals: float) -> dict[str, float]:
    if daily.empty:
        return {
            "n_days": 0.0,
            "n_signals": n_signals,
            "annual_return": float("nan"),
            "cagr": float("nan"),
            "max_drawdown": float("nan"),
            "annualized_sharpe_net": float("nan"),
            "annualized_sortino_net": float("nan"),
            "avg_turnover": float("nan"),
            "avg_gross_exposure": float("nan"),
        }
    net = pd.to_numeric(daily["net_return"], errors="coerce").fillna(0.0)
    ann = float(np.exp(np.log1p(net).mean() * 252.0) - 1.0)
    start = pd.to_datetime(daily["timestamp"].iloc[0], utc=True)
    end = pd.to_datetime(daily["timestamp"].iloc[-1], utc=True)
    years = max(1e-9, (end - start).total_seconds() / (365.25 * 24 * 3600))
    eq = (1.0 + net).cumprod()
    eq_end = float(eq.iloc[-1]) if not eq.empty else float("nan")
    cagr = float(eq_end ** (1.0 / years) - 1.0) if np.isfinite(eq_end) and eq_end > 0 else float("nan")
    return {
        "n_days": float(len(daily)),
        "n_signals": n_signals,
        "annual_return": ann,
        "cagr": cagr,
        "max_drawdown": _max_drawdown(net),
        "annualized_sharpe_net": _annualized_sharpe(net),
        "annualized_sortino_net": _annualized_sortino(net),
        "avg_turnover": float(pd.to_numeric(daily["turnover"], errors="coerce").mean()),
        "avg_gross_exposure": float(pd.to_numeric(daily["gross_exposure"], errors="coerce").mean()),
    }


def _build_feature_panel(
    bars: pd.DataFrame,
    spy_df: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, list[str]]:
    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["timestamp", "asset", "close"]).sort_values(["asset", "timestamp"])
    bars = bars[bars["timestamp"] >= pd.Timestamp(cfg.start_date, tz="UTC")].copy()
    for c in ("open", "high", "low", "close", "volume"):
        bars[c] = pd.to_numeric(bars[c], errors="coerce")
    bars = bars.dropna(subset=["close", "high", "low"])
    bars = bars[bars["close"] > 0].copy()

    feat_pack = ResearchFeaturePackFamily()
    feat = feat_pack.build(
        bars,
        params={
            "short_window": 12,
            "medium_window": 40,
            "long_window": 120,
            "lookbacks": list(cfg.lookbacks),
            "sr_windows": list(cfg.sr_windows),
            "bb_windows": list(cfg.bb_windows),
            "hit_tolerance": 0.001,
        },
    )

    px = bars[["timestamp", "asset", "close"]].copy()
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["close"]).copy()
    h = int(cfg.prediction_horizon_days)
    px["fwd_ret_h"] = px.groupby("asset", sort=False)["close"].shift(-h) / px["close"] - 1.0
    px["target_up"] = (px["fwd_ret_h"] > 0.0).astype(float)

    panel = feat.merge(px[["timestamp", "asset", "fwd_ret_h", "target_up"]], on=["timestamp", "asset"], how="inner")

    spy = spy_df.copy()
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True, errors="coerce")
    spy = spy.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    spy["close"] = pd.to_numeric(spy["close"], errors="coerce")
    spy = spy.dropna(subset=["close"]).copy()
    spy["spy_fwd_ret_h"] = spy["close"].shift(-h) / spy["close"] - 1.0
    panel = panel.merge(spy[["timestamp", "spy_fwd_ret_h"]], on="timestamp", how="left")
    panel["year"] = panel["timestamp"].dt.year

    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=["fwd_ret_h", "target_up", "spy_fwd_ret_h"]).reset_index(drop=True)

    feature_cols = [c for c in feat.columns if c not in ("timestamp", "asset")]
    return panel, feature_cols


def _fit_predict_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
) -> pd.DataFrame:
    x_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train["target_up"].to_numpy(dtype=int)
    x_test = test[feature_cols].to_numpy(dtype=float)

    if model_name == "logit":
        model: object = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
    elif model_name == "hgbt":
        model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=180,
            min_samples_leaf=200,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    model.fit(x_train, y_train)
    prob_up = model.predict_proba(x_test)[:, 1]
    out = test[["timestamp", "asset", "fwd_ret_h", "spy_fwd_ret_h"]].copy()
    out["prob_up"] = prob_up
    return out


def _weights_from_signals(
    sig_day: pd.DataFrame,
    assets: list[str],
    cfg: Config,
) -> pd.Series:
    w = pd.Series(0.0, index=assets, dtype=float)
    if sig_day.empty:
        return w
    d = sig_day.copy()
    d["raw"] = pd.to_numeric(d["signal"], errors="coerce") * pd.to_numeric(d["strength"], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["asset", "raw"])
    if d.empty:
        return w
    by_asset = d.groupby("asset", as_index=False).agg(raw=("raw", "sum"))
    denom = float(np.sum(np.abs(by_asset["raw"])))
    if denom <= 0.0:
        return w
    by_asset["w"] = (by_asset["raw"] / denom) * cfg.gross_target
    by_asset["w"] = by_asset["w"].clip(-cfg.max_abs_weight_per_asset, cfg.max_abs_weight_per_asset)
    for _, r in by_asset.iterrows():
        a = str(r["asset"])
        if a in w.index:
            w.loc[a] = float(r["w"])
    return w


def _simulate_strategy(
    pred_df: pd.DataFrame,
    spec: StrategySpec,
    cfg: Config,
    fold_id: str,
) -> tuple[pd.DataFrame, float]:
    if pred_df.empty:
        return pd.DataFrame(), 0.0
    p = pred_df.copy()
    p["prob_up"] = pd.to_numeric(p["prob_up"], errors="coerce")
    p["signal"] = 0.0
    if spec.long_only:
        p.loc[p["prob_up"] >= spec.prob_threshold, "signal"] = 1.0
    else:
        p.loc[p["prob_up"] >= spec.prob_threshold, "signal"] = 1.0
        p.loc[p["prob_up"] <= (1.0 - spec.prob_threshold), "signal"] = -1.0
    p["strength"] = (p["prob_up"] - 0.5).abs() * 2.0
    p = p[(p["signal"] != 0.0) & np.isfinite(p["fwd_ret_h"])].copy()

    all_dates = sorted(pred_df["timestamp"].dropna().unique().tolist())
    assets = sorted(pred_df["asset"].dropna().astype(str).unique().tolist())
    if not all_dates or not assets:
        return pd.DataFrame(), 0.0
    by_ts = {ts: g for ts, g in p.groupby("timestamp", sort=True)}
    pred_by_ts = {ts: g for ts, g in pred_df.groupby("timestamp", sort=True)}
    one_way = _one_way_cost_return(cfg)
    prev_w = pd.Series(0.0, index=assets, dtype=float)
    rows: list[dict[str, float | str]] = []

    for ts in all_dates:
        sig_day = by_ts.get(ts)
        w = _weights_from_signals(sig_day=sig_day if sig_day is not None else pd.DataFrame(), assets=assets, cfg=cfg)
        base_day = pred_by_ts.get(ts)
        if base_day is None or base_day.empty:
            continue
        r_map = base_day.set_index("asset")["fwd_ret_h"]
        aligned = r_map.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        gross = float(np.dot(w.to_numpy(dtype=float), aligned))
        turnover = float(np.abs(w - prev_w).sum())
        cost = turnover * one_way
        net = gross - cost
        spy_r = float(pd.to_numeric(base_day["spy_fwd_ret_h"], errors="coerce").dropna().iloc[0])
        rows.append(
            {
                "timestamp": str(ts),
                "net_return": net,
                "spy_return": spy_r,
                "alpha_return": net - spy_r,
                "turnover": turnover,
                "gross_exposure": float(np.abs(w).sum()),
                "strategy": spec.name,
                "fold_id": fold_id,
            }
        )
        prev_w = w

    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily, float(len(p))
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True, errors="coerce")
    daily = daily.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    daily["equity"] = (1.0 + pd.to_numeric(daily["net_return"], errors="coerce").fillna(0.0)).cumprod()
    daily["running_max"] = daily["equity"].cummax()
    daily["drawdown"] = daily["equity"] / daily["running_max"] - 1.0
    return daily, float(len(p))


def _simulate_buy_hold_fold(test: pd.DataFrame, cfg: Config, fold_id: str) -> pd.DataFrame:
    d = test[["timestamp", "asset", "fwd_ret_h", "spy_fwd_ret_h"]].copy()
    d = d.dropna(subset=["timestamp"]).sort_values(["timestamp", "asset"])
    by_ts = d.groupby("timestamp", sort=True)
    rows: list[dict[str, float | str]] = []
    one_way = _one_way_cost_return(cfg)
    did_enter = False
    for ts, g in by_ts:
        r = pd.to_numeric(g["fwd_ret_h"], errors="coerce").dropna()
        if r.empty:
            continue
        gross = float(r.mean())
        turnover = 1.0 if not did_enter else 0.0
        did_enter = True
        net = gross - turnover * one_way
        spy_r = float(pd.to_numeric(g["spy_fwd_ret_h"], errors="coerce").dropna().iloc[0])
        rows.append(
            {
                "timestamp": str(ts),
                "net_return": net,
                "spy_return": spy_r,
                "alpha_return": net - spy_r,
                "turnover": turnover,
                "gross_exposure": 1.0,
                "strategy": "buy_hold_all",
                "fold_id": fold_id,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["equity"] = (1.0 + pd.to_numeric(out["net_return"], errors="coerce").fillna(0.0)).cumprod()
    out["running_max"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["running_max"] - 1.0
    return out


def _yearly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["year"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce").dt.year
    rows: list[dict[str, float | str]] = []
    for (strategy, year), g in d.groupby(["strategy", "year"], sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)
        net = pd.to_numeric(g["net_return"], errors="coerce").fillna(0.0)
        ann = float(np.exp(np.log1p(net).mean() * 252.0) - 1.0) if len(net) > 0 else float("nan")
        rows.append(
            {
                "year": int(year),
                "strategy": str(strategy),
                "n_days": float(len(g)),
                "annual_return": ann,
                "annualized_sharpe_net": _annualized_sharpe(net),
                "annualized_sortino_net": _annualized_sortino(net),
                "max_drawdown": _max_drawdown(net),
                "avg_turnover": float(pd.to_numeric(g["turnover"], errors="coerce").mean()),
                "avg_gross_exposure": float(pd.to_numeric(g["gross_exposure"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy", "year"]).reset_index(drop=True)


def main() -> None:
    out_root = Path("trading_research/examples/_output/40_field_feature_directional_movement")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(UNIVERSE_50, period=cfg.period, interval=cfg.interval)
    spy_df = vendor.fetch_bars(["SPY"], period=cfg.period, interval=cfg.interval)[["timestamp", "close"]]
    if bars.empty or spy_df.empty:
        raise ValueError("Missing Yahoo data for requested universe or SPY benchmark.")

    panel, feature_cols = _build_feature_panel(bars=bars, spy_df=spy_df, cfg=cfg)
    years = sorted(panel["year"].dropna().unique().tolist())
    if len(years) <= cfg.min_train_years:
        raise ValueError("Insufficient years for anchored walk-forward evaluation.")

    specs = _strategy_grid()
    strategy_names = [s.name for s in specs] + ["buy_hold_all"]
    daily_by_strategy: dict[str, list[pd.DataFrame]] = {n: [] for n in strategy_names}
    signal_count_by_strategy: dict[str, list[float]] = {n: [] for n in strategy_names}
    fold_rows: list[dict[str, float | str]] = []

    for fold_idx, test_year in enumerate(years[cfg.min_train_years :], start=1):
        train = panel[panel["year"] < test_year].copy()
        test = panel[panel["year"] == test_year].copy()
        if train.empty or test.empty:
            continue
        fold_id = f"fold_{test_year}"

        pred_by_model: dict[str, pd.DataFrame] = {}
        for m in ("logit", "hgbt"):
            pred_by_model[m] = _fit_predict_fold(train=train, test=test, feature_cols=feature_cols, model_name=m)

        for spec in specs:
            pred = pred_by_model[spec.model_name]
            daily, n_signals = _simulate_strategy(pred_df=pred, spec=spec, cfg=cfg, fold_id=fold_id)
            if daily.empty:
                continue
            daily_by_strategy[spec.name].append(daily)
            signal_count_by_strategy[spec.name].append(n_signals)
            fold_rows.append(
                {
                    "strategy": spec.name,
                    "fold_id": fold_id,
                    "test_year": int(test_year),
                    "train_years": int(train["year"].nunique()),
                    "assets_in_test": int(test["asset"].nunique()),
                    **_portfolio_metrics(daily, n_signals=n_signals),
                }
            )

        bh_daily = _simulate_buy_hold_fold(test=test, cfg=cfg, fold_id=fold_id)
        if not bh_daily.empty:
            daily_by_strategy["buy_hold_all"].append(bh_daily)
            fold_rows.append(
                {
                    "strategy": "buy_hold_all",
                    "fold_id": fold_id,
                    "test_year": int(test_year),
                    "train_years": int(train["year"].nunique()),
                    "assets_in_test": int(test["asset"].nunique()),
                    **_portfolio_metrics(bh_daily, n_signals=0.0),
                }
            )
        print(f"[fold {fold_idx}] {fold_id}: train_rows={len(train)} test_rows={len(test)}")

    fold_df = pd.DataFrame(fold_rows).sort_values(["strategy", "test_year"]).reset_index(drop=True)
    overall_rows: list[dict[str, float | str]] = []
    daily_concat_parts: list[pd.DataFrame] = []
    for name in strategy_names:
        d = pd.concat([x for x in daily_by_strategy[name] if not x.empty], ignore_index=True) if daily_by_strategy[name] else pd.DataFrame()
        if not d.empty:
            d = d.sort_values("timestamp").reset_index(drop=True)
            daily_concat_parts.append(d)
        n_signals = float(np.nansum(signal_count_by_strategy.get(name, [0.0])))
        overall_rows.append({"strategy": name, **_portfolio_metrics(d, n_signals=n_signals)})
    overall_df = pd.DataFrame(overall_rows).sort_values(
        ["annualized_sharpe_net", "annualized_sortino_net", "annual_return"],
        ascending=[False, False, False],
    )

    benchmark = overall_df[overall_df["strategy"] == "buy_hold_all"]
    if benchmark.empty:
        raise ValueError("Missing buy_hold_all benchmark row.")
    b = benchmark.iloc[0]
    overall_df["lookback_months"] = float("nan")
    overall_df["peg_mode"] = "directional_model"
    overall_df["hold_days"] = float(cfg.prediction_horizon_days)
    overall_df.loc[overall_df["strategy"] == "buy_hold_all", "peg_mode"] = "benchmark"
    overall_df.loc[overall_df["strategy"] == "buy_hold_all", "hold_days"] = float("nan")
    overall_df["delta_sharpe_vs_buy_hold"] = overall_df["annualized_sharpe_net"] - float(b["annualized_sharpe_net"])
    overall_df["delta_sortino_vs_buy_hold"] = overall_df["annualized_sortino_net"] - float(b["annualized_sortino_net"])
    overall_df["delta_annual_return_vs_buy_hold"] = overall_df["annual_return"] - float(b["annual_return"])
    overall_df["beats_buy_hold_sharpe"] = overall_df["annualized_sharpe_net"] > float(b["annualized_sharpe_net"])
    overall_df["beats_buy_hold_return"] = overall_df["annual_return"] > float(b["annual_return"])
    overall_df.loc[overall_df["strategy"] == "buy_hold_all", ["delta_sharpe_vs_buy_hold", "delta_sortino_vs_buy_hold", "delta_annual_return_vs_buy_hold"]] = 0.0
    overall_df.loc[overall_df["strategy"] == "buy_hold_all", ["beats_buy_hold_sharpe", "beats_buy_hold_return"]] = False

    daily_all = pd.concat(daily_concat_parts, ignore_index=True) if daily_concat_parts else pd.DataFrame()
    yearly_df = _yearly_metrics(daily_all)

    config_overall = overall_df[overall_df["strategy"] != "buy_hold_all"].copy().reset_index(drop=True)
    cols = [
        "strategy",
        "lookback_months",
        "peg_mode",
        "hold_days",
        "n_days",
        "n_signals",
        "annual_return",
        "cagr",
        "max_drawdown",
        "annualized_sharpe_net",
        "annualized_sortino_net",
        "avg_turnover",
        "avg_gross_exposure",
        "delta_sharpe_vs_buy_hold",
        "delta_sortino_vs_buy_hold",
        "delta_annual_return_vs_buy_hold",
        "beats_buy_hold_sharpe",
        "beats_buy_hold_return",
    ]
    overall_df = overall_df[cols]
    config_overall = config_overall[cols]

    overall_df.to_csv(reports_dir / "overall_performance_with_benchmark.csv", index=False)
    config_overall.to_csv(reports_dir / "config_overall_performance_table.csv", index=False)
    yearly_df.to_csv(reports_dir / "config_yearly_breakdown_table.csv", index=False)
    fold_df.to_csv(reports_dir / "walkforward_fold_metrics.csv", index=False)
    daily_all.to_csv(reports_dir / "daily_all_strategies.csv", index=False)

    summary = {
        "universe_size_requested": len(UNIVERSE_50),
        "universe_size_fetched": int(panel["asset"].nunique()) if not panel.empty else 0,
        "history_start": str(panel["timestamp"].min()) if not panel.empty else None,
        "history_end": str(panel["timestamp"].max()) if not panel.empty else None,
        "n_walkforward_folds": int(fold_df["fold_id"].nunique()) if not fold_df.empty else 0,
        "prediction_horizon_days": cfg.prediction_horizon_days,
        "n_feature_fields": len(feature_cols),
        "strategies_tested": [s.name for s in specs],
        "feature_sample": feature_cols[:25],
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("top overall configs:")
    print(config_overall.sort_values("annualized_sharpe_net", ascending=False).head(12).to_string(index=False))
    print("buy_hold_all:")
    print(overall_df[overall_df["strategy"] == "buy_hold_all"].to_string(index=False))


if __name__ == "__main__":
    main()

