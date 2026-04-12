"""Pegged VWAP month-bag grid test (2013+), with yearly breakdown.

Requested grid:
- Peg lookbacks: 0 (no peg), 3, 6, 9, 12 months
- Peg mode: max / min (for pegged variants; no-peg uses plain rolling VWAP)
- Hold periods: configurable set

Outputs:
- Overall config performance table (Sharpe, Sortino, return, CAGR, max drawdown)
- Yearly breakdown table by config
- Buy-and-hold benchmark and delta columns vs benchmark
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data.vendors.yahoo import YahooMarketDataVendor

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
    rolling_vwap_window: int = 20
    vol_window_days: int = 21
    gross_target: float = 1.0
    max_abs_weight_per_asset: float = 0.05
    transaction_cost_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    vol_floor: float = 1e-4
    lookback_months_grid: tuple[int, ...] = (0, 3, 6, 9, 12)
    hold_days_grid: tuple[int, ...] = (21, 42, 63, 126)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    lookback_months: int
    peg_mode: str  # "none" | "max" | "min"
    hold_days: int


def _one_way_cost_return(cfg: Config) -> float:
    return (cfg.transaction_cost_bps_per_side + cfg.slippage_bps_per_side) / 10_000.0


def _max_drawdown_from_returns(net: pd.Series) -> float:
    x = pd.to_numeric(net, errors="coerce").fillna(0.0)
    if x.empty:
        return float("nan")
    eq = (1.0 + x).cumprod()
    run = eq.cummax()
    return float((eq / run - 1.0).min())


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


def _annualized_sharpe(net: pd.Series) -> float:
    x = pd.to_numeric(net, errors="coerce").dropna()
    if len(x) < 2:
        return float("nan")
    s = float(x.std(ddof=1))
    if s <= 0.0:
        return float("nan")
    return float((x.mean() / s) * math.sqrt(252.0))


def _portfolio_metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {
            "n_days": 0.0,
            "n_trades": float(len(trades)),
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
        "n_trades": float(len(trades)),
        "annual_return": ann,
        "cagr": cagr,
        "max_drawdown": _max_drawdown_from_returns(net),
        "annualized_sharpe_net": _annualized_sharpe(net),
        "annualized_sortino_net": _annualized_sortino(net),
        "avg_turnover": float(pd.to_numeric(daily["turnover"], errors="coerce").mean()),
        "avg_gross_exposure": float(pd.to_numeric(daily["gross_exposure"], errors="coerce").mean()),
    }


def _build_panel(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "asset", "close"]).sort_values(["asset", "timestamp"])
    frame["asset"] = frame["asset"].astype(str)
    for c in ["open", "high", "low", "close", "volume"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame.dropna(subset=["close", "high", "low"])
    frame = frame[frame["close"] > 0].copy()
    frame = frame[frame["timestamp"] >= pd.Timestamp(cfg.start_date, tz="UTC")].copy()

    typ = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    frame["pv"] = typ * frame["volume"].fillna(0.0)
    frame["pv_roll"] = frame.groupby("asset", sort=False)["pv"].transform(
        lambda s: s.rolling(cfg.rolling_vwap_window, min_periods=cfg.rolling_vwap_window).sum()
    )
    frame["v_roll"] = frame.groupby("asset", sort=False)["volume"].transform(
        lambda s: s.fillna(0.0).rolling(cfg.rolling_vwap_window, min_periods=cfg.rolling_vwap_window).sum()
    )
    frame["rolling_vwap"] = frame["pv_roll"] / frame["v_roll"].replace(0.0, np.nan)
    frame["ret_1d"] = frame.groupby("asset", sort=False)["close"].pct_change()
    frame["vol_21"] = frame.groupby("asset", sort=False)["ret_1d"].transform(
        lambda s: s.rolling(cfg.vol_window_days, min_periods=cfg.vol_window_days).std()
    )
    frame["year"] = frame["timestamp"].dt.year
    out = frame.dropna(subset=["rolling_vwap", "vol_21"]).reset_index(drop=True)
    return out


def _build_strategy_grid(cfg: Config) -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    for h in cfg.hold_days_grid:
        specs.append(
            StrategySpec(
                name=f"vwap_nopeg_h{h}",
                lookback_months=0,
                peg_mode="none",
                hold_days=int(h),
            )
        )
    for m in cfg.lookback_months_grid:
        if m <= 0:
            continue
        for mode in ("max", "min"):
            for h in cfg.hold_days_grid:
                specs.append(
                    StrategySpec(
                        name=f"vwap_peg_{mode}_{m}m_h{h}",
                        lookback_months=int(m),
                        peg_mode=mode,
                        hold_days=int(h),
                    )
                )
    return specs


def _level_for_spec(asset_df: pd.DataFrame, spec: StrategySpec) -> pd.Series:
    rv = pd.to_numeric(asset_df["rolling_vwap"], errors="coerce")
    if spec.lookback_months <= 0 or spec.peg_mode == "none":
        return rv
    look = int(round(spec.lookback_months * 21))
    if spec.peg_mode == "max":
        return rv.rolling(look, min_periods=look).max().shift(1)
    if spec.peg_mode == "min":
        return rv.rolling(look, min_periods=look).min().shift(1)
    raise ValueError(f"Unsupported peg mode: {spec.peg_mode}")


def _build_trades(panel: pd.DataFrame, spec: StrategySpec, cfg: Config) -> pd.DataFrame:
    all_dates = sorted(panel["timestamp"].unique().tolist())
    date_to_idx = {ts: i for i, ts in enumerate(all_dates)}
    n_dates = len(all_dates)
    if n_dates < 3:
        return pd.DataFrame()

    rows: list[dict[str, float | str]] = []
    for asset, g in panel.groupby("asset", sort=False):
        gf = g.sort_values("timestamp").copy()
        level = _level_for_spec(gf, spec)
        close = pd.to_numeric(gf["close"], errors="coerce")
        prev_close = close.shift(1)
        prev_level = level.shift(1)

        long_sig = (prev_close <= prev_level) & (close > level)
        short_sig = (prev_close >= prev_level) & (close < level)

        last_end = -1
        for i in range(len(gf)):
            if not bool(long_sig.iloc[i]) and not bool(short_sig.iloc[i]):
                continue
            ts = gf["timestamp"].iloc[i]
            sig_idx = date_to_idx.get(ts)
            if sig_idx is None or sig_idx <= last_end:
                continue
            sign = 1.0 if bool(long_sig.iloc[i]) else -1.0
            lvl = float(level.iloc[i]) if np.isfinite(level.iloc[i]) else float("nan")
            cls = float(close.iloc[i]) if np.isfinite(close.iloc[i]) else float("nan")
            if not (np.isfinite(lvl) and np.isfinite(cls) and lvl > 0 and cls > 0):
                continue
            start_idx = sig_idx + 1
            end_idx = min(sig_idx + int(spec.hold_days), n_dates - 1)
            if start_idx >= n_dates or end_idx <= start_idx:
                continue
            strength = float(max(1e-6, abs(cls / lvl - 1.0)))
            vol = float(gf["vol_21"].iloc[i]) if np.isfinite(gf["vol_21"].iloc[i]) else cfg.vol_floor
            rows.append(
                {
                    "strategy": spec.name,
                    "asset": str(asset),
                    "signal_timestamp": str(ts),
                    "start_idx": float(start_idx),
                    "end_idx": float(end_idx),
                    "signal_sign": float(sign),
                    "signal_strength": strength,
                    "risk_scale": float(1.0 / max(cfg.vol_floor, vol)),
                }
            )
            last_end = end_idx
    return pd.DataFrame(rows)


def _weights_from_active(active: pd.DataFrame, cfg: Config, assets: list[str]) -> pd.Series:
    w = pd.Series(0.0, index=assets, dtype=float)
    if active.empty:
        return w
    a = active.copy()
    for c in ("signal_sign", "signal_strength", "risk_scale"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a.dropna(subset=["signal_sign", "signal_strength", "risk_scale"])
    if a.empty:
        return w
    a["raw_signed"] = a["signal_sign"] * a["signal_strength"] * a["risk_scale"]
    by_asset = a.groupby("asset", as_index=False).agg(raw=("raw_signed", "sum"))
    denom = float(np.sum(np.abs(by_asset["raw"])))
    if denom <= 0.0:
        return w
    by_asset["weight"] = (by_asset["raw"] / denom) * cfg.gross_target
    by_asset["weight"] = by_asset["weight"].clip(-cfg.max_abs_weight_per_asset, cfg.max_abs_weight_per_asset)
    for _, r in by_asset.iterrows():
        asset = str(r["asset"])
        if asset in w.index:
            w.loc[asset] = float(r["weight"])
    return w


def _simulate_strategy(
    panel: pd.DataFrame,
    trades: pd.DataFrame,
    cfg: Config,
    strategy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = panel.pivot(index="timestamp", columns="asset", values="close").sort_index().ffill()
    rets = close.pct_change().fillna(0.0)
    dates = rets.index.to_list()
    assets = rets.columns.to_list()
    if len(dates) < 2:
        return pd.DataFrame(), pd.DataFrame()

    t = trades.copy()
    if t.empty:
        out = pd.DataFrame({"timestamp": rets.index, "net_return": 0.0, "turnover": 0.0, "gross_exposure": 0.0})
        out["strategy"] = strategy_name
        out["equity"] = (1.0 + out["net_return"]).cumprod()
        out["running_max"] = out["equity"].cummax()
        out["drawdown"] = out["equity"] / out["running_max"] - 1.0
        return out, pd.DataFrame()

    t["start_idx"] = pd.to_numeric(t["start_idx"], errors="coerce").astype("Int64")
    t["end_idx"] = pd.to_numeric(t["end_idx"], errors="coerce").astype("Int64")
    t = t.dropna(subset=["start_idx", "end_idx", "asset", "signal_sign", "signal_strength", "risk_scale"]).copy()

    prev_w = pd.Series(0.0, index=assets, dtype=float)
    one_way = _one_way_cost_return(cfg)
    rows: list[dict[str, float | str]] = []

    for i in range(1, len(dates)):
        active = t[(t["start_idx"] <= i) & (t["end_idx"] >= i)]
        w = _weights_from_active(active=active, cfg=cfg, assets=assets)
        gross = float(np.dot(w.values, rets.iloc[i].reindex(assets).fillna(0.0).values))
        turnover = float(np.abs(w - prev_w).sum())
        cost = turnover * one_way
        net = gross - cost
        rows.append(
            {
                "timestamp": str(dates[i]),
                "net_return": net,
                "turnover": turnover,
                "gross_exposure": float(np.abs(w).sum()),
                "strategy": strategy_name,
            }
        )
        prev_w = w

    daily = pd.DataFrame(rows)
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True, errors="coerce")
    daily = daily.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    daily["running_max"] = daily["equity"].cummax()
    daily["drawdown"] = daily["equity"] / daily["running_max"] - 1.0

    trade_rows: list[dict[str, float | str]] = []
    for _, r in t.iterrows():
        sidx = int(r["start_idx"])
        eidx = int(r["end_idx"])
        asset = str(r["asset"])
        if sidx < 0 or eidx >= len(dates) or sidx >= eidx or asset not in close.columns:
            continue
        entry_ts = dates[sidx]
        exit_ts = dates[eidx]
        entry_px = float(close.loc[entry_ts, asset])
        exit_px = float(close.loc[exit_ts, asset])
        sign = float(r["signal_sign"])
        gross_ret = sign * (exit_px / entry_px - 1.0)
        net_ret = gross_ret - 2.0 * _one_way_cost_return(cfg)
        trade_rows.append(
            {
                "strategy": strategy_name,
                "asset": asset,
                "entry_timestamp": str(entry_ts),
                "exit_timestamp": str(exit_ts),
                "hold_days": float(eidx - sidx),
                "net_return": net_ret,
            }
        )
    return daily, pd.DataFrame(trade_rows)


def _simulate_buy_hold_all(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    close = panel.pivot(index="timestamp", columns="asset", values="close").sort_index().ffill()
    rets = close.pct_change().fillna(0.0)
    if rets.empty:
        return pd.DataFrame()
    n_assets = rets.shape[1]
    if n_assets == 0:
        return pd.DataFrame()
    w = np.repeat(1.0 / n_assets, n_assets)
    gross = rets.to_numpy() @ w
    turnover = np.zeros_like(gross)
    if len(gross) > 0:
        turnover[0] = float(np.sum(np.abs(w)))
    net = gross - turnover * _one_way_cost_return(cfg)
    out = pd.DataFrame(
        {
            "timestamp": rets.index,
            "net_return": net,
            "turnover": turnover,
            "gross_exposure": 1.0,
            "strategy": "buy_hold_all",
        }
    )
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    out["running_max"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["running_max"] - 1.0
    return out.reset_index(drop=True)


def _yearly_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    d = daily.copy()
    d["year"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce").dt.year
    rows: list[dict[str, float | str]] = []
    for y, g in d.groupby("year", sort=True):
        x = pd.to_numeric(g["net_return"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "year": int(y),
                "n_days": float(len(g)),
                "annual_return": float(np.exp(np.log1p(x).mean() * 252.0) - 1.0),
                "annualized_sharpe_net": _annualized_sharpe(x),
                "annualized_sortino_net": _annualized_sortino(x),
                "max_drawdown": _max_drawdown_from_returns(x),
                "avg_turnover": float(pd.to_numeric(g["turnover"], errors="coerce").mean()),
                "avg_gross_exposure": float(pd.to_numeric(g["gross_exposure"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def main() -> None:
    out_root = Path("trading_research/examples/_output/39_pegged_vwap_month_bag_test")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(UNIVERSE_50, period=cfg.period, interval=cfg.interval)
    if bars.empty:
        raise ValueError("No bars fetched from Yahoo for requested universe.")
    panel = _build_panel(bars, cfg=cfg)
    if panel.empty:
        raise ValueError("Panel is empty after preprocessing.")

    specs = _build_strategy_grid(cfg)
    all_daily: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    overall_rows: list[dict[str, float | str]] = []
    yearly_rows: list[pd.DataFrame] = []

    buy_hold_daily = _simulate_buy_hold_all(panel=panel, cfg=cfg)
    bh_metrics = _portfolio_metrics(buy_hold_daily, pd.DataFrame())
    overall_rows.append(
        {
            "strategy": "buy_hold_all",
            "lookback_months": float("nan"),
            "peg_mode": "benchmark",
            "hold_days": float("nan"),
            **bh_metrics,
        }
    )
    if not buy_hold_daily.empty:
        y_bh = _yearly_metrics(buy_hold_daily)
        if not y_bh.empty:
            y_bh["strategy"] = "buy_hold_all"
            y_bh["lookback_months"] = float("nan")
            y_bh["peg_mode"] = "benchmark"
            y_bh["hold_days"] = float("nan")
            yearly_rows.append(y_bh)
        all_daily.append(buy_hold_daily)

    for spec in specs:
        trades = _build_trades(panel=panel, spec=spec, cfg=cfg)
        daily, trade_eval = _simulate_strategy(panel=panel, trades=trades, cfg=cfg, strategy_name=spec.name)
        m = _portfolio_metrics(daily, trade_eval)
        overall_rows.append(
            {
                "strategy": spec.name,
                "lookback_months": float(spec.lookback_months),
                "peg_mode": spec.peg_mode,
                "hold_days": float(spec.hold_days),
                **m,
            }
        )
        if not daily.empty:
            dcfg = daily.copy()
            dcfg["lookback_months"] = float(spec.lookback_months)
            dcfg["peg_mode"] = spec.peg_mode
            dcfg["hold_days"] = float(spec.hold_days)
            all_daily.append(dcfg)
            y = _yearly_metrics(daily)
            if not y.empty:
                y["strategy"] = spec.name
                y["lookback_months"] = float(spec.lookback_months)
                y["peg_mode"] = spec.peg_mode
                y["hold_days"] = float(spec.hold_days)
                yearly_rows.append(y)
        if not trade_eval.empty:
            tcfg = trade_eval.copy()
            tcfg["lookback_months"] = float(spec.lookback_months)
            tcfg["peg_mode"] = spec.peg_mode
            tcfg["hold_days"] = float(spec.hold_days)
            all_trades.append(tcfg)
        print(f"[done] {spec.name}: trades={len(trades)}")

    overall_df = pd.DataFrame(overall_rows)
    if overall_df.empty:
        raise ValueError("No overall results produced.")

    bh = overall_df[overall_df["strategy"] == "buy_hold_all"]
    if not bh.empty:
        b = bh.iloc[0]
        overall_df["delta_sharpe_vs_buy_hold"] = pd.to_numeric(overall_df["annualized_sharpe_net"], errors="coerce") - float(
            b["annualized_sharpe_net"]
        )
        overall_df["delta_sortino_vs_buy_hold"] = pd.to_numeric(overall_df["annualized_sortino_net"], errors="coerce") - float(
            b["annualized_sortino_net"]
        )
        overall_df["delta_annual_return_vs_buy_hold"] = pd.to_numeric(overall_df["annual_return"], errors="coerce") - float(
            b["annual_return"]
        )
        overall_df["beats_buy_hold_sharpe"] = overall_df["delta_sharpe_vs_buy_hold"] > 0.0
        overall_df["beats_buy_hold_return"] = overall_df["delta_annual_return_vs_buy_hold"] > 0.0

    overall_df = overall_df.sort_values(
        ["annualized_sharpe_net", "annualized_sortino_net", "annual_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    config_only_df = overall_df[overall_df["strategy"] != "buy_hold_all"].copy()

    yearly_df = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
    if not yearly_df.empty:
        yearly_df = yearly_df.sort_values(["strategy", "year"]).reset_index(drop=True)

    daily_all = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    best = config_only_df.iloc[0].to_dict() if not config_only_df.empty else {}
    summary = {
        "universe_size_requested": len(UNIVERSE_50),
        "universe_size_fetched": int(panel["asset"].nunique()) if not panel.empty else 0,
        "history_start": str(panel["timestamp"].min()) if not panel.empty else None,
        "history_end": str(panel["timestamp"].max()) if not panel.empty else None,
        "n_configs_tested": int(len(config_only_df)),
        "best_config": best,
        "config": {
            "period": cfg.period,
            "interval": cfg.interval,
            "start_date": cfg.start_date,
            "rolling_vwap_window": cfg.rolling_vwap_window,
            "lookback_months_grid": list(cfg.lookback_months_grid),
            "hold_days_grid": list(cfg.hold_days_grid),
            "one_way_cost_return": _one_way_cost_return(cfg),
        },
    }

    overall_df.to_csv(reports_dir / "overall_performance_with_benchmark.csv", index=False)
    config_only_df.to_csv(reports_dir / "config_overall_performance_table.csv", index=False)
    yearly_df.to_csv(reports_dir / "config_yearly_breakdown_table.csv", index=False)
    daily_all.to_csv(reports_dir / "daily_all_strategies.csv", index=False)
    trades_all.to_csv(reports_dir / "trades_all_configs.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("top overall configs:")
    print(config_only_df.head(15).to_string(index=False))
    if not bh.empty:
        print("buy_hold_all:")
        print(bh.to_string(index=False))


if __name__ == "__main__":
    main()

