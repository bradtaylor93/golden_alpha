"""Compare fixed-horizon vs inverse-signal exits for the best pegged-cross entry.

Best entry from prior study:
- Long when close crosses above 1-month pegged high.

This script compares two exit policies on the same OOS split:
1) fixed-horizon exit (63 trading days),
2) inverse exit (first cross below 1-month pegged low).

Trade simulation uses non-overlapping trades per asset to keep comparison fair.
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
class ExitStudyConfig:
    test_ratio: float = 0.30
    peg_window_days: int = 21
    fixed_horizon_days: int = 63
    trailing_stop_pct: float = 0.10


def _t_stat(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) < 2:
        return float("nan")
    std = float(x.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(x.mean() / std * math.sqrt(len(x)))


def _split_frame(panel: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = pd.Index(sorted(panel["timestamp"].dropna().unique()))
    if len(dates) < 10:
        raise ValueError("Insufficient timestamps for train/test split.")
    split_idx = int((1.0 - test_ratio) * len(dates))
    split_idx = min(max(split_idx, 1), len(dates) - 1)
    split_ts = pd.Timestamp(dates[split_idx])
    train = panel[panel["timestamp"] < split_ts].copy()
    test = panel[panel["timestamp"] >= split_ts].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty partition.")
    return train, test, split_ts


def _build_panel(bars: pd.DataFrame, cfg: ExitStudyConfig) -> pd.DataFrame:
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "asset", "close"]).sort_values(["asset", "timestamp"])
    frame["asset"] = frame["asset"].astype(str)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["high"] = pd.to_numeric(frame["high"], errors="coerce")
    frame["low"] = pd.to_numeric(frame["low"], errors="coerce")
    frame = frame.dropna(subset=["close", "high", "low"])
    frame = frame[frame["close"] > 0].copy()

    g = frame.groupby("asset", sort=False)
    lb = cfg.peg_window_days
    frame["pegged_high_1m"] = g["high"].rolling(lb, min_periods=lb).max().shift(1).reset_index(level=0, drop=True)
    frame["pegged_low_1m"] = g["low"].rolling(lb, min_periods=lb).min().shift(1).reset_index(level=0, drop=True)

    prev_close = g["close"].shift(1)
    frame["entry_long"] = ((prev_close <= frame["pegged_high_1m"]) & (frame["close"] > frame["pegged_high_1m"])).astype(int)
    frame["inverse_exit"] = ((prev_close >= frame["pegged_low_1m"]) & (frame["close"] < frame["pegged_low_1m"])).astype(int)

    out = frame[
        [
            "timestamp",
            "asset",
            "close",
            "entry_long",
            "inverse_exit",
        ]
    ].dropna()
    return out.reset_index(drop=True)


def _simulate_non_overlapping_trades(
    asset_df: pd.DataFrame,
    *,
    mode: str,
    fixed_horizon_days: int,
    use_fixed_horizon: bool,
    use_inverse_exit: bool,
    trailing_stop_pct: float | None,
) -> pd.DataFrame:
    df = asset_df.sort_values("timestamp").reset_index(drop=True)
    trades: list[dict[str, float | str]] = []

    i = 1
    n = len(df)
    while i < n - 1:
        if int(df.loc[i, "entry_long"]) != 1:
            i += 1
            continue

        entry_idx = i
        entry_px = float(df.loc[entry_idx, "close"])
        entry_ts = pd.Timestamp(df.loc[entry_idx, "timestamp"])
        max_exit_idx = min(entry_idx + fixed_horizon_days, n - 1) if use_fixed_horizon else n - 1

        peak_px = entry_px
        exit_idx = max_exit_idx
        exit_reason = "fixed_horizon" if use_fixed_horizon else "end_of_sample"
        found_exit = False

        for j in range(entry_idx + 1, max_exit_idx + 1):
            px_j = float(df.loc[j, "close"])
            if px_j > peak_px:
                peak_px = px_j

            trailing_hit = False
            if trailing_stop_pct is not None and trailing_stop_pct > 0:
                stop_level = peak_px * (1.0 - trailing_stop_pct)
                trailing_hit = px_j <= stop_level

            inverse_hit = use_inverse_exit and int(df.loc[j, "inverse_exit"]) == 1

            if trailing_hit or inverse_hit:
                exit_idx = j
                if trailing_hit and inverse_hit:
                    exit_reason = "trailing_stop_and_inverse_cross"
                elif trailing_hit:
                    exit_reason = "trailing_stop"
                else:
                    exit_reason = "inverse_cross"
                found_exit = True
                break

        if exit_idx <= entry_idx:
            i += 1
            continue

        exit_px = float(df.loc[exit_idx, "close"])
        exit_ts = pd.Timestamp(df.loc[exit_idx, "timestamp"])
        hold_days = int(exit_idx - entry_idx)
        ret = exit_px / entry_px - 1.0
        daily_log = np.log1p(ret) / hold_days if hold_days > 0 and ret > -1.0 else float("nan")

        trades.append(
            {
                "asset": str(df.loc[entry_idx, "asset"]),
                "entry_timestamp": str(entry_ts),
                "exit_timestamp": str(exit_ts),
                "entry_price": entry_px,
                "exit_price": exit_px,
                "hold_days": float(hold_days),
                "trade_return": float(ret),
                "daily_log_return": float(daily_log),
                "mode": mode,
                "exit_reason": exit_reason if found_exit or use_fixed_horizon else "end_of_sample",
            }
        )
        i = exit_idx + 1

    return pd.DataFrame(trades)


def _summarize(trades: pd.DataFrame) -> dict[str, float | str]:
    if trades.empty:
        return {
            "n_trades": 0.0,
            "mean_trade_return": float("nan"),
            "median_trade_return": float("nan"),
            "win_rate": float("nan"),
            "t_stat_trade_return": float("nan"),
            "mean_hold_days": float("nan"),
            "median_hold_days": float("nan"),
            "annualized_log_return_proxy": float("nan"),
            "compound_return_seq_proxy": float("nan"),
            "p05_trade_return": float("nan"),
            "cvar10_trade_return": float("nan"),
            "worst_trade_return": float("nan"),
        }

    r = pd.to_numeric(trades["trade_return"], errors="coerce").dropna()
    h = pd.to_numeric(trades["hold_days"], errors="coerce").dropna()
    dlog = pd.to_numeric(trades["daily_log_return"], errors="coerce").dropna()
    q10 = float(r.quantile(0.10)) if not r.empty else float("nan")
    cvar10 = float(r[r <= q10].mean()) if not r.empty else float("nan")
    compound = float((1.0 + r).prod() - 1.0) if not r.empty else float("nan")
    ann_log = float(np.exp(dlog.mean() * 252.0) - 1.0) if not dlog.empty else float("nan")
    return {
        "n_trades": float(len(trades)),
        "mean_trade_return": float(r.mean()) if not r.empty else float("nan"),
        "median_trade_return": float(r.median()) if not r.empty else float("nan"),
        "win_rate": float((r > 0).mean()) if not r.empty else float("nan"),
        "t_stat_trade_return": _t_stat(r),
        "mean_hold_days": float(h.mean()) if not h.empty else float("nan"),
        "median_hold_days": float(h.median()) if not h.empty else float("nan"),
        "annualized_log_return_proxy": ann_log,
        "compound_return_seq_proxy": compound,
        "p05_trade_return": float(r.quantile(0.05)) if not r.empty else float("nan"),
        "cvar10_trade_return": cvar10,
        "worst_trade_return": float(r.min()) if not r.empty else float("nan"),
    }


def _asset_level_comparison(fixed: pd.DataFrame, inverse: pd.DataFrame) -> pd.DataFrame:
    def _asset_stats(df: pd.DataFrame, mode: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["asset", f"n_trades_{mode}", f"mean_ret_{mode}", f"win_rate_{mode}"])
        out = (
            df.groupby("asset", as_index=False)
            .agg(
                n_trades=("trade_return", "count"),
                mean_ret=("trade_return", "mean"),
                win_rate=("trade_return", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
            )
            .rename(
                columns={
                    "n_trades": f"n_trades_{mode}",
                    "mean_ret": f"mean_ret_{mode}",
                    "win_rate": f"win_rate_{mode}",
                }
            )
        )
        return out

    af = _asset_stats(fixed, "fixed")
    ai = _asset_stats(inverse, "inverse")
    joined = af.merge(ai, on="asset", how="outer")
    joined["delta_mean_ret_inverse_minus_fixed"] = joined["mean_ret_inverse"] - joined["mean_ret_fixed"]
    joined["delta_win_rate_inverse_minus_fixed"] = joined["win_rate_inverse"] - joined["win_rate_fixed"]
    return joined.sort_values("delta_mean_ret_inverse_minus_fixed", ascending=False).reset_index(drop=True)


def _slugify(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("%", "pct")
        .replace("__", "_")
    )


def main() -> None:
    out_root = Path("trading_research/examples/_output/18_best_pegged_inverse_exit_study")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExitStudyConfig()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(UNIVERSE_50, period="3y", interval="1d")
    if bars.empty:
        raise ValueError("No Yahoo bars returned for requested universe.")

    panel = _build_panel(bars, cfg)
    train, test, split_ts = _split_frame(panel, cfg.test_ratio)

    strategy_defs = [
        {
            "name": "fixed_horizon_63d",
            "use_fixed_horizon": True,
            "use_inverse_exit": False,
            "trailing_stop_pct": None,
        },
        {
            "name": "inverse_exit_cross_below_1m_low",
            "use_fixed_horizon": False,
            "use_inverse_exit": True,
            "trailing_stop_pct": None,
        },
        {
            "name": f"fixed_horizon_63d_with_trailing_{int(cfg.trailing_stop_pct*100)}pct",
            "use_fixed_horizon": True,
            "use_inverse_exit": False,
            "trailing_stop_pct": cfg.trailing_stop_pct,
        },
        {
            "name": f"inverse_exit_with_trailing_{int(cfg.trailing_stop_pct*100)}pct",
            "use_fixed_horizon": False,
            "use_inverse_exit": True,
            "trailing_stop_pct": cfg.trailing_stop_pct,
        },
    ]

    strategy_trades: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, float | str]] = []
    for sdef in strategy_defs:
        frames: list[pd.DataFrame] = []
        for _, g in test.groupby("asset", sort=False):
            frames.append(
                _simulate_non_overlapping_trades(
                    g,
                    mode=str(sdef["name"]),
                    fixed_horizon_days=cfg.fixed_horizon_days,
                    use_fixed_horizon=bool(sdef["use_fixed_horizon"]),
                    use_inverse_exit=bool(sdef["use_inverse_exit"]),
                    trailing_stop_pct=sdef["trailing_stop_pct"],
                )
            )
        trades_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        strategy_trades[str(sdef["name"])] = trades_df
        summary_rows.append({"strategy": str(sdef["name"]), **_summarize(trades_df)})

    comparison = pd.DataFrame(summary_rows)
    fixed_summary = comparison.loc[comparison["strategy"] == "fixed_horizon_63d"].iloc[0].to_dict()
    inverse_summary = comparison.loc[comparison["strategy"] == "inverse_exit_cross_below_1m_low"].iloc[0].to_dict()
    fixed_trailing_summary = (
        comparison.loc[
            comparison["strategy"] == f"fixed_horizon_63d_with_trailing_{int(cfg.trailing_stop_pct*100)}pct"
        ].iloc[0].to_dict()
    )
    inverse_trailing_summary = (
        comparison.loc[
            comparison["strategy"] == f"inverse_exit_with_trailing_{int(cfg.trailing_stop_pct*100)}pct"
        ].iloc[0].to_dict()
    )

    uplift = {
        "delta_mean_trade_return_inverse_minus_fixed": float(
            float(inverse_summary["mean_trade_return"]) - float(fixed_summary["mean_trade_return"])
        ),
        "delta_win_rate_inverse_minus_fixed": float(float(inverse_summary["win_rate"]) - float(fixed_summary["win_rate"])),
        "delta_t_stat_inverse_minus_fixed": float(
            float(inverse_summary["t_stat_trade_return"]) - float(fixed_summary["t_stat_trade_return"])
        ),
        "delta_annualized_log_return_proxy_inverse_minus_fixed": float(
            float(inverse_summary["annualized_log_return_proxy"]) - float(fixed_summary["annualized_log_return_proxy"])
        ),
        "delta_mean_trade_return_fixed_trailing_minus_fixed": float(
            float(fixed_trailing_summary["mean_trade_return"]) - float(fixed_summary["mean_trade_return"])
        ),
        "delta_mean_trade_return_inverse_trailing_minus_inverse": float(
            float(inverse_trailing_summary["mean_trade_return"]) - float(inverse_summary["mean_trade_return"])
        ),
        "delta_win_rate_fixed_trailing_minus_fixed": float(
            float(fixed_trailing_summary["win_rate"]) - float(fixed_summary["win_rate"])
        ),
        "delta_win_rate_inverse_trailing_minus_inverse": float(
            float(inverse_trailing_summary["win_rate"]) - float(inverse_summary["win_rate"])
        ),
    }

    fixed_df = strategy_trades["fixed_horizon_63d"]
    inverse_df = strategy_trades["inverse_exit_cross_below_1m_low"]
    fixed_trailing_df = strategy_trades[f"fixed_horizon_63d_with_trailing_{int(cfg.trailing_stop_pct*100)}pct"]
    inverse_trailing_df = strategy_trades[f"inverse_exit_with_trailing_{int(cfg.trailing_stop_pct*100)}pct"]

    asset_cmp = _asset_level_comparison(fixed_df, inverse_df)
    if not asset_cmp.empty:
        uplift["pct_assets_inverse_better_mean_return"] = float(
            (asset_cmp["delta_mean_ret_inverse_minus_fixed"] > 0).mean()
        )
    else:
        uplift["pct_assets_inverse_better_mean_return"] = float("nan")

    asset_cmp_fixed_trailing = _asset_level_comparison(fixed_df, fixed_trailing_df).rename(
        columns={
            "n_trades_inverse": "n_trades_fixed_trailing",
            "mean_ret_inverse": "mean_ret_fixed_trailing",
            "win_rate_inverse": "win_rate_fixed_trailing",
            "delta_mean_ret_inverse_minus_fixed": "delta_mean_ret_fixed_trailing_minus_fixed",
            "delta_win_rate_inverse_minus_fixed": "delta_win_rate_fixed_trailing_minus_fixed",
        }
    )
    asset_cmp_inverse_trailing = _asset_level_comparison(inverse_df, inverse_trailing_df).rename(
        columns={
            "n_trades_fixed": "n_trades_inverse",
            "mean_ret_fixed": "mean_ret_inverse",
            "win_rate_fixed": "win_rate_inverse",
            "n_trades_inverse": "n_trades_inverse_trailing",
            "mean_ret_inverse": "mean_ret_inverse_trailing",
            "win_rate_inverse": "win_rate_inverse_trailing",
            "delta_mean_ret_inverse_minus_fixed": "delta_mean_ret_inverse_trailing_minus_inverse",
            "delta_win_rate_inverse_minus_fixed": "delta_win_rate_inverse_trailing_minus_inverse",
        }
    )

    if not asset_cmp_fixed_trailing.empty:
        uplift["pct_assets_fixed_trailing_better_mean_return"] = float(
            (asset_cmp_fixed_trailing["delta_mean_ret_fixed_trailing_minus_fixed"] > 0).mean()
        )
    else:
        uplift["pct_assets_fixed_trailing_better_mean_return"] = float("nan")
    if not asset_cmp_inverse_trailing.empty:
        uplift["pct_assets_inverse_trailing_better_mean_return"] = float(
            (asset_cmp_inverse_trailing["delta_mean_ret_inverse_trailing_minus_inverse"] > 0).mean()
        )
    else:
        uplift["pct_assets_inverse_trailing_better_mean_return"] = float("nan")

    summary = {
        "universe_size_requested": len(UNIVERSE_50),
        "universe_size_fetched": int(bars["asset"].nunique()),
        "start_timestamp": str(panel["timestamp"].min()),
        "end_timestamp": str(panel["timestamp"].max()),
        "split_timestamp": str(split_ts),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "entry_rule": "long_cross_above_1m_pegged_high",
        "fixed_exit_rule": f"exit_after_{cfg.fixed_horizon_days}_trading_days",
        "inverse_exit_rule": "exit_on_cross_below_1m_pegged_low_or_end_of_sample",
        "trailing_stop_pct": cfg.trailing_stop_pct,
        "uplift": uplift,
    }

    for strategy_name, trades_df in strategy_trades.items():
        slug = _slugify(strategy_name)
        trades_df.to_csv(reports_dir / f"trades_{slug}.csv", index=False)
    comparison.to_csv(reports_dir / "strategy_comparison.csv", index=False)
    asset_cmp.to_csv(reports_dir / "asset_level_comparison.csv", index=False)
    asset_cmp_fixed_trailing.to_csv(reports_dir / "asset_level_fixed_vs_fixed_trailing.csv", index=False)
    asset_cmp_inverse_trailing.to_csv(reports_dir / "asset_level_inverse_vs_inverse_trailing.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("universe fetched:", summary["universe_size_fetched"])
    print("test rows:", summary["test_rows"], "split timestamp:", summary["split_timestamp"])
    print(comparison.to_string(index=False))
    print("\nuplift:", uplift)


if __name__ == "__main__":
    main()
