"""Cross-event study for pegged local highs/lows over 1/3/6 months.

Signal definitions (daily bars, 50 equities, ~3 years):
- Long signal when close crosses ABOVE pegged local high.
- Short signal when close crosses BELOW pegged local low.

Peg windows:
- 1 month  ~= 21 trading days
- 3 months ~= 63 trading days
- 6 months ~= 126 trading days

We evaluate forward returns after each cross event and report:
- mean return, win rate, t-stat
- long-short spread
- event counts and yearly event frequency
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
class CrossConfig:
    test_ratio: float = 0.30
    months_to_days_1: int = 21
    months_to_days_3: int = 63
    months_to_days_6: int = 126
    forward_horizons: tuple[int, ...] = (21, 63, 126)


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


def _build_panel(bars: pd.DataFrame, cfg: CrossConfig) -> pd.DataFrame:
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "asset", "close"]).sort_values(["asset", "timestamp"])
    frame["asset"] = frame["asset"].astype(str)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["high"] = pd.to_numeric(frame["high"], errors="coerce")
    frame["low"] = pd.to_numeric(frame["low"], errors="coerce")
    frame = frame.dropna(subset=["close", "high", "low"])
    frame = frame[frame["close"] > 0].copy()

    lookbacks = {
        "1m": cfg.months_to_days_1,
        "3m": cfg.months_to_days_3,
        "6m": cfg.months_to_days_6,
    }

    g = frame.groupby("asset", sort=False)
    for label, lb in lookbacks.items():
        # Pegged levels from prior history only (shift by 1 day).
        frame[f"pegged_high_{label}"] = (
            g["high"].rolling(lb, min_periods=lb).max().shift(1).reset_index(level=0, drop=True)
        )
        frame[f"pegged_low_{label}"] = (
            g["low"].rolling(lb, min_periods=lb).min().shift(1).reset_index(level=0, drop=True)
        )

        prev_close = g["close"].shift(1)
        high_level = frame[f"pegged_high_{label}"]
        low_level = frame[f"pegged_low_{label}"]

        # Cross above high = long trigger.
        frame[f"long_cross_above_high_{label}"] = (
            (prev_close <= high_level) & (frame["close"] > high_level)
        ).astype(int)
        # Cross below low = short trigger.
        frame[f"short_cross_below_low_{label}"] = (
            (prev_close >= low_level) & (frame["close"] < low_level)
        ).astype(int)

    for h in cfg.forward_horizons:
        frame[f"fwd_ret_{h}d"] = g["close"].shift(-h) / frame["close"] - 1.0

    keep = ["timestamp", "asset", "close"]
    for label in lookbacks:
        keep.extend(
            [
                f"pegged_high_{label}",
                f"pegged_low_{label}",
                f"long_cross_above_high_{label}",
                f"short_cross_below_low_{label}",
            ]
        )
    for h in cfg.forward_horizons:
        keep.append(f"fwd_ret_{h}d")
    out = frame[keep].dropna().reset_index(drop=True)
    return out


def _event_metrics(
    test: pd.DataFrame,
    side_col: str,
    fwd_col: str,
    side: str,
) -> dict[str, float | str]:
    ev = test[test[side_col] == 1].copy()
    if ev.empty:
        return {
            "side": side,
            "n_events": 0.0,
            "mean_event_return": float("nan"),
            "win_rate": float("nan"),
            "t_stat": float("nan"),
            "events_per_year": 0.0,
        }

    if side == "long":
        pnl = pd.to_numeric(ev[fwd_col], errors="coerce")
    else:
        pnl = -pd.to_numeric(ev[fwd_col], errors="coerce")
    pnl = pnl.dropna()

    span_days = (
        (ev["timestamp"].max() - ev["timestamp"].min()).total_seconds() / 86400.0 if len(ev) > 1 else np.nan
    )
    events_per_year = float(len(ev) / (span_days / 365.25)) if span_days and span_days > 0 else float("nan")

    return {
        "side": side,
        "n_events": float(len(ev)),
        "mean_event_return": float(pnl.mean()) if not pnl.empty else float("nan"),
        "win_rate": float((pnl > 0).mean()) if not pnl.empty else float("nan"),
        "t_stat": _t_stat(pnl),
        "events_per_year": events_per_year,
    }


def _evaluate(test: pd.DataFrame, cfg: CrossConfig) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for label in ("1m", "3m", "6m"):
        long_col = f"long_cross_above_high_{label}"
        short_col = f"short_cross_below_low_{label}"
        for horizon in cfg.forward_horizons:
            fwd_col = f"fwd_ret_{horizon}d"
            # Unconditional baseline return over the same horizon (all test rows).
            baseline_ret = pd.to_numeric(test[fwd_col], errors="coerce")
            baseline_long_mean = float(baseline_ret.mean()) if not baseline_ret.empty else float("nan")
            baseline_short_mean = -baseline_long_mean if not pd.isna(baseline_long_mean) else float("nan")
            long_m = _event_metrics(test, long_col, fwd_col, side="long")
            short_m = _event_metrics(test, short_col, fwd_col, side="short")

            long_mean = long_m["mean_event_return"]
            short_mean = short_m["mean_event_return"]
            long_short_spread = (
                float(long_mean - short_mean)
                if not (pd.isna(long_mean) or pd.isna(short_mean))
                else float("nan")
            )

            rows.extend(
                [
                    {
                        "peg_window": label,
                        "horizon_days": float(horizon),
                        "signal": "long_cross_above_high",
                        "baseline_mean_return": baseline_long_mean,
                        "alpha_vs_baseline": float(long_mean - baseline_long_mean)
                        if not (pd.isna(long_mean) or pd.isna(baseline_long_mean))
                        else float("nan"),
                        **long_m,
                    },
                    {
                        "peg_window": label,
                        "horizon_days": float(horizon),
                        "signal": "short_cross_below_low",
                        "baseline_mean_return": baseline_short_mean,
                        "alpha_vs_baseline": float(short_mean - baseline_short_mean)
                        if not (pd.isna(short_mean) or pd.isna(baseline_short_mean))
                        else float("nan"),
                        **short_m,
                    },
                    {
                        "peg_window": label,
                        "horizon_days": float(horizon),
                        "signal": "long_short_spread",
                        "side": "ls",
                        "baseline_mean_return": float(0.0),
                        "alpha_vs_baseline": long_short_spread,
                        "n_events": float(min(long_m["n_events"], short_m["n_events"])),
                        "mean_event_return": long_short_spread,
                        "win_rate": float("nan"),
                        "t_stat": float("nan"),
                        "events_per_year": float("nan"),
                    },
                ]
            )
    return pd.DataFrame(rows)


def main() -> None:
    out_root = Path("trading_research/examples/_output/17_pegged_high_low_cross_study")
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = CrossConfig()
    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(UNIVERSE_50, period="3y", interval="1d")
    if bars.empty:
        raise ValueError("No Yahoo bars returned for requested universe.")

    panel = _build_panel(bars, cfg)
    train, test, split_ts = _split_frame(panel, cfg.test_ratio)
    perf = _evaluate(test, cfg)

    viability = perf[perf["signal"].isin(["long_cross_above_high", "short_cross_below_low"])].copy()
    viability = viability.sort_values("t_stat", ascending=False)

    summary = {
        "universe_size_requested": len(UNIVERSE_50),
        "universe_size_fetched": int(bars["asset"].nunique()),
        "start_timestamp": str(panel["timestamp"].min()),
        "end_timestamp": str(panel["timestamp"].max()),
        "split_timestamp": str(split_ts),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "best_signal_by_tstat": (
            {
                "peg_window": str(viability.iloc[0]["peg_window"]),
                "horizon_days": float(viability.iloc[0]["horizon_days"]),
                "signal": str(viability.iloc[0]["signal"]),
                "t_stat": float(viability.iloc[0]["t_stat"]),
                "mean_event_return": float(viability.iloc[0]["mean_event_return"]),
                "win_rate": float(viability.iloc[0]["win_rate"]),
                "n_events": float(viability.iloc[0]["n_events"]),
            }
            if not viability.empty
            else None
        ),
    }

    panel.to_csv(reports_dir / "panel_features.csv", index=False)
    perf.to_csv(reports_dir / "cross_event_performance.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("reports:", reports_dir.resolve())
    print("universe fetched:", summary["universe_size_fetched"])
    print("test rows:", summary["test_rows"], "split timestamp:", summary["split_timestamp"])
    print(perf.to_string(index=False))
    if summary["best_signal_by_tstat"] is not None:
        print("\nbest signal by t-stat:", summary["best_signal_by_tstat"])


if __name__ == "__main__":
    main()
