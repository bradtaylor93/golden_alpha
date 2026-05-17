"""Broker-neutral live signal generation for the validated state-overlay stack.

The module does not place trades.  It produces target weights, risk checks, and
optional order previews so execution can be reviewed before routing through a
broker.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import download_yahoo_ohlcv
from .strategies import ATHDipRecoveryConfig, ATHDipRecoveryStrategy
from .universes import symbols_for


DEFAULT_ASSET_ETFS = ("SPY", "TLT", "GLD", "XLP", "XLU", "XLV", "XLE", "XLF", "XLK", "XLI", "XLY")
DEFAULT_CONFIG = {
    "start": "2010-01-01",
    "portfolio_value": 100000.0,
    "max_gross_exposure": 1.75,
    "max_single_name_weight": 0.08,
    "min_trade_value": 100.0,
    "stock_top_n": 30,
    "asset_top_n": 1,
    "target_vol": 0.40,
    "max_leverage": 2.0,
    "state_overlay": {
        "breadth_low_threshold": 0.3770491803278688,
        "breadth_high_threshold": 0.5081967213114754,
        "spy_vol_high_threshold": 0.15456650250069123,
        "low_breadth_scale": 0.70,
        "high_breadth_scale": 1.45,
        "high_vol_scale": 0.70,
    },
    "sleeve_weights": {
        "stock_rs_126": 0.55,
        "stock_mom_12_1": 0.25,
        "ath_dip_recovery": 0.10,
        "asset_abs_mom": 0.10,
    },
}


@dataclass(frozen=True)
class LiveRiskReport:
    as_of: str
    gross_exposure: float
    net_exposure: float
    max_abs_weight: float
    active_positions: int
    state_scale: float
    vol_scale: float
    final_scale: float
    breadth_20d: float
    spy_vol_21d: float
    warnings: list[str]


def build_live_targets(config: dict[str, Any]) -> tuple[pd.DataFrame, LiveRiskReport]:
    """Download data and build next-session target weights."""

    stock_symbols = _load_stock_symbols(config)
    asset_symbols = list(config.get("asset_symbols", DEFAULT_ASSET_ETFS))
    symbols = sorted(set(stock_symbols + asset_symbols + ["SPY"]))
    bars = download_yahoo_ohlcv(symbols, start=str(config.get("start", "2010-01-01")), chunk_size=25)
    close = bars.pivot(index="date", columns="symbol", values="close").sort_index()
    close = close.dropna(axis=1, thresh=max(50, int(len(close) * 0.25)))
    stock_symbols = [symbol for symbol in stock_symbols if symbol in close.columns]
    asset_symbols = [symbol for symbol in asset_symbols if symbol in close.columns]

    stock_weights_126 = _strength_weights(close[stock_symbols], lookback=126, top_n=int(config["stock_top_n"]))
    stock_weights_12_1 = _strength_weights(
        close[stock_symbols],
        lookback=252,
        top_n=int(config["stock_top_n"]),
        skip=21,
    )
    ath_weights = _ath_weights(bars[bars["symbol"].isin(stock_symbols)], top_n=int(config["stock_top_n"]))
    asset_weights = _asset_absolute_momentum(close[asset_symbols], top_n=int(config["asset_top_n"]))

    sleeve_weights = config["sleeve_weights"]
    raw_weights = pd.DataFrame(index=close.index)
    for sleeve_name, frame in [
        ("stock_rs_126", stock_weights_126),
        ("stock_mom_12_1", stock_weights_12_1),
        ("ath_dip_recovery", ath_weights),
        ("asset_abs_mom", asset_weights),
    ]:
        scaled = frame.reindex(index=close.index).fillna(0.0) * float(sleeve_weights[sleeve_name])
        raw_weights = raw_weights.add(scaled, fill_value=0.0)
    raw_weights = raw_weights.fillna(0.0)
    raw_weights = raw_weights.loc[:, raw_weights.abs().sum(axis=0) > 0.0]
    raw_returns = (raw_weights.shift(1).fillna(0.0) * close.pct_change().reindex_like(raw_weights).fillna(0.0)).sum(axis=1)

    state_scale, breadth, spy_vol = _state_scale(close, stock_symbols, config["state_overlay"])
    vol_scale = _vol_scale(raw_returns, float(config["target_vol"]), float(config["max_leverage"]))
    final_scale = state_scale * vol_scale
    latest = raw_weights.iloc[-1] * final_scale
    latest = _apply_risk_caps(
        latest,
        max_gross=float(config["max_gross_exposure"]),
        max_single=float(config["max_single_name_weight"]),
    )

    as_of = str(raw_weights.index[-1].date())
    targets = (
        latest[latest.abs() > 1e-8]
        .sort_values(key=lambda series: series.abs(), ascending=False)
        .rename("target_weight")
        .reset_index()
        .rename(columns={"index": "symbol"})
    )
    targets["as_of"] = as_of
    latest_prices = close.ffill().iloc[-1]
    targets["latest_price"] = targets["symbol"].map(latest_prices)
    targets["target_value"] = targets["target_weight"] * float(config["portfolio_value"])

    warnings = _risk_warnings(targets, config)
    report = LiveRiskReport(
        as_of=as_of,
        gross_exposure=float(targets["target_weight"].abs().sum()),
        net_exposure=float(targets["target_weight"].sum()),
        max_abs_weight=float(targets["target_weight"].abs().max()) if not targets.empty else 0.0,
        active_positions=int(len(targets)),
        state_scale=float(state_scale),
        vol_scale=float(vol_scale),
        final_scale=float(final_scale),
        breadth_20d=float(breadth),
        spy_vol_21d=float(spy_vol),
        warnings=warnings,
    )
    return targets, report


def preview_orders(
    targets: pd.DataFrame,
    positions: pd.DataFrame,
    portfolio_value: float,
    min_trade_value: float,
) -> pd.DataFrame:
    """Convert current positions into an order preview against target weights."""

    required = {"symbol", "quantity", "price"}
    missing = required - set(positions.columns)
    if missing:
        raise ValueError(f"positions file missing columns: {sorted(missing)}")

    current = positions.copy()
    current["current_value"] = current["quantity"].astype(float) * current["price"].astype(float)
    current["current_weight"] = current["current_value"] / float(portfolio_value)
    merged = current.loc[:, ["symbol", "quantity", "price", "current_weight", "current_value"]].merge(
        targets.loc[:, ["symbol", "target_weight", "target_value", "latest_price"]],
        on="symbol",
        how="outer",
    )
    merged[["quantity", "current_weight", "current_value", "target_weight", "target_value", "latest_price"]] = merged[
        ["quantity", "current_weight", "current_value", "target_weight", "target_value", "latest_price"]
    ].fillna(0.0)
    merged["price"] = merged["price"].replace(0.0, np.nan).fillna(merged["latest_price"].replace(0.0, np.nan))
    merged["delta_value"] = merged["target_value"] - merged["current_value"]
    merged["order_shares_estimate"] = np.where(
        merged["price"].notna(),
        merged["delta_value"] / merged["price"],
        np.nan,
    )
    merged["action"] = np.where(merged["delta_value"] > 0.0, "BUY", "SELL")
    merged.loc[merged["delta_value"].abs() < min_trade_value, "action"] = "HOLD"
    return merged.sort_values("delta_value", key=lambda series: series.abs(), ascending=False).reset_index(drop=True)


def _load_stock_symbols(config: dict[str, Any]) -> list[str]:
    symbols_file = config.get("stock_symbols_file")
    if symbols_file and Path(symbols_file).exists():
        frame = pd.read_csv(symbols_file)
        if "downloaded_symbols" in frame.columns:
            return frame["downloaded_symbols"].dropna().astype(str).str.upper().tolist()
        if "symbol" in frame.columns:
            return frame["symbol"].dropna().astype(str).str.upper().tolist()
    return symbols_for(["broad_stock_sample"])


def _strength_weights(close: pd.DataFrame, lookback: int, top_n: int, skip: int = 0) -> pd.DataFrame:
    momentum = close.shift(skip) / close.shift(skip + lookback) - 1.0
    vol = close.pct_change().rolling(63, min_periods=30).std().replace(0.0, np.nan)
    score = (momentum / vol).where(momentum > 0.0)
    return _signal_inverse_vol_weights(score, close, top_n=top_n)


def _ath_weights(bars: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    positions = ATHDipRecoveryStrategy(
        ATHDipRecoveryConfig(ath_lookback=756, dip_pct=0.08, recovery_trigger_pct=0.025)
    ).generate_positions(bars)
    pivot = positions.pivot(index="date", columns="symbol", values="position").fillna(0.0).sort_index()
    counts = (pivot > 0.0).sum(axis=1).replace(0, np.nan)
    weights = pivot.div(counts, axis=0).fillna(0.0)
    if top_n > 0:
        weights = weights.where(weights.rank(axis=1, ascending=False, method="first") <= top_n, 0.0)
        weights = weights.div(weights.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    return weights


def _asset_absolute_momentum(close: pd.DataFrame, top_n: int) -> pd.DataFrame:
    score = close / close.shift(252) - 1.0
    return _signal_inverse_vol_weights(score.where(score > 0.0), close, top_n=top_n)


def _signal_inverse_vol_weights(score: pd.DataFrame, close: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = score.rank(axis=1, ascending=False, method="first") <= top_n
    vol = close.pct_change().rolling(63, min_periods=30).std().replace(0.0, np.nan)
    raw = (score.clip(lower=0.0) / vol).where(selected, 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    return raw.div(gross, axis=0).fillna(0.0)


def _state_scale(close: pd.DataFrame, stock_symbols: list[str], state_config: dict[str, float]) -> tuple[float, float, float]:
    stock_close = close[stock_symbols]
    breadth = float(((stock_close / stock_close.shift(20) - 1.0) > 0.0).mean(axis=1).iloc[-1])
    spy_returns = close["SPY"].pct_change()
    spy_vol = float(spy_returns.rolling(21, min_periods=10).std().iloc[-1] * np.sqrt(252))
    scale = 1.0
    if breadth < state_config["breadth_low_threshold"]:
        scale *= state_config["low_breadth_scale"]
    if breadth > state_config["breadth_high_threshold"]:
        scale *= state_config["high_breadth_scale"]
    if spy_vol > state_config["spy_vol_high_threshold"]:
        scale *= state_config["high_vol_scale"]
    return scale, breadth, spy_vol


def _vol_scale(raw_returns: pd.Series, target_vol: float, max_leverage: float) -> float:
    realized = float(raw_returns.rolling(63, min_periods=20).std().iloc[-1] * np.sqrt(252))
    if not np.isfinite(realized) or realized <= 0.0:
        return 0.75
    return float(np.clip(target_vol / realized, 0.20, max_leverage))


def _apply_risk_caps(weights: pd.Series, max_gross: float, max_single: float) -> pd.Series:
    capped = weights.clip(lower=-max_single, upper=max_single)
    gross = capped.abs().sum()
    if gross > max_gross and gross > 0.0:
        capped = capped * (max_gross / gross)
    return capped


def _risk_warnings(targets: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    gross = float(targets["target_weight"].abs().sum()) if not targets.empty else 0.0
    if gross > float(config["max_gross_exposure"]) + 1e-6:
        warnings.append("gross exposure exceeds configured maximum")
    if not targets.empty and targets["target_weight"].abs().max() > float(config["max_single_name_weight"]) + 1e-6:
        warnings.append("single-name weight exceeds configured maximum")
    if len(targets) == 0:
        warnings.append("no active targets generated")
    return warnings


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        user_config = json.loads(Path(path).read_text(encoding="utf-8"))
        config = _deep_update(config, user_config)
    return config


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _write_outputs(targets: pd.DataFrame, report: LiveRiskReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_dir / "target_weights.csv", index=False)
    (output_dir / "risk_report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate broker-neutral live targets and order previews.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate-targets")
    generate.add_argument("--config", default=None)
    generate.add_argument("--output", default="ath_reversion_research/reports/live/latest")

    orders = subparsers.add_parser("preview-orders")
    orders.add_argument("--targets", required=True)
    orders.add_argument("--positions", required=True)
    orders.add_argument("--portfolio-value", type=float, required=True)
    orders.add_argument("--min-trade-value", type=float, default=100.0)
    orders.add_argument("--output", default="ath_reversion_research/reports/live/latest/orders_preview.csv")
    args = parser.parse_args(argv)

    if args.command == "generate-targets":
        config = load_config(args.config)
        targets, report = build_live_targets(config)
        _write_outputs(targets, report, Path(args.output))
        print(json.dumps(asdict(report), indent=2))
        return 0

    targets = pd.read_csv(args.targets)
    positions = pd.read_csv(args.positions)
    preview = preview_orders(targets, positions, args.portfolio_value, args.min_trade_value)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.to_csv(output, index=False)
    print(f"Wrote order preview to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
