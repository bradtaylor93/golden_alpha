"""Signal definitions for ATH dip recovery and exponential reversion shorts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ATHDipRecoveryConfig:
    """Parameters for buying assets that dip after reaching an all-time high."""

    ath_lookback: int = 756
    dip_pct: float = 0.08
    recovery_trigger_pct: float = 0.025
    max_hold_days: int = 60
    stop_loss_pct: float = 0.12
    take_profit_pct: float = 0.10
    position_size: float = 1.0


@dataclass(frozen=True)
class ExponentialReversionConfig:
    """Parameters for shorting stretched exponential moves after confirmation."""

    trend_lookback: int = 126
    stretch_zscore: float = 2.0
    confirmation_days: int = 3
    confirmation_drop_pct: float = 0.02
    max_hold_days: int = 30
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.08
    position_size: float = -1.0


@dataclass(frozen=True)
class MovingAverageTouchConfig:
    """Parameters for buying stocks when they touch the 200-day moving average."""

    moving_average_days: int = 200
    touch_band_pct: float = 0.005
    require_prior_above_days: int = 20
    require_bounce_confirmation: bool = False
    require_ma_rising: bool = False
    ma_slope_lookback: int = 20
    require_above_long_ma: bool = False
    max_hold_days: int = 40
    stop_loss_pct: float = 0.07
    take_profit_pct: float = 0.10
    exit_below_ma_pct: float = 0.02
    position_size: float = 1.0
    name: str = "ma200_touch"


class Strategy(Protocol):
    """Protocol implemented by causal daily-position strategies."""

    name: str

    def generate_positions(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return long-form target positions with date, symbol, position."""


class ATHDipRecoveryStrategy:
    """Buy ATH assets after a dip and exit once the dip recovers.

    The signal is intentionally conservative:
    - require the asset to have been at an ATH recently,
    - require a drawdown from that ATH,
    - enter only after the asset starts recovering from the local trough,
    - exit on recovery, stop loss, or max holding age.
    """

    name = "ath_dip_recovery"

    def __init__(self, config: ATHDipRecoveryConfig | None = None) -> None:
        self.config = config or ATHDipRecoveryConfig()

    def generate_positions(self, bars: pd.DataFrame) -> pd.DataFrame:
        pieces = [_ath_dip_positions(group, self.config) for _, group in bars.groupby("symbol")]
        return pd.concat(pieces, ignore_index=True) if pieces else _empty_positions()


class ExponentialReversionShortStrategy:
    """Short exponential-style upside dislocations only after a confirmed drop."""

    name = "exponential_reversion_short"

    def __init__(self, config: ExponentialReversionConfig | None = None) -> None:
        self.config = config or ExponentialReversionConfig()

    def generate_positions(self, bars: pd.DataFrame) -> pd.DataFrame:
        pieces = [_exponential_short_positions(group, self.config) for _, group in bars.groupby("symbol")]
        return pd.concat(pieces, ignore_index=True) if pieces else _empty_positions()


class MovingAverageTouchStrategy:
    """Buy 200MA touches, optionally requiring bounce and trend confirmation."""

    def __init__(self, config: MovingAverageTouchConfig | None = None) -> None:
        self.config = config or MovingAverageTouchConfig()
        self.name = self.config.name

    def generate_positions(self, bars: pd.DataFrame) -> pd.DataFrame:
        pieces = [_ma_touch_positions(group, self.config) for _, group in bars.groupby("symbol")]
        return pd.concat(pieces, ignore_index=True) if pieces else _empty_positions()


def _empty_positions() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "symbol", "position", "signal"])


def _ath_dip_positions(group: pd.DataFrame, config: ATHDipRecoveryConfig) -> pd.DataFrame:
    g = group.sort_values("date").reset_index(drop=True).copy()
    close = g["close"].astype(float)
    rolling_ath = close.rolling(config.ath_lookback, min_periods=max(20, config.ath_lookback // 4)).max()
    drawdown = close / rolling_ath - 1.0
    trough_since_ath = close.where(drawdown <= 0).rolling(config.max_hold_days, min_periods=1).min()

    positions: list[float] = []
    signals: list[str] = []
    in_trade = False
    entry_price = 0.0
    entry_idx = -1

    for idx, price in enumerate(close):
        if not np.isfinite(price) or not np.isfinite(rolling_ath.iloc[idx]):
            positions.append(0.0)
            signals.append("insufficient_history")
            continue

        if in_trade:
            pnl = price / entry_price - 1.0
            holding_days = idx - entry_idx
            recovered = drawdown.iloc[idx] >= -config.recovery_trigger_pct
            stopped = pnl <= -config.stop_loss_pct
            target_hit = pnl >= config.take_profit_pct
            timed_out = holding_days >= config.max_hold_days
            if recovered or stopped or target_hit or timed_out:
                in_trade = False
                positions.append(0.0)
                signals.append("exit")
            else:
                positions.append(config.position_size)
                signals.append("hold")
            continue

        dipped = drawdown.iloc[idx] <= -config.dip_pct
        trough = trough_since_ath.iloc[idx]
        recovering = np.isfinite(trough) and price >= trough * (1.0 + config.recovery_trigger_pct)
        if dipped and recovering:
            in_trade = True
            entry_price = float(price)
            entry_idx = idx
            positions.append(config.position_size)
            signals.append("enter")
        else:
            positions.append(0.0)
            signals.append("flat")

    return pd.DataFrame(
        {
            "date": g["date"],
            "symbol": g["symbol"],
            "position": positions,
            "signal": signals,
        }
    )


def _ma_touch_positions(group: pd.DataFrame, config: MovingAverageTouchConfig) -> pd.DataFrame:
    g = group.sort_values("date").reset_index(drop=True).copy()
    close = g["close"].astype(float)
    low = g["low"].astype(float)
    high = g["high"].astype(float)
    ma = close.rolling(
        config.moving_average_days,
        min_periods=max(50, config.moving_average_days // 2),
    ).mean()
    prior_above_ratio = (
        (close.shift(1) > ma.shift(1))
        .rolling(config.require_prior_above_days, min_periods=config.require_prior_above_days)
        .mean()
    )
    ma_slope = ma / ma.shift(config.ma_slope_lookback) - 1.0

    positions: list[float] = []
    signals: list[str] = []
    in_trade = False
    entry_price = 0.0
    entry_idx = -1

    for idx, price in enumerate(close):
        current_ma = ma.iloc[idx]
        if not np.isfinite(price) or not np.isfinite(current_ma):
            positions.append(0.0)
            signals.append("insufficient_history")
            continue

        if in_trade:
            pnl = price / entry_price - 1.0
            holding_days = idx - entry_idx
            stopped = pnl <= -config.stop_loss_pct
            target_hit = pnl >= config.take_profit_pct
            lost_ma = price <= current_ma * (1.0 - config.exit_below_ma_pct)
            timed_out = holding_days >= config.max_hold_days
            if stopped or target_hit or lost_ma or timed_out:
                in_trade = False
                positions.append(0.0)
                signals.append("exit")
            else:
                positions.append(config.position_size)
                signals.append("hold")
            continue

        touched = (
            low.iloc[idx] <= current_ma * (1.0 + config.touch_band_pct)
            and high.iloc[idx] >= current_ma * (1.0 - config.touch_band_pct)
        )
        had_prior_uptrend = prior_above_ratio.iloc[idx] >= 0.8
        bounce_confirmed = price > current_ma and close.pct_change().iloc[idx] > 0.0
        ma_rising = ma_slope.iloc[idx] > 0.0
        above_long_ma = price >= current_ma

        filters_pass = bool(touched and had_prior_uptrend)
        if config.require_bounce_confirmation:
            filters_pass = filters_pass and bool(bounce_confirmed)
        if config.require_ma_rising:
            filters_pass = filters_pass and bool(ma_rising)
        if config.require_above_long_ma:
            filters_pass = filters_pass and bool(above_long_ma)

        if filters_pass:
            in_trade = True
            entry_price = float(price)
            entry_idx = idx
            positions.append(config.position_size)
            signals.append("enter")
        else:
            positions.append(0.0)
            signals.append("flat")

    return pd.DataFrame(
        {
            "date": g["date"],
            "symbol": g["symbol"],
            "position": positions,
            "signal": signals,
        }
    )


def _exponential_short_positions(group: pd.DataFrame, config: ExponentialReversionConfig) -> pd.DataFrame:
    g = group.sort_values("date").reset_index(drop=True).copy()
    close = g["close"].astype(float)
    log_close = np.log(close.replace(0.0, np.nan))
    trend = _rolling_linear_log_trend(log_close, config.trend_lookback)
    residual = log_close - trend
    residual_std = residual.rolling(config.trend_lookback, min_periods=max(20, config.trend_lookback // 3)).std()
    stretch_z = residual / residual_std
    rolling_peak = close.rolling(config.confirmation_days + 1, min_periods=1).max()
    drop_from_peak = close / rolling_peak - 1.0

    positions: list[float] = []
    signals: list[str] = []
    in_trade = False
    entry_price = 0.0
    entry_idx = -1
    setup_active = False

    for idx, price in enumerate(close):
        if not np.isfinite(price) or not np.isfinite(stretch_z.iloc[idx]):
            positions.append(0.0)
            signals.append("insufficient_history")
            continue

        if in_trade:
            short_pnl = entry_price / price - 1.0
            holding_days = idx - entry_idx
            reverted_to_trend = residual.iloc[idx] <= 0.0
            stopped = short_pnl <= -config.stop_loss_pct
            target_hit = short_pnl >= config.take_profit_pct
            timed_out = holding_days >= config.max_hold_days
            if reverted_to_trend or stopped or target_hit or timed_out:
                in_trade = False
                setup_active = False
                positions.append(0.0)
                signals.append("exit")
            else:
                positions.append(config.position_size)
                signals.append("hold")
            continue

        if stretch_z.iloc[idx] >= config.stretch_zscore:
            setup_active = True

        confirmed_drop = (
            setup_active
            and drop_from_peak.iloc[idx] <= -config.confirmation_drop_pct
            and close.pct_change().iloc[idx] < 0.0
        )
        if confirmed_drop:
            in_trade = True
            entry_price = float(price)
            entry_idx = idx
            positions.append(config.position_size)
            signals.append("enter")
        else:
            positions.append(0.0)
            signals.append("flat" if not setup_active else "setup")

    return pd.DataFrame(
        {
            "date": g["date"],
            "symbol": g["symbol"],
            "position": positions,
            "signal": signals,
        }
    )


def _rolling_linear_log_trend(series: pd.Series, lookback: int) -> pd.Series:
    """Fit a causal linear trend in log price on each rolling window."""

    values = series.to_numpy(dtype=float)
    trend = np.full(len(values), np.nan)
    x = np.arange(lookback, dtype=float)
    x_mean = float(x.mean())
    x_var = float(((x - x_mean) ** 2).sum())
    min_periods = max(20, lookback // 3)

    for end in range(min_periods - 1, len(values)):
        start = max(0, end - lookback + 1)
        y = values[start : end + 1]
        valid = np.isfinite(y)
        if valid.sum() < min_periods:
            continue
        xv = np.arange(len(y), dtype=float)[valid]
        yv = y[valid]
        xv_mean = float(xv.mean())
        denom = float(((xv - xv_mean) ** 2).sum())
        if denom <= 0:
            continue
        beta = float(((xv - xv_mean) * (yv - yv.mean())).sum() / denom)
        alpha = float(yv.mean() - beta * xv_mean)
        trend[end] = alpha + beta * (len(y) - 1)

    # The precomputed x variables document the full-window formula and avoid
    # reallocating for the common case in future vectorized optimization.
    _ = (x, x_mean, x_var)
    return pd.Series(trend, index=series.index)
