"""Volatility estimators and rolling shape features."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.stats import kurtosis, skew

from vol_shape_regimes.config import FeatureConfig, VolatilityConfig


def compute_returns(close: pd.Series, return_type: str = "log") -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    if return_type == "log":
        out = np.log(close / close.shift(1))
    elif return_type == "simple":
        out = close.pct_change()
    else:
        raise ValueError(f"Unsupported return type: {return_type}")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_vol_series(df: pd.DataFrame, method: str, cfg: VolatilityConfig) -> pd.Series:
    """Build realized-vol proxy series indexed to df rows."""
    rets = compute_returns(df["close"], return_type=cfg.return_type)
    m = method.lower()
    if m == "abs_return":
        vol = rets.abs()
    elif m == "squared_return":
        vol = rets.pow(2)
    elif m == "rolling_std":
        vol = rets.rolling(cfg.rolling_window, min_periods=max(5, cfg.rolling_window // 4)).std()
    elif m == "ewma_std":
        vol = rets.ewm(span=cfg.ewma_span, adjust=False).std()
    elif m == "parkinson":
        if not {"high", "low"}.issubset(df.columns):
            raise ValueError("Parkinson volatility requires high and low columns.")
        hl = np.log(pd.to_numeric(df["high"], errors="coerce") / pd.to_numeric(df["low"], errors="coerce"))
        vol = np.sqrt((hl.pow(2)) / (4.0 * np.log(2.0)))
    elif m == "garman_klass":
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            raise ValueError("Garman-Klass volatility requires OHLC columns.")
        log_hl = np.log(pd.to_numeric(df["high"], errors="coerce") / pd.to_numeric(df["low"], errors="coerce"))
        log_co = np.log(pd.to_numeric(df["close"], errors="coerce") / pd.to_numeric(df["open"], errors="coerce"))
        gk_var = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
        vol = np.sqrt(gk_var.clip(lower=0.0))
    else:
        raise ValueError(f"Unsupported volatility method: {method}")

    vol = pd.to_numeric(vol, errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    vol.name = "vol"
    return vol


def _acf(window: np.ndarray, lag: int) -> float:
    if lag >= len(window):
        return 0.0
    x = window - np.mean(window)
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    num = float(np.dot(x[lag:], x[:-lag]))
    return num / denom


def _pacf_proxy(window: np.ndarray, lag: int) -> float:
    if lag < 1 or len(window) <= lag + 2:
        return 0.0
    y = window[lag:]
    cols = [window[lag - i - 1 : len(window) - i - 1] for i in range(lag)]
    x = np.column_stack(cols)
    x = np.c_[np.ones(len(x)), x]
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    # coefficient for lag-k variable (last regressor)
    return float(beta[-1])


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0
    x = np.clip(x, 0.0, None)
    if float(np.sum(x)) <= 0.0:
        return 0.0
    x_sorted = np.sort(x)
    n = len(x_sorted)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x_sorted) / (n * np.sum(x_sorted))) - (n + 1) / n)


def _longest_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for flag in mask:
        if flag:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _local_peak_count(x: np.ndarray) -> int:
    if len(x) < 3:
        return 0
    return int(np.sum((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])))


def _hurst_proxy(window: np.ndarray) -> float:
    # Lightweight persistence proxy based on variance growth over scales.
    n = len(window)
    if n < 16:
        return 0.5
    scales = np.array([2, 4, 8], dtype=int)
    scales = scales[scales < n // 2]
    if len(scales) < 2:
        return 0.5
    vars_: list[float] = []
    for s in scales:
        agg = pd.Series(window).rolling(s, min_periods=s).mean().dropna().to_numpy()
        vars_.append(float(np.var(agg)))
    x = np.log(scales.astype(float))
    y = np.log(np.maximum(np.array(vars_), 1e-12))
    slope = np.polyfit(x, y, 1)[0]
    return float(np.clip(0.5 + slope / 2.0, 0.0, 1.0))


def _summary_features(window: np.ndarray, cfg: FeatureConfig) -> dict[str, float]:
    q10, q25, q50, q75, q90 = np.quantile(window, [0.1, 0.25, 0.5, 0.75, 0.9])
    mean_v = float(np.mean(window))
    std_v = float(np.std(window))
    iqr = float(q75 - q25)
    top_q = float(np.quantile(window, cfg.spike_quantile))
    run_q = float(np.quantile(window, cfg.run_quantile))
    sorted_v = np.sort(window)
    probs = sorted_v / max(float(np.sum(sorted_v)), 1e-12)
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    calm_mask = window <= run_q
    high_mask = window > run_q
    t = np.arange(len(window), dtype=float)
    slope = float(np.polyfit(t, window, 1)[0])
    curve = float(np.polyfit(t, window, 2)[0]) if len(window) >= 8 else 0.0

    feats: dict[str, float] = {
        "sum_mean": mean_v,
        "sum_std": std_v,
        "sum_cv": float(mean_v / max(std_v, 1e-12)),
        "sum_min": float(np.min(window)),
        "sum_max": float(np.max(window)),
        "sum_median": float(q50),
        "sum_q10": float(q10),
        "sum_q25": float(q25),
        "sum_q75": float(q75),
        "sum_q90": float(q90),
        "sum_iqr": iqr,
        "sum_skew": float(skew(window, bias=False)),
        "sum_kurt": float(kurtosis(window, fisher=True, bias=False)),
        "sum_tail_ratio_q90_q50": float(q90 / max(q50, 1e-12)),
        "sum_tail_ratio_q90_q10": float(q90 / max(q10, 1e-12)),
        "sum_entropy": entropy,
        "sum_acf_1": _acf(window, 1),
        "sum_trend_slope": slope,
        "sum_trend_curvature": curve,
        "sum_spike_count": float(np.sum(window >= top_q)),
        "sum_mass_top10pct": float(np.sum(window[window >= top_q]) / max(np.sum(window), 1e-12)),
        "sum_longest_calm_run": float(_longest_run(calm_mask)),
        "sum_longest_high_run": float(_longest_run(high_mask)),
        "sum_local_peaks": float(_local_peak_count(window)),
        "sum_gini": _gini(window),
        "sum_hurst_proxy": _hurst_proxy(window),
        "sum_jaggedness": float(np.mean(np.abs(np.diff(window)))),
    }
    for lag in range(1, max(1, cfg.pacf_lags) + 1):
        feats[f"sum_pacf_{lag}"] = _pacf_proxy(window, lag)
    return feats


def _quantile_embedding(window: np.ndarray, points: int) -> dict[str, float]:
    qs = np.linspace(0.0, 1.0, points)
    vals = np.quantile(window, qs)
    return {f"qf_{i:02d}": float(v) for i, v in enumerate(vals)}


def _hist_embedding(window: np.ndarray, bins: int) -> dict[str, float]:
    lo = float(np.quantile(window, 0.01))
    hi = float(np.quantile(window, 0.99))
    if hi <= lo:
        hi = lo + 1e-8
    hist, _ = np.histogram(window, bins=bins, range=(lo, hi), density=True)
    hist = hist / max(float(np.sum(hist)), 1e-12)
    return {f"hist_{i:02d}": float(v) for i, v in enumerate(hist)}


def _temporal_embedding(window: np.ndarray, cfg: FeatureConfig) -> dict[str, float]:
    x = np.asarray(window, dtype=float)
    z = (x - np.mean(x)) / max(float(np.std(x)), 1e-12)

    # Linear interpolation downsample to keep fixed-width temporal shape.
    idx_src = np.linspace(0.0, 1.0, len(z))
    idx_dst = np.linspace(0.0, 1.0, cfg.temporal_downsample)
    down = np.interp(idx_dst, idx_src, z)

    dct_n = min(cfg.temporal_dct_components, len(z))
    dct_vals = dct(z, norm="ortho")[:dct_n]

    feats: dict[str, float] = {f"tpath_{i:02d}": float(v) for i, v in enumerate(down)}
    for i, v in enumerate(dct_vals):
        feats[f"dct_{i:02d}"] = float(v)
    for lag in range(1, max(1, cfg.temporal_acf_lags) + 1):
        feats[f"tacf_{lag}"] = _acf(z, lag)
    feats["temporal_concentration_top20"] = float(
        np.sum(np.sort(np.abs(z))[-max(1, int(0.2 * len(z))) :]) / max(np.sum(np.abs(z)), 1e-12)
    )
    return feats


def _window_feature_map(window: np.ndarray, cfg: FeatureConfig) -> dict[str, float]:
    feats: dict[str, float] = {}
    modes = set(m.lower() for m in cfg.modes)
    if "summary" in modes:
        feats.update(_summary_features(window, cfg))
    if "quantile" in modes:
        feats.update(_quantile_embedding(window, cfg.quantile_points))
    if "histogram" in modes:
        feats.update(_hist_embedding(window, cfg.histogram_bins))
    if "temporal" in modes:
        feats.update(_temporal_embedding(window, cfg))
    if not feats:
        raise ValueError("No feature modes enabled. Check config.features.modes.")
    return feats


def _iter_windows(series: pd.Series, lookback: int) -> Iterable[tuple[int, np.ndarray]]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    for end_idx in range(lookback - 1, len(values)):
        start_idx = end_idx - lookback + 1
        win = values[start_idx : end_idx + 1]
        if np.isnan(win).any():
            continue
        yield end_idx, win


def build_window_features(vol_series: pd.Series, cfg: FeatureConfig, dates: pd.Series) -> pd.DataFrame:
    """Build trailing-window volatility-shape features at each valid date t."""
    rows: list[dict[str, float | str]] = []
    for end_idx, win in _iter_windows(vol_series, cfg.lookback_window):
        feat_row = _window_feature_map(win, cfg)
        feat_row["date"] = pd.to_datetime(dates.iloc[end_idx], utc=True, errors="coerce").tz_localize(None)
        feat_row["vol_t"] = float(vol_series.iloc[end_idx])
        rows.append(feat_row)
    if not rows:
        raise ValueError("No window features were created; check lookback window and data quality.")
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    feature_cols = [c for c in out.columns if c not in {"date"}]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
