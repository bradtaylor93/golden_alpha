from __future__ import annotations

import pandas as pd

from vol_shape_regimes.config import DataConfig, FeatureConfig, VolatilityConfig
from vol_shape_regimes.data import load_price_data
from vol_shape_regimes.features import build_vol_series, build_window_features


def test_feature_build_has_expected_columns() -> None:
    pdata = load_price_data(DataConfig(source="synthetic", synthetic_rows=500, random_state=7))
    vol = build_vol_series(pdata.frame, method="ewma_std", cfg=VolatilityConfig(method="ewma_std"))
    feats = build_window_features(
        vol,
        FeatureConfig(lookback_window=40, modes=("summary", "quantile", "temporal")),
        dates=pdata.frame["date"],
    )
    assert not feats.empty
    assert "date" in feats.columns
    assert "vol_t" in feats.columns
    assert any(c.startswith("sum_") for c in feats.columns)
    assert any(c.startswith("qf_") for c in feats.columns)
    assert any(c.startswith("dct_") for c in feats.columns)
    assert pd.to_datetime(feats["date"]).is_monotonic_increasing
