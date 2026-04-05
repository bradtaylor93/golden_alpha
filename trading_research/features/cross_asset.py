"""Cross-asset contextual features."""

from __future__ import annotations

import pandas as pd


class CrossAssetFeatureFamily:
    name = "cross_asset"
    version = "1"

    def build(self, bars: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
        ts_col = "timestamp"
        df = bars.sort_values([ts_col, "asset"]).copy()
        ret = df.groupby("asset", sort=False)["close"].pct_change().fillna(0.0)
        market = ret.groupby(df[ts_col]).mean().rename("market_ret_1")
        out = df[[ts_col, "asset"]].merge(market.reset_index(), on=ts_col, how="left")
        out["asset_minus_market_ret_1"] = ret.to_numpy() - out["market_ret_1"].fillna(0.0).to_numpy()
        return out.fillna(0.0)
