# Unrelated Strategy Lab Results

Searched for daily and hourly sleeves that are less related to the current high-return large-cap momentum portfolio.
All candidates use causal one-bar-lag execution, 10 bps turnover cost, and inverse-volatility or signal-strength sizing.

## Best standalone candidates

| strategy | grain | annual_return | annual_std | sharpe | max_drawdown | correlation_to_max_return | total_return |
| --- | --- | --- | --- | --- | --- | --- | --- |
| daily_large_cap_lowvol_trend | 1d | 17.25% | 17.65% | 0.98 | -28.94% | 0.88 | 1237.66% |
| daily_asset_abs_mom_252_top1 | 1d | 12.03% | 20.16% | 0.60 | -38.00% | 0.50 | 536.51% |
| daily_asset_abs_mom_252_top3 | 1d | 7.34% | 13.11% | 0.56 | -21.74% | 0.70 | 217.13% |
| daily_asset_abs_mom_252_top2 | 1d | 6.60% | 14.64% | 0.45 | -23.72% | 0.62 | 183.37% |
| daily_asset_abs_mom_126_top2 | 1d | 5.05% | 14.66% | 0.34 | -38.99% | 0.59 | 123.17% |
| daily_asset_abs_mom_126_top1 | 1d | 0.87% | 19.68% | 0.04 | -67.61% | 0.45 | 15.12% |
| daily_large_cap_5d_reversal | 1d | -1.55% | 19.90% | -0.08 | -48.25% | 0.72 | -22.51% |
| daily_risk_off_tlt_gld | 1d | -0.87% | 6.52% | -0.13 | -30.54% | 0.08 | -13.30% |
| daily_large_cap_volume_thrust | 1d | -4.31% | 28.05% | -0.15 | -78.02% | 0.27 | -51.25% |
| daily_large_cap_rsi_reversal | 1d | -9.53% | 19.63% | -0.49 | -88.42% | 0.47 | -80.44% |
| daily_sector_market_neutral_rs | 1d | -8.77% | 10.67% | -0.82 | -80.15% | -0.05 | -77.61% |
| hourly_large_cap_vwap_fade_2h | 2h | -7.99% | 6.64% | -1.20 | -32.43% | 0.06 | -27.31% |

## Lowest-correlation candidates

| strategy | grain | annual_return | annual_std | sharpe | max_drawdown | correlation_to_max_return | total_return |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hourly_large_cap_rsi_fade_1h | 1h | -58.86% | 15.25% | -3.86 | -94.99% | 0.01 | -94.89% |
| hourly_large_cap_rsi_fade_2h | 2h | -39.72% | 14.53% | -2.73 | -86.20% | -0.02 | -85.61% |
| daily_sector_market_neutral_rs | 1d | -8.77% | 10.67% | -0.82 | -80.15% | -0.05 | -77.61% |
| hourly_large_cap_vwap_fade_2h | 2h | -7.99% | 6.64% | -1.20 | -32.43% | 0.06 | -27.31% |
| daily_risk_off_tlt_gld | 1d | -0.87% | 6.52% | -0.13 | -30.54% | 0.08 | -13.30% |
| hourly_large_cap_vwap_fade_1h | 1h | -38.74% | 10.93% | -3.54 | -81.16% | 0.11 | -80.62% |
| hourly_sector_vwap_fade_1h | 1h | -36.07% | 5.78% | -6.24 | -77.64% | 0.12 | -77.63% |
| hourly_sector_market_neutral_mom_1h | 1h | -47.35% | 7.54% | -6.28 | -88.33% | -0.13 | -88.33% |
| daily_large_cap_volume_thrust | 1d | -4.31% | 28.05% | -0.15 | -78.02% | 0.27 | -51.25% |
| hourly_large_cap_opening_range_1h | 1h | -39.93% | 23.35% | -1.71 | -84.55% | 0.40 | -81.84% |
| daily_asset_abs_mom_126_top1 | 1d | 0.87% | 19.68% | 0.04 | -67.61% | 0.45 | 15.12% |
| daily_large_cap_rsi_reversal | 1d | -9.53% | 19.63% | -0.49 | -88.42% | 0.47 | -80.44% |

## Portfolio combination tests

| portfolio | annual_return | annual_std | sharpe | max_drawdown | total_return |
| --- | --- | --- | --- | --- | --- |
| base_overlay_daily_asset_abs_mom_252_top1_25_retarget35 | 54.55% | 37.79% | 1.44 | -35.65% | 32560.08% |
| base_overlay_daily_asset_abs_mom_252_top1_50_retarget35 | 54.04% | 37.73% | 1.43 | -36.67% | 31179.30% |
| base_overlay_unrelated_basket_25_retarget35 | 53.20% | 37.80% | 1.41 | -37.43% | 28984.12% |
| base_overlay_daily_asset_abs_mom_252_top2_25_retarget35 | 53.15% | 37.75% | 1.41 | -38.05% | 28853.80% |
| base_overlay_daily_asset_abs_mom_126_top2_25_retarget35 | 52.85% | 37.81% | 1.40 | -40.49% | 28113.64% |
| base_overlay_daily_asset_abs_mom_126_top1_25_retarget35 | 51.87% | 37.84% | 1.37 | -36.69% | 25798.48% |
| base_overlay_unrelated_basket_50_retarget35 | 51.75% | 37.75% | 1.37 | -36.37% | 25528.35% |
| base_overlay_daily_asset_abs_mom_252_top2_50_retarget35 | 51.70% | 37.69% | 1.37 | -35.97% | 25417.34% |
| base_overlay_daily_asset_abs_mom_126_top2_50_retarget35 | 51.08% | 37.80% | 1.35 | -39.64% | 24056.77% |
| base_max_return_35 | 50.07% | 35.45% | 1.41 | -43.69% | 21993.86% |
| base_overlay_daily_asset_abs_mom_126_top1_50_retarget35 | 48.82% | 37.82% | 1.29 | -38.82% | 19677.50% |
| base_90_daily_asset_abs_mom_252_top1_10 | 46.70% | 33.02% | 1.41 | -41.37% | 16231.39% |

## Interpretation

- The best addition was asset-class absolute momentum with inverse-volatility sizing. It is not fully unrelated, but it is meaningfully less concentrated than the large-cap stock-only core.
- Overlaying the asset-momentum sleeve and re-targeting volatility improved return and drawdown versus simply replacing part of the core.
- Hourly VWAP/RSI fade variants remained fragile on Yahoo intraday data after costs, even with signal-strength sizing.
- Market-neutral sector momentum and risk-off ETF trend had low correlation but were too weak after costs.