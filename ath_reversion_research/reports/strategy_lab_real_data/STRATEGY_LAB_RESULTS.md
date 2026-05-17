# Strategy Lab Real-Data Results

Exploratory tests for sector rotation/relative strength, breakout, momentum, VWAP reclaim, and bear-market short-weak variants.

## Data

- Daily rows: 200,826; daily symbols: 61.
- Hourly rows: 192,331; hourly symbols: 38.
- Daily sample starts in 2010. Hourly Yahoo data is limited to the recent ~729 days.
- Transaction cost assumption: 10 bps on absolute turnover.

## Top overall variants

| strategy | universe | grain | annual_return | annual_std | sharpe | max_drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| large_caps_relative_strength_126d | large_caps | 1d | 25.48% | 19.64% | 1.30 | -34.17% |
| large_caps_momentum_12_1 | large_caps | 1d | 19.80% | 19.54% | 1.01 | -36.81% |
| hourly_large_caps_momentum_2h_24bar | hourly_large_caps | 2h | 4.74% | 18.52% | 0.26 | -37.94% |
| hourly_lower_caps_momentum_4h_24bar | hourly_lower_caps | 4h | 7.12% | 32.81% | 0.22 | -49.27% |
| sector_rotation_top3_63d | sector_etfs | 1d | 2.38% | 14.15% | 0.17 | -40.91% |
| lower_caps_relative_strength_126d | lower_caps | 1d | 6.12% | 54.24% | 0.11 | -95.37% |
| sector_rotation_regime_defensive | sector_etfs | 1d | 1.21% | 12.30% | 0.10 | -44.64% |
| hourly_large_caps_momentum_4h_24bar | hourly_large_caps | 4h | -0.03% | 19.94% | -0.00 | -40.83% |
| lower_caps_momentum_12_1 | lower_caps | 1d | -14.29% | 50.53% | -0.28 | -98.55% |
| lower_caps_vwap_reclaim_20d | lower_caps | 1d | -10.37% | 34.93% | -0.30 | -96.78% |
| lower_caps_breakout_55d_trend | lower_caps | 1d | -17.30% | 46.17% | -0.37 | -98.40% |
| large_caps_vwap_reclaim_20d | large_caps | 1d | -6.96% | 16.42% | -0.42 | -84.20% |

## Best bear-regime variants

| strategy | universe | grain | regime | annual_return | annual_std | sharpe | max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hourly_lower_caps_bear_short_weak_4h_24bar | hourly_lower_caps | 4h | bear_normal_vol | 352.40% | 30.77% | 11.45 | -7.57% |
| hourly_large_caps_bear_short_weak_4h_24bar | hourly_large_caps | 4h | bear_normal_vol | 121.40% | 20.00% | 6.07 | -3.73% |
| hourly_sector_bear_short_weak_2h_24bar | hourly_sector | 2h | bear_normal_vol | 63.44% | 13.25% | 4.79 | -2.94% |
| hourly_large_caps_bear_short_weak_1h_24bar | hourly_large_caps | 1h | bear_normal_vol | 100.21% | 22.50% | 4.45 | -4.59% |
| hourly_large_caps_bear_short_weak_2h_24bar | hourly_large_caps | 2h | bear_normal_vol | 69.47% | 19.33% | 3.59 | -3.86% |
| hourly_sector_bear_short_weak_4h_24bar | hourly_sector | 4h | bear_normal_vol | 50.07% | 13.96% | 3.59 | -2.47% |
| hourly_lower_caps_bear_short_weak_1h_24bar | hourly_lower_caps | 1h | bear_normal_vol | 77.67% | 28.51% | 2.72 | -8.70% |
| hourly_lower_caps_bear_short_weak_2h_24bar | hourly_lower_caps | 2h | bear_normal_vol | 68.93% | 29.23% | 2.36 | -9.02% |
| large_caps_bear_short_weak_63d | large_caps | 1d | bear_normal_vol | 46.38% | 26.69% | 1.74 | -9.77% |
| hourly_sector_bear_short_weak_1h_24bar | hourly_sector | 1h | bear_normal_vol | 12.57% | 14.55% | 0.86 | -7.22% |
| lower_caps_vwap_reclaim_20d | lower_caps | 1d | bear_high_vol | 26.12% | 40.84% | 0.64 | -36.03% |
| hourly_lower_caps_bear_short_weak_2h_24bar | hourly_lower_caps | 2h | bear_high_vol | 9.13% | 40.22% | 0.23 | -17.80% |

## Interpretation

- Sector rotation and relative-strength variants are intended to be risk-on engines.
- Bear-market candidates are primarily the explicit short-weak momentum variants and defensive sector rotation.
- Hourly variants are intentionally short-history exploratory signals; treat them as hypothesis generation.
- Full CSV outputs: `summary.csv`, `regime_summary.csv`, `bear_market_rankings.csv`, and `strategy_returns.csv`.