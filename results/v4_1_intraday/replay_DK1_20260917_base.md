# Nurex V4.1 Intraday walk-forward replay — DK1

Book: `v41id-replay:DK1`  
Test window: 2025-07-02 00:00:00 → 2026-09-17 07:15:00 UTC  
Cost model: 2.35 EUR/MWh (stress x2.0)  
Collateral: 4,021 EUR  
Every quarter was forecast with information available at intraday gate closure (delivery start minus 60 min); models retrained every 30 days on labels known at that time.

## Trading
| metric | value |
|---|---|
| quarters | 39438 |
| trades | 6605 |
| trade_rate_pct | 16.75 |
| buy | 793 |
| sell | 5812 |
| mwh_traded | 2,745.10 |
| net_eur | 62,780.67 |
| gross_eur | 69,231.66 |
| cost_eur | 6,450.98 |
| net_eur_per_mwh | 22.87 |
| win_rate_pct | 60.05 |
| days | 413 |
| positive_days_pct | 56.90 |
| worst_day_eur | -1,603.30 |
| max_drawdown_eur | -5,045.24 |
| max_drawdown_pct_collateral | -125.46 |
| sharpe_daily_ann | 3.55 |
| net_eur_at_stress_costs | 56,329.69 |

## Forecast quality
| metric | value |
|---|---|
| mae_model | 67.66 |
| mae_zero | 59.12 |
| mae_median_q50 | 56.96 |
| dir_acc_model_pct | 41.60 |
| dir_acc_always_negative_pct | 53.33 |
| share_positive_spread_pct | 25.41 |
| brier_up_model | 0.19 |
| brier_up_climatology | 0.19 |
| pinball_q10 | 10.33 |
| coverage_q10_pct | 11.88 |
| pinball_q50 | 28.48 |
| coverage_q50_pct | 48.61 |
| pinball_q90 | 24.97 |
| coverage_q90_pct | 89.34 |

## Naive baselines (1 MWh every quarter, same costs)
| strategy                     |   trades |     net_eur |   net_eur_per_mwh |
|:-----------------------------|---------:|------------:|------------------:|
| always_sell                  |    39438 | -116,371.38 |             -2.95 |
| always_buy                   |    39438 |  -68,987.22 |             -1.75 |
| sign_of_last_published       |    39438 |   77,595.48 |              1.97 |
| sign_of_same_quarter_7d_mean |    39218 |  -13,247.29 |             -0.34 |

## Monthly
| month   |   trades |    mwh |   net_eur |   mean_abs_spread |
|:--------|---------:|-------:|----------:|------------------:|
| 2025-07 |       44 |  61.60 | 15,898.86 |             84.74 |
| 2025-08 |      581 | 199.00 |  8,644.96 |             58.64 |
| 2025-09 |     1072 | 453.40 | 20,966.42 |             87.04 |
| 2025-10 |      347 | 123.50 |    968.25 |             69.81 |
| 2025-11 |      152 |  54.90 |     59.87 |             64.50 |
| 2025-12 |      152 |  95.70 |    294.56 |             40.11 |
| 2026-01 |      474 | 278.30 | -2,898.07 |             47.20 |
| 2026-02 |      656 | 184.40 |  1,123.70 |             34.99 |
| 2026-03 |      498 | 159.10 |  2,408.78 |             49.42 |
| 2026-04 |      617 | 366.30 |  5,284.86 |             43.92 |
| 2026-05 |      423 | 200.90 |  3,362.67 |             44.72 |
| 2026-06 |      762 | 245.10 |   -690.52 |             71.68 |
| 2026-07 |      248 | 126.70 |  3,799.28 |             65.61 |
| 2026-08 |      171 |  64.60 |  1,303.72 |             55.33 |
| 2026-09 |      408 | 131.60 |  2,253.32 |             70.15 |

## Retraining periods and chosen thresholds
| period_start        | cut                 |   n_train | buy                            | sell                           |   val_net_eur |   val_trades | no_trade   |
|:--------------------|:--------------------|----------:|:-------------------------------|:-------------------------------|--------------:|-------------:|:-----------|
| 2025-07-02 00:00:00 | 2025-07-01 23:00:00 |     11469 | {'margin': 10.0, 'pmin': 0.45} |                                |      76625.3  |          681 | False      |
| 2025-08-01 00:00:00 | 2025-07-31 23:00:00 |     14349 | {'margin': 40.0, 'pmin': 0.45} | {'margin': 20.0, 'pmin': 0.65} |      35314.9  |         2129 | False      |
| 2025-08-31 00:00:00 | 2025-08-30 23:00:00 |     17229 | {'margin': 0.0, 'pmin': 0.45}  | {'margin': 5.0, 'pmin': 0.45}  |      26311.8  |         2890 | False      |
| 2025-09-30 00:00:00 | 2025-09-29 23:00:00 |     20109 | {'margin': 5.0, 'pmin': 0.45}  | {'margin': 20.0, 'pmin': 0.6}  |      24852.9  |         1163 | False      |
| 2025-10-30 00:00:00 | 2025-10-29 23:00:00 |     22989 | {'margin': 10.0, 'pmin': 0.5}  | {'margin': 0.0, 'pmin': 0.7}   |      23034.1  |         1031 | False      |
| 2025-11-29 00:00:00 | 2025-12-12 11:00:00 |     24147 | {'margin': 0.0, 'pmin': 0.45}  | {'margin': 20.0, 'pmin': 0.7}  |       6599.57 |          323 | False      |
| 2025-12-29 00:00:00 | 2025-12-28 23:00:00 |     25725 | {'margin': 5.0, 'pmin': 0.45}  | {'margin': 40.0, 'pmin': 0.65} |       2971.76 |          194 | False      |
| 2026-01-28 00:00:00 | 2026-01-27 23:00:00 |     28605 |                                | {'margin': 10.0, 'pmin': 0.5}  |       1893.18 |          577 | False      |
| 2026-02-27 00:00:00 | 2026-02-26 23:00:00 |     31485 |                                | {'margin': 10.0, 'pmin': 0.55} |       1521.49 |          725 | False      |
| 2026-03-29 00:00:00 | 2026-03-28 23:00:00 |     34365 | {'margin': 2.0, 'pmin': 0.45}  | {'margin': 0.0, 'pmin': 0.6}   |       9263.72 |         1523 | False      |
| 2026-04-28 00:00:00 | 2026-04-27 23:00:00 |     37245 | {'margin': 40.0, 'pmin': 0.45} | {'margin': 0.0, 'pmin': 0.6}   |      12729.9  |         1487 | False      |
| 2026-05-28 00:00:00 | 2026-05-27 23:00:00 |     40125 | {'margin': 5.0, 'pmin': 0.45}  | {'margin': 0.0, 'pmin': 0.5}   |      11059.5  |         1483 | False      |
| 2026-06-27 00:00:00 | 2026-06-26 23:00:00 |     43005 | {'margin': 0.0, 'pmin': 0.45}  | {'margin': 0.0, 'pmin': 0.7}   |       8527.47 |          443 | False      |
| 2026-07-27 00:00:00 | 2026-07-26 23:00:00 |     45885 |                                | {'margin': 20.0, 'pmin': 0.55} |       4735.12 |          470 | False      |
| 2026-08-26 00:00:00 | 2026-08-25 23:00:00 |     48765 |                                | {'margin': 0.0, 'pmin': 0.55}  |       7185.2  |         1034 | False      |

## Feature importance (last model, % gain)
|                          |   pct |
|:-------------------------|------:|
| da_dev_day               |  5.35 |
| da                       |  2.85 |
| da_spread_SE3            |  2.47 |
| last_spread_absmean_96   |  2.22 |
| da_spread_NO2            |  2.17 |
| last_spread_mean_96      |  2.02 |
| da_x_wind                |  1.97 |
| da_z_day                 |  1.88 |
| da_spread_DK2            |  1.81 |
| da_next4                 |  1.68 |
| spread_d1                |  1.65 |
| da_day_std               |  1.62 |
| da_next1                 |  1.59 |
| rev_off                  |  1.56 |
| spread_d7                |  1.55 |
| da_day_range             |  1.55 |
| f_5h_sol                 |  1.49 |
| minute_of_day            |  1.47 |
| live_thermal_dk          |  1.46 |
| last_mfrr_price_down_eur |  1.42 |
| last_pos_frac_96         |  1.42 |
| last_spread_mean_16      |  1.39 |
| rev_sol                  |  1.39 |
| da_prev4                 |  1.31 |
| fast_satisfied_demand_mw |  1.3  |

## Robustness
| metric | value |
|---|---|
| signals | 6980 |
| mwh | 3,181.80 |
| net_eur | 67,463.72 |
| net_eur_per_mwh | 21.20 |
| net_without_top20_eur | 33,106.89 |
| top20_share_pct | 50.93 |
| win_rate_pct | 59.84 |
| net_after_overlay_spreads_clipped_300_eur | 51,833.49 |

## Monthly (after risk overlay vs raw model signals)
| m       |   trades |   net_eur |   raw_signals |   raw_net_eur |
|:--------|---------:|----------:|--------------:|--------------:|
| 2025-07 |       44 |    15,899 |            44 |        15,899 |
| 2025-08 |      585 |     8,641 |           585 |         8,641 |
| 2025-09 |     1068 |    20,971 |          1092 |        23,810 |
| 2025-10 |      347 |       968 |           471 |         3,360 |
| 2025-11 |      152 |        60 |           166 |           109 |
| 2025-12 |      152 |       295 |           152 |           295 |
| 2026-01 |      477 |    -2,918 |           501 |        -4,791 |
| 2026-02 |      657 |     1,180 |           657 |         1,353 |
| 2026-03 |      496 |     2,389 |           521 |         2,786 |
| 2026-04 |      615 |     5,268 |           615 |         6,612 |
| 2026-05 |      424 |     3,365 |           424 |         3,421 |
| 2026-06 |      761 |      -693 |           913 |        -2,197 |
| 2026-07 |      248 |     3,799 |           253 |         4,078 |
| 2026-08 |      173 |     1,276 |           173 |         1,276 |
| 2026-09 |      406 |     2,281 |           413 |         2,812 |

## Direction skill vs climatology
| metric | value |
|---|---|
| brier_up | 0.18 |
| brier_skill_up_pct | 1.96 |
| auc_up | 0.61 |
| brier_down | 0.24 |
| brier_skill_down_pct | 4.10 |
| auc_down | 0.63 |

## Extra intraday baseline (1 MWh, same costs)
| strategy                 |   trades |    net_eur |   net_eur_per_mwh |
|:-------------------------|---------:|-----------:|------------------:|
| sign_of_latest_NRV (t-6) |    34229 | 127,675.47 |              3.73 |
