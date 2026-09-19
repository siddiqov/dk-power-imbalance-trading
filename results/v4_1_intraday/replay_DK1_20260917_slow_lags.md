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
| trades | 6478 |
| trade_rate_pct | 16.43 |
| buy | 556 |
| sell | 5922 |
| mwh_traded | 2,470.80 |
| net_eur | 64,670.62 |
| gross_eur | 70,477.00 |
| cost_eur | 5,806.38 |
| net_eur_per_mwh | 26.17 |
| win_rate_pct | 60.25 |
| days | 413 |
| positive_days_pct | 55.69 |
| worst_day_eur | -1,602.61 |
| max_drawdown_eur | -2,836.54 |
| max_drawdown_pct_collateral | -70.54 |
| sharpe_daily_ann | 3.51 |
| net_eur_at_stress_costs | 58,864.24 |

## Forecast quality
| metric | value |
|---|---|
| mae_model | 68.02 |
| mae_zero | 59.12 |
| mae_median_q50 | 57.07 |
| dir_acc_model_pct | 41.21 |
| dir_acc_always_negative_pct | 53.33 |
| share_positive_spread_pct | 25.41 |
| brier_up_model | 0.19 |
| brier_up_climatology | 0.19 |
| pinball_q10 | 10.35 |
| coverage_q10_pct | 12.14 |
| pinball_q50 | 28.53 |
| coverage_q50_pct | 48.41 |
| pinball_q90 | 25.07 |
| coverage_q90_pct | 88.81 |

## Naive baselines (1 MWh every quarter, same costs)
| strategy                     |   trades |     net_eur |   net_eur_per_mwh |
|:-----------------------------|---------:|------------:|------------------:|
| always_sell                  |    39438 | -116,371.38 |             -2.95 |
| always_buy                   |    39438 |  -68,987.22 |             -1.75 |
| sign_of_last_published       |    39438 |   91,146.10 |              2.31 |
| sign_of_same_quarter_7d_mean |    39218 |  -13,247.29 |             -0.34 |

## Monthly
| month   |   trades |    mwh |   net_eur |   mean_abs_spread |
|:--------|---------:|-------:|----------:|------------------:|
| 2025-07 |       32 |  44.80 | 12,833.23 |             84.74 |
| 2025-08 |      989 | 321.80 | 10,878.58 |             58.64 |
| 2025-09 |     1064 | 483.70 | 26,260.34 |             87.04 |
| 2025-10 |      667 | 205.70 |    100.66 |             69.81 |
| 2025-11 |      241 |  71.30 |     61.45 |             64.50 |
| 2025-12 |      333 | 112.60 |    890.74 |             40.11 |
| 2026-01 |      150 | 157.00 |   -704.85 |             47.20 |
| 2026-02 |       60 |  18.00 |    438.44 |             34.99 |
| 2026-03 |      132 |  39.10 |   -196.67 |             49.42 |
| 2026-04 |      597 | 159.80 |  3,294.24 |             43.92 |
| 2026-05 |      388 | 194.00 |  2,701.28 |             44.72 |
| 2026-06 |      864 | 277.20 |    107.29 |             71.68 |
| 2026-07 |      249 | 120.90 |  2,615.79 |             65.61 |
| 2026-08 |      241 | 122.40 |  3,310.05 |             55.33 |
| 2026-09 |      471 | 142.50 |  2,080.04 |             70.15 |

## Retraining periods and chosen thresholds
| period_start        | cut                 |   n_train | buy                            | sell                           |   val_net_eur |   val_trades | no_trade   |
|:--------------------|:--------------------|----------:|:-------------------------------|:-------------------------------|--------------:|-------------:|:-----------|
| 2025-07-02 00:00:00 | 2025-07-01 23:00:00 |     11468 | {'margin': 10.0, 'pmin': 0.45} |                                |      45510.9  |          598 | False      |
| 2025-08-01 00:00:00 | 2025-07-31 23:00:00 |     14348 | {'margin': 20.0, 'pmin': 0.45} | {'margin': 10.0, 'pmin': 0.6}  |      27916.8  |         2729 | False      |
| 2025-08-31 00:00:00 | 2025-08-30 23:00:00 |     17228 | {'margin': 0.0, 'pmin': 0.45}  | {'margin': 0.0, 'pmin': 0.55}  |      30552.9  |         2880 | False      |
| 2025-09-30 00:00:00 | 2025-09-29 23:00:00 |     20108 | {'margin': 0.0, 'pmin': 0.45}  | {'margin': 0.0, 'pmin': 0.6}   |      30336.5  |         2153 | False      |
| 2025-10-30 00:00:00 | 2025-10-29 23:00:00 |     22988 | {'margin': 40.0, 'pmin': 0.45} | {'margin': 5.0, 'pmin': 0.6}   |      35688.2  |         1704 | False      |
| 2025-11-29 00:00:00 | 2025-12-12 11:00:00 |     24147 | {'margin': 40.0, 'pmin': 0.45} | {'margin': 0.0, 'pmin': 0.6}   |       7026.72 |          438 | False      |
| 2025-12-29 00:00:00 | 2025-12-28 23:00:00 |     25724 | {'margin': 0.0, 'pmin': 0.5}   | {'margin': 40.0, 'pmin': 0.6}  |       2510.25 |          141 | False      |
| 2026-01-28 00:00:00 | 2026-01-27 23:00:00 |     28604 |                                | {'margin': 40.0, 'pmin': 0.45} |        411.01 |           77 | False      |
| 2026-02-27 00:00:00 | 2026-02-26 23:00:00 |     31484 |                                | {'margin': 40.0, 'pmin': 0.45} |        567.7  |          114 | False      |
| 2026-03-29 00:00:00 | 2026-03-28 23:00:00 |     34364 |                                | {'margin': 5.0, 'pmin': 0.55}  |       6839.92 |         1455 | False      |
| 2026-04-28 00:00:00 | 2026-04-27 23:00:00 |     37244 | {'margin': 20.0, 'pmin': 0.45} | {'margin': 10.0, 'pmin': 0.55} |      12359.7  |         1355 | False      |
| 2026-05-28 00:00:00 | 2026-05-27 23:00:00 |     40124 | {'margin': 10.0, 'pmin': 0.45} | {'margin': 0.0, 'pmin': 0.45}  |       7275.16 |         1829 | False      |
| 2026-06-27 00:00:00 | 2026-06-26 23:00:00 |     43004 | {'margin': 20.0, 'pmin': 0.45} | {'margin': 20.0, 'pmin': 0.5}  |       6732.61 |          522 | False      |
| 2026-07-27 00:00:00 | 2026-07-26 23:00:00 |     45884 | {'margin': 0.0, 'pmin': 0.45}  | {'margin': 20.0, 'pmin': 0.45} |       6056.62 |          547 | False      |
| 2026-08-26 00:00:00 | 2026-08-25 23:00:00 |     48764 |                                | {'margin': 5.0, 'pmin': 0.5}   |       5918.61 |         1146 | False      |

## Feature importance (last model, % gain)
|                          |   pct |
|:-------------------------|------:|
| da_dev_day               |  5.31 |
| da                       |  2.98 |
| da_spread_SE3            |  2.51 |
| last_spread_absmean_96   |  2.36 |
| da_spread_NO2            |  2.22 |
| last_spread_mean_96      |  2.12 |
| da_x_wind                |  1.98 |
| da_z_day                 |  1.88 |
| da_next4                 |  1.74 |
| da_spread_DK2            |  1.73 |
| da_next1                 |  1.69 |
| da_day_std               |  1.66 |
| da_day_range             |  1.66 |
| spread_d7                |  1.57 |
| spread_d1                |  1.57 |
| rev_sol                  |  1.54 |
| minute_of_day            |  1.5  |
| last_mfrr_price_down_eur |  1.48 |
| rev_off                  |  1.43 |
| last_pos_frac_96         |  1.4  |
| f_5h_sol                 |  1.36 |
| da_day_mean              |  1.36 |
| da_prev4                 |  1.35 |
| live_thermal_dk          |  1.34 |
| da_prev1                 |  1.34 |

## Robustness
| metric | value |
|---|---|
| signals | 6872 |
| mwh | 2,853.20 |
| net_eur | 71,267.94 |
| net_eur_per_mwh | 24.98 |
| net_without_top20_eur | 36,207.23 |
| top20_share_pct | 49.20 |
| win_rate_pct | 60.45 |
| net_after_overlay_spreads_clipped_300_eur | 50,205.21 |

## Monthly (after risk overlay vs raw model signals)
| m       |   trades |   net_eur |   raw_signals |   raw_net_eur |
|:--------|---------:|----------:|--------------:|--------------:|
| 2025-07 |       32 |    12,833 |            32 |        12,833 |
| 2025-08 |      992 |    10,851 |           996 |        11,065 |
| 2025-09 |     1061 |    26,287 |          1098 |        29,311 |
| 2025-10 |      667 |       101 |           904 |         2,385 |
| 2025-11 |      241 |        61 |           247 |          -455 |
| 2025-12 |      333 |       891 |           333 |           891 |
| 2026-01 |      150 |      -705 |           150 |          -682 |
| 2026-02 |       60 |       438 |            60 |           438 |
| 2026-03 |      134 |      -179 |           159 |           270 |
| 2026-04 |      595 |     3,277 |           599 |         4,214 |
| 2026-05 |      388 |     2,701 |           388 |         2,680 |
| 2026-06 |      864 |       107 |           914 |           837 |
| 2026-07 |      249 |     2,616 |           280 |         2,030 |
| 2026-08 |      243 |     3,311 |           243 |         3,305 |
| 2026-09 |      469 |     2,079 |           469 |         2,145 |

## Direction skill vs climatology
| metric | value |
|---|---|
| brier_up | 0.18 |
| brier_skill_up_pct | 1.71 |
| auc_up | 0.60 |
| brier_down | 0.24 |
| brier_skill_down_pct | 3.74 |
| auc_down | 0.62 |

## Extra intraday baseline (1 MWh, same costs)
| strategy                 |   trades |   net_eur |   net_eur_per_mwh |
|:-------------------------|---------:|----------:|------------------:|
| sign_of_latest_NRV (t-6) |    34228 | 95,741.18 |              2.80 |
