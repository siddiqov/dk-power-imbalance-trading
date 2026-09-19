# Nurex V4.1 Intraday walk-forward replay — DK2

Book: `v41id-replay:DK2`  
Test window: 2025-07-02 00:00:00 → 2026-09-17 07:15:00 UTC  
Cost model: 2.35 EUR/MWh (stress x2.0)  
Collateral: 4,021 EUR  
Every quarter was forecast with information available at intraday gate closure (delivery start minus 60 min); models retrained every 30 days on labels known at that time.

## Trading
| metric | value |
|---|---|
| quarters | 39192 |
| trades | 8452 |
| trade_rate_pct | 21.57 |
| buy | 63 |
| sell | 8389 |
| mwh_traded | 2,797.80 |
| net_eur | 22,473.82 |
| gross_eur | 29,048.65 |
| cost_eur | 6,574.83 |
| net_eur_per_mwh | 8.03 |
| win_rate_pct | 54.65 |
| days | 410 |
| positive_days_pct | 50.49 |
| worst_day_eur | -2,405.91 |
| max_drawdown_eur | -5,814.85 |
| max_drawdown_pct_collateral | -144.60 |
| sharpe_daily_ann | 2.44 |
| net_eur_at_stress_costs | 15,898.99 |

## Forecast quality
| metric | value |
|---|---|
| mae_model | 73.86 |
| mae_zero | 66.75 |
| mae_median_q50 | 66.60 |
| dir_acc_model_pct | 38.93 |
| dir_acc_always_negative_pct | 49.40 |
| share_positive_spread_pct | 19.40 |
| brier_up_model | 0.15 |
| brier_up_climatology | 0.16 |
| pinball_q10 | 13.20 |
| coverage_q10_pct | 10.95 |
| pinball_q50 | 33.30 |
| coverage_q50_pct | 45.95 |
| pinball_q90 | 30.17 |
| coverage_q90_pct | 90.01 |

## Naive baselines (1 MWh every quarter, same costs)
| strategy                     |   trades |     net_eur |   net_eur_per_mwh |
|:-----------------------------|---------:|------------:|------------------:|
| always_sell                  |    39192 | -156,402.11 |             -3.99 |
| always_buy                   |    39192 |  -27,800.29 |             -0.71 |
| sign_of_last_published       |    39192 |   83,353.25 |              2.13 |
| sign_of_same_quarter_7d_mean |    38950 |   -2,461.68 |             -0.06 |

## Monthly
| month   |   trades |    mwh |   net_eur |   mean_abs_spread |
|:--------|---------:|-------:|----------:|------------------:|
| 2025-07 |      566 | 146.80 |   -690.03 |             86.18 |
| 2025-08 |      157 |  47.10 |  1,214.24 |             67.87 |
| 2025-09 |      606 | 181.60 |  3,097.32 |             97.83 |
| 2025-10 |      945 | 252.50 |   -383.61 |             78.94 |
| 2025-11 |      141 |  35.80 | -1,632.96 |             78.65 |
| 2025-12 |        0 |   0.00 |      0.00 |             53.55 |
| 2026-01 |       34 |  13.60 |    164.87 |             54.85 |
| 2026-02 |      468 | 161.60 |  2,282.08 |             61.79 |
| 2026-03 |      218 |  85.30 |  1,436.15 |             61.22 |
| 2026-04 |      966 | 305.70 |  3,667.57 |             53.70 |
| 2026-05 |      973 | 370.20 |  2,105.47 |             60.66 |
| 2026-06 |     1005 | 353.50 |  4,383.93 |             55.25 |
| 2026-07 |      954 | 366.70 |  3,722.95 |             69.16 |
| 2026-08 |     1025 | 291.50 | -3,162.21 |             56.76 |
| 2026-09 |      394 | 185.90 |  6,268.05 |             67.74 |

## Retraining periods and chosen thresholds
| period_start        | cut                 |   n_train | buy                            | sell                           |   val_net_eur |   val_trades | no_trade   |
|:--------------------|:--------------------|----------:|:-------------------------------|:-------------------------------|--------------:|-------------:|:-----------|
| 2025-07-02 00:00:00 | 2025-07-01 23:00:00 |     11468 | {'margin': 40.0, 'pmin': 0.45} | {'margin': 10.0, 'pmin': 0.65} |      20365.1  |         3531 | False      |
| 2025-08-01 00:00:00 | 2025-07-31 23:00:00 |     14348 |                                | {'margin': 40.0, 'pmin': 0.7}  |       6174.66 |         1890 | False      |
| 2025-08-31 00:00:00 | 2025-08-30 23:00:00 |     17228 | {'margin': 0.0, 'pmin': 0.45}  | {'margin': 10.0, 'pmin': 0.55} |      14964.9  |         2071 | False      |
| 2025-09-30 00:00:00 | 2025-09-29 23:00:00 |     20108 |                                | {'margin': 10.0, 'pmin': 0.45} |      11903.1  |         2306 | False      |
| 2025-10-30 00:00:00 | 2025-10-29 23:00:00 |     22988 |                                | {'margin': 10.0, 'pmin': 0.45} |       6944.28 |         1522 | False      |
| 2025-11-29 00:00:00 | 2025-12-15 00:30:00 |     24147 |                                |                                |          0    |            0 | True       |
| 2025-12-29 00:00:00 | 2025-12-28 23:00:00 |     25478 |                                |                                |          0    |            0 | True       |
| 2026-01-28 00:00:00 | 2026-01-27 23:00:00 |     28358 |                                | {'margin': 10.0, 'pmin': 0.45} |       2315.04 |          880 | False      |
| 2026-02-27 00:00:00 | 2026-02-26 23:00:00 |     31238 | {'margin': 20.0, 'pmin': 0.45} | {'margin': 0.0, 'pmin': 0.65}  |       3436.79 |          551 | False      |
| 2026-03-29 00:00:00 | 2026-03-28 23:00:00 |     34118 |                                | {'margin': 0.0, 'pmin': 0.45}  |       8164.66 |         1616 | False      |
| 2026-04-28 00:00:00 | 2026-04-27 23:00:00 |     36998 |                                | {'margin': 5.0, 'pmin': 0.45}  |      13244.1  |         2109 | False      |
| 2026-05-28 00:00:00 | 2026-05-27 23:00:00 |     39878 | {'margin': 0.0, 'pmin': 0.45}  | {'margin': 0.0, 'pmin': 0.45}  |      19052    |         1845 | False      |
| 2026-06-27 00:00:00 | 2026-06-26 23:00:00 |     42758 |                                | {'margin': 5.0, 'pmin': 0.5}   |       6312.6  |         1486 | False      |
| 2026-07-27 00:00:00 | 2026-07-26 23:00:00 |     45638 | {'margin': 10.0, 'pmin': 0.45} | {'margin': 0.0, 'pmin': 0.45}  |       9627    |         2333 | False      |
| 2026-08-26 00:00:00 | 2026-08-25 23:00:00 |     48518 |                                | {'margin': 0.0, 'pmin': 0.5}   |       5100.83 |         1594 | False      |

## Feature importance (last model, % gain)
|                        |   pct |
|:-----------------------|------:|
| da_dev_day             |  5.04 |
| last_spread_absmean_96 |  3.78 |
| da                     |  3.02 |
| da_spread_DE           |  2.9  |
| da_z_day               |  2.59 |
| da_spread_DK1          |  2.41 |
| rev_sol_recent         |  2.2  |
| rev_off                |  2.15 |
| da_spread_SE4          |  2.11 |
| last_spread_mean_96    |  2.06 |
| last_spread_mean_16    |  1.8  |
| da_day_mean            |  1.78 |
| month                  |  1.76 |
| live_wind_err_dk       |  1.59 |
| da_day_range           |  1.58 |
| last_pos_frac_96       |  1.57 |
| da_next4               |  1.43 |
| spread_d1              |  1.43 |
| rev_on                 |  1.37 |
| da_x_wind              |  1.37 |
| last_spread_std_16     |  1.36 |
| rev_sol                |  1.34 |
| live_thermal_dk_d1h    |  1.3  |
| rev_wind               |  1.29 |
| da_day_std             |  1.21 |

## Robustness
| metric | value |
|---|---|
| signals | 9421 |
| mwh | 3,430.10 |
| net_eur | 24,252.92 |
| net_eur_per_mwh | 7.07 |
| net_without_top20_eur | 16,164.27 |
| top20_share_pct | 33.35 |
| win_rate_pct | 54.08 |
| net_after_overlay_spreads_clipped_300_eur | 37,035.97 |

## Monthly (after risk overlay vs raw model signals)
| m       |   trades |   net_eur |   raw_signals |   raw_net_eur |
|:--------|---------:|----------:|--------------:|--------------:|
| 2025-07 |      566 |      -690 |           751 |         1,864 |
| 2025-08 |      157 |     1,214 |           157 |         1,214 |
| 2025-09 |      606 |     3,097 |           892 |         4,118 |
| 2025-10 |      946 |      -381 |          1318 |        -1,488 |
| 2025-11 |      140 |    -1,635 |           140 |        -2,060 |
| 2025-12 |        0 |         0 |             0 |             0 |
| 2026-01 |       34 |       165 |            34 |           165 |
| 2026-02 |      468 |     2,282 |           478 |         2,922 |
| 2026-03 |      225 |     1,382 |           225 |         1,382 |
| 2026-04 |      962 |     3,746 |          1001 |         4,389 |
| 2026-05 |      971 |     2,081 |           971 |         2,081 |
| 2026-06 |     1004 |     4,385 |          1004 |         4,537 |
| 2026-07 |      959 |     3,848 |           972 |         4,465 |
| 2026-08 |     1023 |    -3,289 |          1087 |        -5,604 |
| 2026-09 |      391 |     6,270 |           391 |         6,270 |

## Direction skill vs climatology
| metric | value |
|---|---|
| brier_up | 0.15 |
| brier_skill_up_pct | 0.99 |
| auc_up | 0.59 |
| brier_down | 0.24 |
| brier_skill_down_pct | 2.25 |
| auc_down | 0.60 |

## Extra intraday baseline (1 MWh, same costs)
| strategy                 |   trades |    net_eur |   net_eur_per_mwh |
|:-------------------------|---------:|-----------:|------------------:|
| sign_of_latest_NRV (t-6) |    30076 | 157,168.04 |              5.23 |
