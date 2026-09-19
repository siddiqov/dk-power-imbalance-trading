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
| trades | 9606 |
| trade_rate_pct | 24.51 |
| buy | 54 |
| sell | 9552 |
| mwh_traded | 3,135.10 |
| net_eur | 34,299.88 |
| gross_eur | 41,667.37 |
| cost_eur | 7,367.48 |
| net_eur_per_mwh | 10.94 |
| win_rate_pct | 54.51 |
| days | 410 |
| positive_days_pct | 56.59 |
| worst_day_eur | -1,515.73 |
| max_drawdown_eur | -2,913.63 |
| max_drawdown_pct_collateral | -72.45 |
| sharpe_daily_ann | 4.34 |
| net_eur_at_stress_costs | 26,932.40 |

## Forecast quality
| metric | value |
|---|---|
| mae_model | 73.65 |
| mae_zero | 66.75 |
| mae_median_q50 | 66.44 |
| dir_acc_model_pct | 38.87 |
| dir_acc_always_negative_pct | 49.40 |
| share_positive_spread_pct | 19.40 |
| brier_up_model | 0.15 |
| brier_up_climatology | 0.16 |
| pinball_q10 | 13.18 |
| coverage_q10_pct | 11.09 |
| pinball_q50 | 33.22 |
| coverage_q50_pct | 45.93 |
| pinball_q90 | 30.08 |
| coverage_q90_pct | 89.94 |

## Naive baselines (1 MWh every quarter, same costs)
| strategy                     |   trades |     net_eur |   net_eur_per_mwh |
|:-----------------------------|---------:|------------:|------------------:|
| always_sell                  |    39192 | -156,402.11 |             -3.99 |
| always_buy                   |    39192 |  -27,800.29 |             -0.71 |
| sign_of_last_published       |    39192 |  105,684.89 |              2.70 |
| sign_of_same_quarter_7d_mean |    38950 |   -2,461.68 |             -0.06 |

## Monthly
| month   |   trades |    mwh |   net_eur |   mean_abs_spread |
|:--------|---------:|-------:|----------:|------------------:|
| 2025-07 |      430 | 113.10 |   -788.33 |             86.18 |
| 2025-08 |      701 | 201.10 |  1,192.30 |             67.87 |
| 2025-09 |     1159 | 306.90 |  8,331.45 |             97.83 |
| 2025-10 |     1169 | 250.50 | -2,276.95 |             78.94 |
| 2025-11 |        2 |   0.80 |     54.17 |             78.65 |
| 2025-12 |        0 |   0.00 |      0.00 |             53.55 |
| 2026-01 |       26 |  10.40 |     32.17 |             54.85 |
| 2026-02 |      242 | 102.10 |  1,316.09 |             61.79 |
| 2026-03 |      502 | 199.50 |  4,684.53 |             61.22 |
| 2026-04 |      880 | 295.30 |  5,052.45 |             53.70 |
| 2026-05 |     1098 | 398.90 |  2,070.53 |             60.66 |
| 2026-06 |      911 | 337.00 |  5,124.66 |             55.25 |
| 2026-07 |     1174 | 441.20 |  5,555.66 |             69.16 |
| 2026-08 |      947 | 309.60 | -2,459.15 |             56.76 |
| 2026-09 |      365 | 168.70 |  6,410.29 |             67.74 |

## Retraining periods and chosen thresholds
| period_start        | cut                 |   n_train | buy                            | sell                           |   val_net_eur |   val_trades | no_trade   |
|:--------------------|:--------------------|----------:|:-------------------------------|:-------------------------------|--------------:|-------------:|:-----------|
| 2025-07-02 00:00:00 | 2025-07-01 23:00:00 |     11469 | {'margin': 40.0, 'pmin': 0.45} | {'margin': 20.0, 'pmin': 0.65} |      19576.2  |         3321 | False      |
| 2025-08-01 00:00:00 | 2025-07-31 23:00:00 |     14349 | {'margin': 40.0, 'pmin': 0.5}  | {'margin': 20.0, 'pmin': 0.45} |       8960.11 |         3214 | False      |
| 2025-08-31 00:00:00 | 2025-08-30 23:00:00 |     17229 |                                | {'margin': 0.0, 'pmin': 0.45}  |       6027.46 |         2990 | False      |
| 2025-09-30 00:00:00 | 2025-09-29 23:00:00 |     20109 | {'margin': 0.0, 'pmin': 0.45}  | {'margin': 0.0, 'pmin': 0.55}  |      16763.3  |         2338 | False      |
| 2025-10-30 00:00:00 | 2025-10-29 23:00:00 |     22989 |                                | {'margin': 0.0, 'pmin': 0.7}   |       5463.58 |          984 | False      |
| 2025-11-29 00:00:00 | 2025-12-15 00:30:00 |     24147 |                                |                                |          0    |            0 | True       |
| 2025-12-29 00:00:00 | 2025-12-28 23:00:00 |     25479 |                                |                                |          0    |            0 | True       |
| 2026-01-28 00:00:00 | 2026-01-27 23:00:00 |     28359 | {'margin': 10.0, 'pmin': 0.45} | {'margin': 20.0, 'pmin': 0.45} |       5438.01 |          609 | False      |
| 2026-02-27 00:00:00 | 2026-02-26 23:00:00 |     31239 |                                | {'margin': 20.0, 'pmin': 0.45} |       3327.9  |          876 | False      |
| 2026-03-29 00:00:00 | 2026-03-28 23:00:00 |     34119 |                                | {'margin': 5.0, 'pmin': 0.45}  |       9053.32 |         1398 | False      |
| 2026-04-28 00:00:00 | 2026-04-27 23:00:00 |     36999 |                                | {'margin': 0.0, 'pmin': 0.45}  |      13923.5  |         2217 | False      |
| 2026-05-28 00:00:00 | 2026-05-27 23:00:00 |     39879 | {'margin': 20.0, 'pmin': 0.45} | {'margin': 5.0, 'pmin': 0.45}  |      25158.9  |         1626 | False      |
| 2026-06-27 00:00:00 | 2026-06-26 23:00:00 |     42759 |                                | {'margin': 5.0, 'pmin': 0.45}  |       7717.66 |         1905 | False      |
| 2026-07-27 00:00:00 | 2026-07-26 23:00:00 |     45639 |                                | {'margin': 0.0, 'pmin': 0.45}  |       9215.75 |         2351 | False      |
| 2026-08-26 00:00:00 | 2026-08-25 23:00:00 |     48519 |                                | {'margin': 0.0, 'pmin': 0.5}   |       5841.28 |         1537 | False      |

## Feature importance (last model, % gain)
|                        |   pct |
|:-----------------------|------:|
| da_dev_day             |  4.87 |
| last_spread_absmean_96 |  3.75 |
| da                     |  3.06 |
| da_spread_DE           |  2.65 |
| da_z_day               |  2.46 |
| da_spread_DK1          |  2.32 |
| da_spread_SE4          |  2.17 |
| rev_off                |  2.06 |
| rev_sol_recent         |  1.98 |
| last_spread_mean_96    |  1.96 |
| da_day_mean            |  1.84 |
| last_spread_mean_16    |  1.8  |
| month                  |  1.58 |
| da_day_range           |  1.55 |
| f_da_on                |  1.53 |
| last_pos_frac_96       |  1.53 |
| spread_d1              |  1.49 |
| live_wind_err_dk       |  1.49 |
| da_x_wind              |  1.36 |
| da_next4               |  1.35 |
| rev_wind               |  1.33 |
| last_spread_std_16     |  1.31 |
| rev_wind_recent        |  1.3  |
| rev_on                 |  1.29 |
| last_spread_absmean_16 |  1.27 |

## Robustness
| metric | value |
|---|---|
| signals | 9968 |
| mwh | 3,523.50 |
| net_eur | 38,112.77 |
| net_eur_per_mwh | 10.82 |
| net_without_top20_eur | 32,799.97 |
| top20_share_pct | 13.94 |
| win_rate_pct | 54.52 |
| net_after_overlay_spreads_clipped_300_eur | 50,405.08 |

## Monthly (after risk overlay vs raw model signals)
| m       |   trades |   net_eur |   raw_signals |   raw_net_eur |
|:--------|---------:|----------:|--------------:|--------------:|
| 2025-07 |      430 |      -788 |           578 |           684 |
| 2025-08 |      701 |     1,192 |           740 |         1,196 |
| 2025-09 |     1167 |     7,676 |          1191 |         9,027 |
| 2025-10 |     1161 |    -1,621 |          1246 |        -1,197 |
| 2025-11 |        2 |        54 |             2 |            54 |
| 2025-12 |        0 |         0 |             0 |             0 |
| 2026-01 |       26 |        32 |            26 |            32 |
| 2026-02 |      243 |     1,332 |           243 |         1,456 |
| 2026-03 |      506 |     4,717 |           506 |         4,717 |
| 2026-04 |      879 |     5,028 |           894 |         5,738 |
| 2026-05 |     1095 |     2,046 |          1096 |         2,045 |
| 2026-06 |      911 |     5,137 |           911 |         5,436 |
| 2026-07 |     1178 |     5,669 |          1186 |         6,219 |
| 2026-08 |      944 |    -2,585 |           986 |        -3,706 |
| 2026-09 |      363 |     6,412 |           363 |         6,412 |

## Direction skill vs climatology
| metric | value |
|---|---|
| brier_up | 0.15 |
| brier_skill_up_pct | 1.18 |
| auc_up | 0.59 |
| brier_down | 0.24 |
| brier_skill_down_pct | 2.42 |
| auc_down | 0.60 |

## Extra intraday baseline (1 MWh, same costs)
| strategy                 |   trades |    net_eur |   net_eur_per_mwh |
|:-------------------------|---------:|-----------:|------------------:|
| sign_of_latest_NRV (t-6) |    30076 | 224,144.32 |              7.45 |
