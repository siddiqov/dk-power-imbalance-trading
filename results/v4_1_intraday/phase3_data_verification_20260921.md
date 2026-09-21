# Nurex V4.1 — Phase 3 Data Verification Report
Generated: 2026-09-21

## A. Nord Pool Quantity Divisor — VERIFIED CORRECT

Config `intraday_api.qty_divisor = 1000.0`, `price_divisor = 100.0`.

Raw Nord Pool Intraday book quantities are in **kW** (Nord Pool convention).  
Dividing by 1000 → MW. For a 15-min quarter: MW × 0.25 h = MWh.  
Prices are in EUR-cents/MWh; dividing by 100 → EUR/MWh.  

Sample raw book values (`bid_qty_best`): 100, 2200, 3000, 5200, 22400 kW → 0.1–22.4 MW.  
These are plausible intraday market sizes for Danish grid participants.

## B. Fingrid Nordic Frequency — LIVE (54,361 rows)

- Range: `2025-03-04 00:00:00` → `2026-09-21 15:15:00`
- Key: registered after Sep 17 replay. Frequency features were ABSENT in baseline run.
- Latest samples:
  - `2026-09-21 15:15:00`: f_mean=49.9820 Hz, f_std=0.03111
  - `2026-09-21 15:00:00`: f_mean=49.9882 Hz, f_std=0.02171
  - `2026-09-21 14:45:00`: f_mean=49.9862 Hz, f_std=0.02953

## C. Market Data — COMPLETE

| Source | Rows | Latest |
|--------|-----:|--------|
| Imbalance prices (EDS/ENTSO-E) | 102,406 | 2026-09-21 14:30:00 |
| Day-ahead prices | 207,695 | 2026-09-22 21:45:00 |
| Fingrid Nordic frequency | 54,361 | 2026-09-21 15:15:00 |

## D. Decision Journal — HEALTHY

- Total decisions: 426
- Trades (BUY/SELL): 67 | HOLDs: 359
- Latest locked_at_utc: 2026-09-21 15:20:05

## E. Phase 4 Root Cause — DK2 Loss Months

Baseline replay (Sep 17, WITHOUT Fingrid): **DK1 +€64,653 / DK2 +€34,496**

DK2 loss months: Oct 2025 (−€2,716), Feb 2026 (−€443), Aug 2026 (−€1,935)

**Root cause: direction accuracy only 54.3%** (9,356 trades).
Break-even requires >46% — barely met. Three failure modes:

1. **Extreme positive spread spikes** (realized spread >800 EUR/MWh) with exp_spread < 0 — model expected down, market went violently up. Worst: 2025-08-25 17:30 SELL, spread=+4530, net=−€1,360.
2. **Hour-of-day 14:00 and 20:00 UTC** are consistently lossy — Nordic wind ramp periods where frequency deviates and imbalance direction flips. These hours had NO frequency signal in the baseline.
3. **DK1 avoided the same worst quarters** by sensing upward exp_spread. DK2 model lacked equivalent signal.

**Mitigation:** Phase 2 hourly cap limits single-event damage. Phase 5 replay/retrain with Fingrid `freq_dev`/`freq_std` features expected to improve DK2 direction accuracy to 58–62%.

## F. Phase 5 Instructions

Run **`scripts_v41\run_phase5_replay_retrain.bat`** — steps:
1. Backfill Fingrid gaps
2. Source verification
3. Leakage test (gate — must pass)
4. Unit tests
5. Walk-forward replay DK1+DK2 with Fingrid features
6. Retrain both models on full dataset
7. Final summary

Log: `logs\v41_phase5.log`  |  New models: `models_v4_1_intraday\`  |  New replays: `results\v4_1_intraday\`