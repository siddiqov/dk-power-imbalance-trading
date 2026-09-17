# Nurex V4.1 Intraday

This package replaces the old V4.1 trainer (`train_v4_1.py` + `src/feature_engineering_v4_1.py` + `src/commercial_strategy_v4_1.py`) for **intraday** trading. It is simulation / paper trading only: no orders are sent.

## Why it was rebuilt (audit 17 Sep 2026)

| Audit finding | What changed |
|---|---|
| `shift(4)` used quarter t−4, which is still in delivery at T−60; the imbalance price is also published later | Every source has its own publication lag (config `availability`), measured in time, not in rows. |
| `wind_forecast_revision` used the final intraday forecast (leak) | Uses Forecast5Hour and Forecast1Hour only once they are published. |
| `ffill().bfill()` pulled future values backwards | No filling. Missing values stay NaN (LightGBM handles NaN natively). |
| Legacy `energy_data.db`: 195-day DK1 hole, duplicate balance rows, wrong DK1 load sign since Jan 2026, DK2 truncated at 25,000 rows | Uses the paginated V4.2 store `Nurex_V4_2/data/nurex42.duckdb` (both zones from Mar 2025 to today). |
| Calendar features were in UTC | Local Danish time, DST-safe, plus Danish public holidays. |
| Calibration used non-time-ordered folds; quantile models were never validated; no CV report | Walk-forward replay: retrain every 30 days, thresholds tuned on the last 60 days, markdown/CSV/JSON report. |
| EV used one magnitude for wins and losses (always SELL); minimum volume overrode the risk cap | V4.2 two-part model (P(up/flat/down) × regime-specific size). Sizing respects the stressed-loss cap; there is no minimum-volume override. |
| Built-in leakage test was invalid | `leaktest` corrupts all post-decision raw data and asserts that no feature changes. |
| Dashboard overwrote model inputs with a live snapshot, used a 0.51 € fee, and dropped negative prices | The dashboard only displays V4.1 decisions from `dashboard_adapter.py`. One cost model is used everywhere, and negative prices are settled. |

## Decision and information set

For delivery quarter *q*, the decision time is *a = q − 60 min* (`gate_lead_minutes`). The model may use:

- Day-ahead prices for DK1/DK2/DE/NO2/SE3/SE4 once published (D−1, 13:00 local).
- Wind and solar forecasts: day-ahead, 5-hour, and 1-hour (the 1-hour forecast only from q − 60 min).
- The imbalance price / spread of quarters that ended at least 30 min before *a*.
- Satisfied demand (NRV), dominating direction and aFRR volumes of quarters that ended at least 15 min before *a* (observed on EDS: about 7 min).
- PowerSystemRightNow 1-minute data (flows, aFRR, wind/solar/thermal) up to *a* − 5 min.
- The same state for the other Danish zone, and same-quarter history from previous days.

The target is spread = imbalance price − day-ahead price. The traded intraday price is approximated by the day-ahead price plus slippage in the cost (2.35 €/MWh in total, an assumption). **Replace this with the real intraday price once Nord Pool intraday data is available.**

## Data sources (added 17 Sep 2026)

| Source | What V4.1 uses | Known at decision time | History |
|---|---|---|---|
| Energinet EDS (V4.2 store) | imbalance/NRV/aFRR, DA prices, wind/solar forecasts, live flows | per-source lags | Mar 2025 → now |
| **ENTSO-E** (`ENTSOE_API_KEY`) | actual load + DA load forecast (DK1, DK2, DE), DA schedules, physical flows, NTC per border, DE wind/solar forecast | DA: D-1 13:00 (wind/solar 18:00); actuals: +90 min | backfill |
| **Live flow − schedule** | PSRN physical flow per cable minus ENTSO-E schedule (sign auto-calibrated) | +5 min | with ENTSO-E |
| **Nord Pool UMM** (public) | unavailable MW: own zone planned/unplanned, transmission, neighbours, new unplanned in last 2 h | message version valid at decision time | backfill (paged) |
| **Open-Meteo previous-day run** | temperature, 100 m wind, cloud cover, radiation (+7-day anomaly, next-hour change) | forecast issued ≥ 1 day before | backfill |
| **Fingrid frequency** (`FINGRID_API_KEY`, free) | Nordic frequency deviation, volatility, range (DK2 relevance) | +5 min | backfill |
| **Nord Pool Intraday API** (read-only) | 15-min and hourly contract VWAP/last/range/turnover, ID−DA, last-hour trades and signed aggressor volume, order book mid/spread/depth imbalance | receive time | **live only**: recorded from now on |

Features of a source enter the model automatically once it covers more than 2% of the training rows. The Nord Pool intraday features therefore activate after about two weeks of recording, and become meaningful after a few months.

## Paper-trading journal (locked decisions)

`cycle` locks every quarter whose gate closes in the next 20 minutes, using only the data and model available at that moment, and writes it to `data/v41_journal.sqlite`. A locked decision is never changed, even after retraining. If the PC was off when a gate passed, the quarter is stored as MISSED/HOLD; it is never decided later with newer information. Settlement adds the published imbalance spread and the PnL (MWh × spread − MWh × cost). The dashboard marks locked decisions with 🔒 and shows the journal totals in the sidebar. `hour_all_same_side = 1` flags hours where all four quarters point the same way (hourly-product candidates, 10 MW hourly = 2.5 MWh per quarter).

## Commands (run from `Basic_Approach`)

```
python train_v4_1.py update       # new data (Nurex_V4_2/run.py update)
python train_v4_1.py leaktest     # must print PASS
python train_v4_1.py replay       # walk-forward report -> results/v4_1_intraday/
python train_v4_1.py train        # live models -> models_v4_1_intraday/
python train_v4_1.py predict --area DK1 --day 2026-09-17
python train_v4_1.py all          # update + collect + leaktest + replay + train
python train_v4_1.py collect      # ENTSO-E, UMM, weather, frequency (first run = full backfill)
python train_v4_1.py sources      # coverage of every source
python train_v4_1.py probe-nordpool   # Nord Pool login + 60 s of live data (check units/areas)
python train_v4_1.py record-intraday  # 24/7 Nord Pool recorder -> Nurex_V4_2/data/nordpool_id/
python train_v4_1.py cycle        # update + collect + lock upcoming gates + settle (scheduled every 15 min)
python train_v4_1.py lock         # lock upcoming gates + settle only
python train_v4_1.py journal      # honest paper-trading result (locked decisions only)
powershell -ExecutionPolicy Bypass -File scripts_v41\install_tasks_v41.ps1   # cycle (15 min) + recorder + weekly retrain
python -m pytest tests_v41 -q     # parsers, point-in-time, leakage (incl. all new sources)
streamlit run dashboard_v4_1.py --server.port 5005
```

Requirements: the same as `Nurex_V4_2/requirements.txt` (lightgbm, duckdb, pandas, pyyaml, tabulate for reports).

## Open points before real money

1. Nord Pool intraday history: the read-only API is live only. For backtests on past intraday prices, the Nord Pool intraday market data service is needed.
2. Confirm the Danish local gate closure, the BRP cost and the permission for deliberate imbalance positions.
3. Confirm publication delays on the production feed; if the feed is slower, raise the `availability` lags.
4. Liquidity: the 10 MWh per quarter cap must fit the traded volume of DK 15-min contracts.
