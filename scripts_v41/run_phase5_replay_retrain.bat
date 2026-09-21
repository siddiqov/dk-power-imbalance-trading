@echo off
REM Nurex V4.1 - Phase 5: Full walk-forward replay + retrain with Fingrid frequency data
REM
REM Prerequisites:
REM   - Fingrid API key registered in .env (FINGRID_API_KEY)
REM   - frequency table populated (54,361+ rows, checked 2026-09-21)
REM   - Phase 1 (stale-data guard, live risk overlay) applied - commit 596a3d9
REM   - Phase 2 (area-id fix, hourly cap)             applied - commit cfd808a
REM   - Phase 3 (live.py/pipeline.py importlib fix)   applied - commit be679d2
REM
REM What this does:
REM   1. Backfills any Fingrid frequency gaps (free API, ~30 sec)
REM   2. Verifies data sources (incl. frequency row count)
REM   3. Leakage test (6 random samples) - gate: must pass before replay
REM   4. Runs unit tests
REM   5. Walk-forward replay DK1 + DK2 (with Fingrid features, ~5-20 min)
REM   6. Retrains both models on full dataset including frequency features
REM   7. Prints final summary
REM
REM Output: logs\v41_phase5.log  (console stays blank on purpose - tail the log)
REM Run:    double-click this file or: cmd /c scripts_v41\run_phase5_replay_retrain.bat

cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs

set LOG=logs\v41_phase5.log
set TS=%DATE:~-4%-%DATE:~3,2%-%DATE:~0,2%_%TIME:~0,2%-%TIME:~3,2%

echo ==== PHASE 5 START %DATE% %TIME% > %LOG%

REM 1. Backfill Fingrid frequency (idempotent - skips already-loaded quarters)
echo ==== STEP 1: BACKFILL FREQUENCY >> %LOG%
"%PY%" train_v4_1.py collect --source frequency >> %LOG% 2>&1
if errorlevel 1 (
    echo [WARN] Frequency backfill reported errors - check FINGRID_API_KEY. Continuing... >> %LOG%
)

REM 2. Verify all sources including frequency
echo ==== STEP 2: SOURCE VERIFICATION >> %LOG%
"%PY%" train_v4_1.py sources >> %LOG% 2>&1

REM 3. Leakage test (6 samples) - must pass
echo ==== STEP 3: LEAKAGE TEST >> %LOG%
"%PY%" train_v4_1.py leaktest --samples 6 >> %LOG% 2>&1
if errorlevel 1 (
    echo [FATAL] Leakage test FAILED - aborting. Check %LOG% >> %LOG%
    exit /b 1
)

REM 4. Unit tests
echo ==== STEP 4: UNIT TESTS >> %LOG%
"%PY%" -m pytest tests_v41 -q >> %LOG% 2>&1
if errorlevel 1 (
    echo [WARN] Some unit tests failed - check %LOG% before proceeding. Continuing... >> %LOG%
)

REM 5. Walk-forward replay (DK1 + DK2) with Fingrid features
echo ==== STEP 5: WALK-FORWARD REPLAY (DK1+DK2 with Fingrid) >> %LOG%
"%PY%" train_v4_1.py replay >> %LOG% 2>&1
if errorlevel 1 (
    echo [FATAL] Replay FAILED - aborting retrain. Check %LOG% >> %LOG%
    exit /b 1
)

REM 6. Retrain models on full dataset including frequency features
echo ==== STEP 6: RETRAIN MODELS >> %LOG%
"%PY%" train_v4_1.py train >> %LOG% 2>&1
if errorlevel 1 (
    echo [FATAL] Training FAILED - old models kept. Check %LOG% >> %LOG%
    exit /b 1
)

REM 7. Final source/model summary
echo ==== STEP 7: FINAL SUMMARY >> %LOG%
"%PY%" train_v4_1.py sources >> %LOG% 2>&1

echo ==== PHASE 5 DONE %DATE% %TIME% >> %LOG%
echo.
echo Phase 5 complete. Results in: %LOG%
echo New replay CSVs: results\v4_1_intraday\
echo New models:      models_v4_1_intraday\
