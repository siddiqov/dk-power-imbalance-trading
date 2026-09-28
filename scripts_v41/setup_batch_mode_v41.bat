@echo off
REM ============================================================================
REM Nurex V4.1 - switch to the client's HOURLY BATCH schedule (2026-09-27)
REM   every hour the next hour's 4 quarters are LOCKED 2h15 before the first quarter
REM   (21:45 -> Q1-Q4, 07:45 -> Q41-Q44); all later quarters stay provisional.
REM
REM Steps (log: logs\v41_batch_setup.log):
REM   0. private copy of the store, so the live 15-min cycle is never blocked
REM   1. leak test at the batch decision times          (must PASS)
REM   2. walk-forward replay DK1 + DK2 at batch schedule (report: results\v4_1_batch\)
REM   3. train the batch models                        (models_v4_1_batch\)
REM   4. only if 1 and 3 succeeded: set intraday.mode = batch in config_v41.yaml
REM The gate-mode models, journal and reports are left untouched (switch back = set mode: gate).
REM Takes about 1 hour. The PC must stay awake.
REM ============================================================================
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
set LOG=logs\v41_batch_setup.log
set COPY=%CD%\Nurex_V4_2\data\nurex42.setup_copy.duckdb
echo ==== %DATE% %TIME% batch-mode setup start ==== >> %LOG%

echo [0/4] copying the data store (waits for a quiet moment between live cycles)...
"%PY%" scripts_v41\copy_store_v41.py "%COPY%" >> %LOG% 2>&1
if errorlevel 1 (
  echo Could not copy the store - nothing switched. See %LOG%
  exit /b 1
)

echo [1/4] leak test (batch schedule)...
"%PY%" train_v4_1.py --db "%COPY%" --mode batch leaktest --samples 6 >> %LOG% 2>&1
if errorlevel 1 (
  echo LEAK TEST FAILED - nothing switched. See %LOG%
  echo ==== leak test FAILED ==== >> %LOG%
  exit /b 1
)

echo [2/4] walk-forward replay DK1 + DK2 at the batch schedule (about 50 min)...
"%PY%" train_v4_1.py --db "%COPY%" --mode batch replay >> %LOG% 2>&1
if errorlevel 1 echo WARNING: replay failed - continuing with training. See %LOG%

echo [3/4] training batch models...
"%PY%" train_v4_1.py --db "%COPY%" --mode batch train >> %LOG% 2>&1
if errorlevel 1 (
  echo TRAINING FAILED - nothing switched. See %LOG%
  echo ==== train FAILED ==== >> %LOG%
  exit /b 1
)
if not exist models_v4_1_batch\v4_1_batch_DK1.pkl (
  echo Batch model file missing - nothing switched.
  exit /b 1
)

echo [4/4] switching config_v41.yaml to intraday.mode: batch ...
"%PY%" -c "import re,pathlib;p=pathlib.Path('v4_1_intraday/config_v41.yaml');s=p.read_text(encoding='utf-8');s2=re.sub(r'(?m)^(\s*mode:\s*)gate\b',r'\1batch',s,count=1);p.write_text(s2,encoding='utf-8');print('mode: batch' if s2!=s else 'mode line not changed (already batch?)')" >> %LOG% 2>&1
del /q "%COPY%" "%COPY%.wal" 2>nul
echo ==== %DATE% %TIME% batch-mode setup done ==== >> %LOG%
echo.
echo DONE. From the next 15-minute cycle the journal locks hourly batches
echo   journal : data\v41_batch_journal.sqlite
echo   files   : results\v4_1_batch\batches\DK1 and DK2 (one CSV + JSON per hour)
echo   report  : results\v4_1_batch\replay_*.md
echo Restart the V4.1 dashboard so it picks up the new mode.
