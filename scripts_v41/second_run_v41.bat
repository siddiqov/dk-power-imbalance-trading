@echo off
REM Nurex V4.1 - second run: full outage history (all versions), order book check, ENTSO-E key check,
REM leakage test, walk-forward replay with the new sources, retrain. Output: logs\v41_second_run.log
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
set LOG=logs\v41_second_run.log
echo ==== START %DATE% %TIME% > %LOG%
echo ==== UMM FULL HISTORY (all versions) >> %LOG%
"%PY%" train_v4_1.py collect --source umm --start 2025-03-04 >> %LOG% 2>&1
echo ==== ENTSOE >> %LOG%
"%PY%" train_v4_1.py collect --source entsoe >> %LOG% 2>&1
echo ==== PROBE NORDPOOL (streaming + order book) >> %LOG%
"%PY%" train_v4_1.py probe-nordpool --seconds 60 >> %LOG% 2>&1
echo ==== SOURCES >> %LOG%
"%PY%" train_v4_1.py sources >> %LOG% 2>&1
echo ==== LEAKTEST >> %LOG%
"%PY%" train_v4_1.py leaktest --samples 3 >> %LOG% 2>&1
echo ==== REPLAY >> %LOG%
"%PY%" train_v4_1.py replay >> %LOG% 2>&1
echo ==== TRAIN >> %LOG%
"%PY%" train_v4_1.py train >> %LOG% 2>&1
echo ==== DONE %DATE% %TIME% >> %LOG%
