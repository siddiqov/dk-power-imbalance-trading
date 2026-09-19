@echo off
REM Nurex V4.1 - first real run of the new data sources. Output: logs\v41_first_run.log
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
set LOG=logs\v41_first_run.log
echo ==== START %DATE% %TIME% > %LOG%
echo ==== PROBE NORDPOOL >> %LOG%
"%PY%" train_v4_1.py probe-nordpool --seconds 90 >> %LOG% 2>&1
echo ==== COLLECT UMM + WEATHER >> %LOG%
"%PY%" train_v4_1.py collect --source umm,weather >> %LOG% 2>&1
echo ==== COLLECT ENTSOE >> %LOG%
"%PY%" train_v4_1.py collect --source entsoe >> %LOG% 2>&1
echo ==== SOURCES >> %LOG%
"%PY%" train_v4_1.py sources >> %LOG% 2>&1
echo ==== LEAKTEST >> %LOG%
"%PY%" train_v4_1.py leaktest --samples 3 >> %LOG% 2>&1
echo ==== DONE %DATE% %TIME% >> %LOG%
