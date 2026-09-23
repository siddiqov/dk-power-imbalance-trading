@echo off
REM Nurex V4.1 - 22 Sep fixes: leak test + unit tests, then restart the V4.1 dashboard (port 5005)
REM Output: logs\v41_fix_0922.log
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
set LOG=logs\v41_fix_0922.log
echo ==== FIX VERIFY START %DATE% %TIME% > %LOG%
echo Python: %PY% >> %LOG%

echo ==== STEP 1: STOP DASHBOARD ON PORT 5005 >> %LOG%
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":5005 .*LISTENING"') do (
    echo killing PID %%p >> %LOG%
    taskkill /F /T /PID %%p >> %LOG% 2>&1
)

echo ==== STEP 2: COMPILE CHECK >> %LOG%
"%PY%" -m py_compile dashboard_v4_1.py v4_1_intraday\dashboard_adapter.py v4_1_intraday\pipeline_id.py src\tournament_tables_v2.py src\commercial_strategy_v3_1.py >> %LOG% 2>&1
echo compile exit %ERRORLEVEL% >> %LOG%

echo ==== STEP 3: LEAKAGE TEST >> %LOG%
"%PY%" train_v4_1.py leaktest --samples 6 >> %LOG% 2>&1
echo leaktest exit %ERRORLEVEL% >> %LOG%

echo ==== STEP 4: UNIT TESTS >> %LOG%
"%PY%" -m pytest tests_v41 -q -p no:cacheprovider >> %LOG% 2>&1
echo pytest exit %ERRORLEVEL% >> %LOG%

echo ==== STEP 5: START DASHBOARD >> %LOG%
start "Nurex V4.1 Dashboard (5005)" cmd /k scripts_v41\run_dashboard_v41.bat
echo ==== FIX VERIFY DONE %DATE% %TIME% >> %LOG%
