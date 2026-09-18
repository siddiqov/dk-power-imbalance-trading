@echo off
REM Nurex V4.1 - one-shot: install deps, register scheduled tasks, first data cycle, start dashboard.
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs

echo [1/4] Installing missing packages (xgboost, catboost, optuna, torch CPU)...
"%PY%" -m pip install --quiet xgboost catboost optuna || goto :failed
"%PY%" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu || goto :failed

echo [2/4] Registering scheduled tasks (Nurex_V41_Update every 15 min, Nurex_V41_Recorder at logon)...
powershell -ExecutionPolicy Bypass -File "scripts_v41\install_tasks_v41.ps1" || goto :failed

echo [3/4] Running one data cycle now (log: logs\v41_update.log)...
"%PY%" train_v4_1.py update  >> logs\v41_update.log 2>&1
"%PY%" train_v4_1.py collect >> logs\v41_update.log 2>&1

echo [4/4] Starting the V4.1 dashboard on http://127.0.0.1:5005/ (Ctrl+C to stop)...
"%PY%" -m streamlit run dashboard_v4_1.py --server.port 5005
goto :eof

:failed
echo.
echo Setup step failed - see the message above. Dashboard not started.
pause
