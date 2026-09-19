@echo off
REM Nurex V4.1 dashboard - installs the legacy V2/V3.1 ML dependencies that dashboard_v4_1.py
REM imports at module level (tournament comparison tabs: xgboost, catboost, torch, optuna).
REM These are declared in the repo's root requirements.txt but were never installed into this venv.
REM Output: logs\v41_dashboard_deps.log   (this can take several minutes - torch is a large download)
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
set LOG=logs\v41_dashboard_deps.log
echo ==== START %DATE% %TIME% > %LOG%
"%PY%" -m pip install xgboost catboost torch optuna >> %LOG% 2>&1
echo ==== DONE %DATE% %TIME% >> %LOG%
echo Done. Check %LOG% for errors, then run scripts_v41\run_dashboard_v41.bat again.
