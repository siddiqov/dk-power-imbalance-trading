@echo off
REM Read-only check of the V4.1 models (recommended + comparison) - 2026-10-04
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
"%PY%" check_models_v41.py %1 > logs\v41_model_check.log 2>&1
type logs\v41_model_check.log | findstr /V /C:"Warning" /C:"warnings.warn"
echo.
echo Full output: logs\v41_model_check.log
pause
