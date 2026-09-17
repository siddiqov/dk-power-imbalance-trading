@echo off
REM Nurex V4.1 - Nord Pool intraday market data recorder (runs until stopped, reconnects by itself).
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
"%PY%" train_v4_1.py record-intraday >> logs\v41_recorder.log 2>&1
