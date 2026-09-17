@echo off
REM Nurex V4.1 - paper-trading cycle every 15 minutes:
REM   EDS update + ENTSO-E/UMM/weather/frequency + lock upcoming gate closures + settle.
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
"%PY%" train_v4_1.py cycle >> logs\v41_cycle.log 2>&1
