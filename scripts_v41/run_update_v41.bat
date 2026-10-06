@echo off
REM Nurex V4.1 - paper-trading cycle every 15 minutes:
REM   EDS update + lock + ENTSO-E/UMM/weather/frequency + lock again + settle.
REM 2026-10-06: if Python ends abnormally (e.g. a crash), log the exit code and retry the lock at once,
REM so a crash can never cost a batch (the lock guard at :37 is a second safety net).
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
"%PY%" train_v4_1.py cycle >> logs\v41_cycle.log 2>&1
set RC=%ERRORLEVEL%
if "%RC%"=="0" goto :eof
echo %DATE% %TIME% WARNING: cycle ended with exit code %RC% - retrying the lock >> logs\v41_cycle.log
"%PY%" train_v4_1.py lock >> logs\v41_cycle.log 2>&1
if errorlevel 1 (
  timeout /t 20 /nobreak > nul
  "%PY%" train_v4_1.py lock >> logs\v41_cycle.log 2>&1
)
