@echo off
REM Nurex V4.1 lock guard - see lock_guard_v41.py. Scheduled hourly at :37 (Nurex_V41_LockGuard).
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
"%PY%" scripts_v41\lock_guard_v41.py >> logs\v41_lockguard_console.log 2>&1
