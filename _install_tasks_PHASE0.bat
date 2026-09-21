@echo off
:: Phase 0 - Install Nurex V4.1 Scheduled Tasks
:: DOUBLE-CLICK this file to run as administrator.
:: It will auto-elevate via UAC prompt.

net session >nul 2>&1
if %errorLevel% == 0 (
    goto :run_as_admin
) else (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:run_as_admin
cd /d "%~dp0"
echo ==========================================
echo  Nurex V4.1 - Phase 0: Install Tasks
echo ==========================================

:: Install the three tasks
powershell -ExecutionPolicy Bypass -File "scripts_v41\install_tasks_v41.ps1"

:: Also run decompiler if Python 3.14 venv exists
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if exist "%PY%" (
    echo.
    echo ==========================================
    echo  Phase 0.2: Extracting .pyc stubs
    echo ==========================================
    "%PY%" _decompile_phase0.py
) else (
    echo [SKIP] Python venv not found at %PY%
)

:: Query tasks to verify
echo.
echo ==========================================
echo  Verification: Task Scheduler
echo ==========================================
schtasks /query /fo LIST /tn "Nurex_V41_Cycle" 2>nul || echo [ERROR] Nurex_V41_Cycle not found
schtasks /query /fo LIST /tn "Nurex_V41_Recorder" 2>nul || echo [ERROR] Nurex_V41_Recorder not found
schtasks /query /fo LIST /tn "Nurex_V41_Train" 2>nul || echo [ERROR] Nurex_V41_Train not found

echo.
echo ==========================================
echo  Phase 0 Complete!
echo  Check logs above for any errors.
echo ==========================================
pause
