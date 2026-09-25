@echo off
REM ===========================================================
REM  START_NUREX.bat - starts the LIVE collector and the V4.1
REM  dashboard (port 5005) in their own windows.
REM  Safe to run again: it stops the old ones first.
REM  Keep the two windows it opens running.
REM ===========================================================
setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PY=%ROOT%\Nurex_V4_2\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo WARNING: .venv Python not found - falling back to PATH python.
  set "PY=python"
)

echo [1/4] Stopping old Nurex windows...
taskkill /F /FI "WINDOWTITLE eq Nurex LIVE*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Nurex Dashboard 5005*" >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5005 " ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
timeout /t 3 /nobreak >nul

echo [2/4] Starting LIVE collector (paper trading + settlement)...
start "Nurex LIVE" cmd /k "cd /d "%ROOT%\Nurex_V4_2" && "%PY%" run.py live"
timeout /t 5 /nobreak >nul

echo [3/4] Starting dashboard on http://127.0.0.1:5005/ ...
start "Nurex Dashboard 5005" cmd /k "cd /d "%ROOT%" && "%PY%" -m streamlit run dashboard_v4_1.py --server.port 5005 --server.headless true"

echo [4/4] Waiting for the dashboard to come up, then opening the browser...
timeout /t 25 /nobreak >nul
start "" http://127.0.0.1:5005/

echo.
echo Done. Two windows are now running:
echo   "Nurex LIVE"             - locks decisions and settles PnL every 5 min
echo   "Nurex Dashboard 5005"   - the dashboard itself
echo Closing either one stops that part. This window can be closed.
echo.
pause
