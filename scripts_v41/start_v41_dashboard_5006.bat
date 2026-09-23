@echo off
REM Nurex V4.1-only dashboard: stop the retired V4.2 dashboard and any old copy, then start on port 5006.
cd /d "%~dp0.."
if not exist logs mkdir logs
set LOG=logs\v41_live_dashboard_start.log
echo ==== START %DATE% %TIME% > %LOG%
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*nurex42*dashboard.py*' -or $_.CommandLine -like '*run.py dashboard*' -or $_.CommandLine -like '*dashboard_v41_live.py*' } | ForEach-Object { 'stopping {0} {1}' -f $_.ProcessId, $_.CommandLine; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >> %LOG% 2>&1
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":5006 .*LISTENING"') do (
    echo killing port-5006 PID %%p >> %LOG%
    taskkill /F /T /PID %%p >> %LOG% 2>&1
)
timeout /t 3 /nobreak > nul
start "Nurex V4.1 Live Dashboard (5006)" cmd /k "Nurex_V4_2\.venv\Scripts\streamlit.exe run dashboard_v41_live.py --server.port 5006 --server.headless true"
timeout /t 20 /nobreak > nul
echo -- port 5006 after start: >> %LOG%
netstat -ano | findstr ":5006" >> %LOG%
echo ==== DONE %TIME% >> %LOG%
