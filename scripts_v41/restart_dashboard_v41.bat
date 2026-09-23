@echo off
REM Nurex V4.1 - stop every running copy of dashboard_v4_1.py and start one fresh (port 5005)
REM Output: logs\v41_dashboard_restart.log
cd /d "%~dp0.."
if not exist logs mkdir logs
set LOG=logs\v41_dashboard_restart.log
echo ==== RESTART %DATE% %TIME% > %LOG%
echo -- processes running dashboard_v4_1.py before: >> %LOG%
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*dashboard_v4_1.py*' } | ForEach-Object { '{0} {1}' -f $_.ProcessId, $_.CommandLine; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >> %LOG% 2>&1
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":5005 .*LISTENING"') do (
    echo killing port-5005 PID %%p >> %LOG%
    taskkill /F /T /PID %%p >> %LOG% 2>&1
)
timeout /t 3 /nobreak > nul
echo -- starting dashboard >> %LOG%
start "Nurex V4.1 Dashboard (5005)" cmd /k "Nurex_V4_2\.venv\Scripts\streamlit.exe run dashboard_v4_1.py --server.port 5005 --server.headless true"
timeout /t 25 /nobreak > nul
echo -- port 5005 after start: >> %LOG%
netstat -ano | findstr ":5005" >> %LOG%
echo ==== DONE %TIME% >> %LOG%
