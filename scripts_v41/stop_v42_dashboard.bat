@echo off
REM Stops the retired Nurex V4.2 dashboard (nurex42\dashboard.py on port 5006). Touches nothing else.
cd /d "%~dp0.."
set LOG=logs\v42_dashboard_stop.log
echo ==== STOP V4.2 DASHBOARD %DATE% %TIME% > %LOG%
echo -- port 5006 before: >> %LOG%
netstat -ano | findstr ":5006" >> %LOG%
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*nurex42*dashboard.py*' -or $_.CommandLine -like '*run.py dashboard*' } | ForEach-Object { 'stopping {0} {1}' -f $_.ProcessId, $_.CommandLine; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >> %LOG% 2>&1
timeout /t 3 /nobreak > nul
echo -- port 5006 after: >> %LOG%
netstat -ano | findstr ":5006" >> %LOG%
echo -- V4.2 live process (left running): >> %LOG%
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run.py live*' } | ForEach-Object { '{0} {1}' -f $_.ProcessId, $_.CommandLine }" >> %LOG% 2>&1
echo ==== DONE >> %LOG%
