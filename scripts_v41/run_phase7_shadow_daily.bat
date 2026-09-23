@echo off
:: Nurex V4.1 — Phase 7 Shadow Monitor (daily check + weekly report on Sundays)
:: Installed as Windows Task Scheduler task: Nurex_V41_ShadowMonitor
:: Runs at 23:45 local time every day.

setlocal
cd /d "%~dp0.."

set "LOG=logs\v41_shadow_daily_run.log"
echo. >> "%LOG%"
echo ==== SHADOW MONITOR %DATE% %TIME% >> "%LOG%"

python shadow_monitor.py >> "%LOG%" 2>&1
if %errorlevel% neq 0 (
    echo ERROR: shadow_monitor.py exited with code %errorlevel% >> "%LOG%"
    echo ERROR: shadow_monitor.py failed - check logs\v41_shadow_daily_run.log
) else (
    echo OK >> "%LOG%"
)

echo ==== DONE %TIME% >> "%LOG%"
endlocal
