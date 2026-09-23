@echo off
:: Nurex V4.1 — Install Phase 7 Shadow Monitor Task Scheduler entry
:: Run ONCE from an elevated (Administrator) command prompt.
:: Creates: Nurex_V41_ShadowMonitor — daily at 23:45 local time

setlocal
cd /d "%~dp0.."
set "BASE=%CD%"
set "TASK=Nurex_V41_ShadowMonitor"
set "BAT=%BASE%\scripts_v41\run_phase7_shadow_daily.bat"

echo Installing scheduled task: %TASK%
echo Script: %BAT%
echo.

schtasks /delete /tn "%TASK%" /f >nul 2>&1

schtasks /create ^
  /tn "%TASK%" ^
  /tr "\"%BAT%\"" ^
  /sc DAILY ^
  /st 23:45 ^
  /ru "%USERNAME%" ^
  /f

if %errorlevel% equ 0 (
    echo.
    echo SUCCESS: Task "%TASK%" installed.
    echo It will run daily at 23:45 and write to logs\v41_shadow_daily.log
    echo.
    echo To verify:   schtasks /query /tn "%TASK%" /v /fo list
    echo To run now:  schtasks /run  /tn "%TASK%"
) else (
    echo.
    echo FAILED — run this script as Administrator.
)
endlocal
