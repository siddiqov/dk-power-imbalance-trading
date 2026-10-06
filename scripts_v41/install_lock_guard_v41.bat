@echo off
REM Nurex V4.1 (2026-10-06): installs the hourly lock guard (runs at :37, hidden window) and hides
REM the cycle/train task windows. Safe to run more than once.
set H=%~dp0
schtasks /Create /F /TN "Nurex_V41_LockGuard" /SC HOURLY /ST 00:37 /TR "wscript.exe \"%H%run_hidden_v41.vbs\" run_lock_guard_v41.bat"
schtasks /Change /TN "Nurex_V41_Cycle" /TR "wscript.exe \"%H%run_hidden_v41.vbs\" run_update_v41.bat"
schtasks /Change /TN "Nurex_V41_Train" /TR "wscript.exe \"%H%run_hidden_v41.vbs\" run_train_v41.bat"
echo.
echo Check:
schtasks /Query /TN "Nurex_V41_LockGuard" /V /FO LIST | findstr /C:"Task To Run" /C:"Next Run Time" /C:"Status"
schtasks /Query /TN "Nurex_V41_Cycle" /V /FO LIST | findstr /C:"Task To Run" /C:"Status"
echo.
echo Test the guard once now (it only locks if a batch is actually missing):
call "%H%run_lock_guard_v41.bat"
type "%H%..\logs\v41_lockguard.log"
echo.
pause
