@echo off
REM Nurex V4.1 (2026-10-05): stop the black console window popping up every 15 minutes.
REM Points the Nurex_V41_Cycle and Nurex_V41_Train scheduled tasks at run_hidden_v41.vbs, which runs
REM the same .bat files with no window. Schedules, logs and trading are unchanged.
REM Undo: run install_tasks_v41.ps1 again (it recreates the tasks with the plain .bat files).
set VBS=%~dp0run_hidden_v41.vbs
schtasks /Change /TN "Nurex_V41_Cycle" /TR "wscript.exe \"%VBS%\" run_update_v41.bat"
schtasks /Change /TN "Nurex_V41_Train" /TR "wscript.exe \"%VBS%\" run_train_v41.bat"
echo.
echo Check (the "Task To Run" lines should now start with wscript.exe):
schtasks /Query /TN "Nurex_V41_Cycle" /V /FO LIST | findstr /C:"Task To Run" /C:"Status"
schtasks /Query /TN "Nurex_V41_Train" /V /FO LIST | findstr /C:"Task To Run" /C:"Status"
echo.
pause
