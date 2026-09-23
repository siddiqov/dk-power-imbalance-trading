@echo off
title Nurex V4.1 Dashboard (port 5005)
echo Stopping old dashboard on port 5006 (run.py dashboard, wrong working folder)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5006 " ^| findstr LISTENING') do taskkill /F /PID %%a
echo Stopping any previous dashboard on port 5005...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5005 " ^| findstr LISTENING') do taskkill /F /PID %%a
cd /d "%~dp0"
echo Working folder: %CD%
"Nurex_V4_2\.venv\Scripts\python.exe" start_dashboard_v4_1.py
pause
