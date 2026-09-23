@echo off
echo === Nurex V4.2 - Full Restart ===
echo.
echo Step 1: Killing all Python processes on port 5006...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5006 "') do (
    taskkill /F /PID %%a 2>nul
)
echo Step 2: Killing PID 20580 (duckdb lock holder)...
taskkill /F /PID 20580 2>nul
echo Step 3: Killing any remaining Python processes holding duckdb...
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST 2^>nul ^| findstr "PID"') do (
    taskkill /F /PID %%a 2>nul
)
echo Step 4: Waiting 4 seconds...
timeout /t 4 /nobreak >nul
echo Step 5: Starting LIVE data collector...
cd /d "C:\Users\Hafeez\Documents\Nurex_Trading\Basic_Approach\Nurex_V4_2"
start "Nurex LIVE" cmd /k "python run.py live"
echo Step 6: Waiting 3 seconds for live to initialise...
timeout /t 3 /nobreak >nul
echo Step 7: Starting DASHBOARD...
start "Nurex Dashboard" cmd /k "python run.py dashboard"
echo.
echo Both processes started! Dashboard at http://localhost:5006
pause
