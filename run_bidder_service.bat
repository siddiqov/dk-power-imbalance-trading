@echo off
title 24/7 Power Trading Bidder Publisher Service
echo ======================================================================
echo           Starting Power Trading 24/7 Bidder Publisher
echo ======================================================================
echo.

cd /d "%~dp0"

echo Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

echo Checking dependencies...
pip install -r requirements.txt --quiet

echo.
echo Starting Auto Publisher daemon (updates every 15 minutes)...
echo Press Ctrl+C to stop.
echo.
python scripts/auto_publisher.py

pause
