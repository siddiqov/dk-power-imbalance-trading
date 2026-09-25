@echo off
echo ============================================================
echo  NUREX BUY-FIX RETRAIN — DK1 + DK2
echo ============================================================

cd /d "%~dp0"

echo.
echo [1/2] Training DK1...
python train_v4_1.py train --area DK1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: DK1 training failed with code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] Training DK2...
python train_v4_1.py train --area DK2
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: DK2 training failed with code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo  RETRAIN COMPLETE — BUY fix active. Restart dashboard.
echo ============================================================
pause
