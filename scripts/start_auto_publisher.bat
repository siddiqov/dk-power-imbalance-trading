@echo off
cd /d "c:\Users\Hafeez\Documents\Nurex_Trading\Basic_Approach"
set PYTHONUNBUFFERED=1
set PYTHONPATH=c:\Users\Hafeez\Documents\Nurex_Trading\Basic_Approach

:loop
"C:\Python314\python.exe" -u "scripts\auto_publisher.py"
timeout /t 10 /nobreak >nul
goto loop
