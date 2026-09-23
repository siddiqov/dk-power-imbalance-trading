@echo off
echo Running diagnostic...
C:\Python314\python.exe "C:\Users\Hafeez\Documents\Nurex_Trading\Basic_Approach\test_diag.py" > "C:\Users\Hafeez\Documents\Nurex_Trading\Basic_Approach\logs\test_diag_out.txt" 2>&1
echo Done. Exit code: %ERRORLEVEL%
pause
