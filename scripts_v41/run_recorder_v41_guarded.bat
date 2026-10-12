@echo off
REM Nurex V4.1 - Nord Pool recorder (guarded, self-looping, rotating log).
REM This bat loops so the process auto-restarts after unexpected exit.
REM Task Scheduler should use "Do not start a new instance" if already running.
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
if not exist Nurex_V4_2\runtime mkdir Nurex_V4_2\runtime

:loop
echo %DATE% %TIME% Starting Nord Pool recorder...
"%PY%" train_v4_1.py record-intraday
echo %DATE% %TIME% Recorder exited with code %ERRORLEVEL% - restarting in 5s...
timeout /t 5 /nobreak >nul
goto loop
