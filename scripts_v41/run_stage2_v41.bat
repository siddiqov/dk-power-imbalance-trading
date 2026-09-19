@echo off
REM Nurex V4.1 - stage 2: leakage test, walk-forward replay with all new sources, retrain.
REM Collection is already complete (see logs\v41_first_run.log), so nothing is downloaded again.
REM Output: logs\v41_stage2.log   (the console stays blank on purpose - watch the log)
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
set LOG=logs\v41_stage2.log
echo ==== START %DATE% %TIME% > %LOG%
echo ==== LEAKTEST >> %LOG%
"%PY%" train_v4_1.py leaktest --samples 6 >> %LOG% 2>&1
echo ==== TESTS >> %LOG%
"%PY%" -m pytest tests_v41 -q >> %LOG% 2>&1
echo ==== REPLAY >> %LOG%
"%PY%" train_v4_1.py replay >> %LOG% 2>&1
echo ==== TRAIN >> %LOG%
"%PY%" train_v4_1.py train >> %LOG% 2>&1
echo ==== SOURCES >> %LOG%
"%PY%" train_v4_1.py sources >> %LOG% 2>&1
echo ==== DONE %DATE% %TIME% >> %LOG%
