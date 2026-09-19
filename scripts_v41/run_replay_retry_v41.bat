@echo off
REM Nurex V4.1 - fix: pandas.to_markdown needs the 'tabulate' package, missing on this venv.
REM Installs it, then reruns the walk-forward replay report only (leaktest/tests/train already passed).
REM Output: logs\v41_replay_retry.log   (console stays blank on purpose - watch the log)
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
set LOG=logs\v41_replay_retry.log
echo ==== START %DATE% %TIME% > %LOG%
echo ==== INSTALL TABULATE >> %LOG%
"%PY%" -m pip install tabulate>=0.9 >> %LOG% 2>&1
echo ==== REPLAY >> %LOG%
"%PY%" train_v4_1.py replay >> %LOG% 2>&1
echo ==== DONE %DATE% %TIME% >> %LOG%
