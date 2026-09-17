@echo off
REM Nurex V4.1 - weekly retraining (new model is used from the next locked decision on).
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
"%PY%" train_v4_1.py train >> logs\v41_train.log 2>&1
