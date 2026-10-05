@echo off
REM Crash-guard test (2026-10-05) - RESEARCH ONLY: research copy of the data, nothing live changes.
REM Replays DK1 logistic, DK1 LightGBM, DK2 LightGBM (~2 h) and scores each with the BUY crash
REM guard at -30 (live), off, -60, -100, -150.  Results: results\research_v41\guard\guard_results.csv
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
echo ==== %DATE% %TIME% crash-guard test ==== >> logs\v41_guard_test.log
"%PY%" research_v41\rs_guard.py >> logs\v41_guard_test.log 2>&1
echo ==== %DATE% %TIME% crash-guard test finished (exit %ERRORLEVEL%) ==== >> logs\v41_guard_test.log
