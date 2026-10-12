@echo off
REM ============================================================================
REM Nurex V4.1 - DK model improvement research, ONE GO.  RUN ON THE DENMARK PC ONLY.
REM   Plan: "Nurex V4.1 - DK Model Improvement Plan" (phases 0, 1 and 2).
REM   * Research only: works on a private copy of the store. It does NOT change the live
REM     models, config_v41.yaml, the journals or the scheduled tasks. Live trading continues.
REM   * Runs at LOW priority so the 15-min cycle and the lock guard keep priority.
REM   * Takes about 8-12 hours (16 walk-forward replays). Keep the PC on. Closing this window
REM     stops it; running this file again RESUMES (finished replays are skipped).
REM   * Results: results\research_dk\REPORT.md  (+ week_review.md, summary.csv, best.json)
REM     Log:     results\research_dk\research.log
REM ============================================================================
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
set OUT=results\research_dk
set COPY=%CD%\Nurex_V4_2\data\nurex42.research_dk_copy.duckdb
if not exist %OUT% mkdir %OUT%
echo ==== %DATE% %TIME% DK research start ====
echo ==== %DATE% %TIME% DK research start ==== >> logs\v41_research_dk.log

if exist "%COPY%" (
  echo Using the existing store copy ^(resume^). Delete %COPY% to start from fresh data.
) else (
  echo Copying the store ^(waits for a quiet moment, max ~10 min^)...
  "%PY%" scripts_v41\copy_store_v41.py "%COPY%"
  if errorlevel 1 (
    echo FAILED to copy the store - try again in a few minutes.
    if not "%NUREX_UNATTENDED%"=="1" pause
    exit /b 1
  )
)

start "" /b /low /wait "%PY%" scripts_v41\research_dk_v41.py --db "%COPY%" --out %OUT%
set RC=%ERRORLEVEL%
echo ==== %DATE% %TIME% DK research finished, exit code %RC% ==== >> logs\v41_research_dk.log
echo.
if "%RC%"=="0" (
  echo DONE. All replays finished. Removing the store copy.
  del /q "%COPY%" "%COPY%.wal" 2>nul
) else (
  echo Finished with exit code %RC% - see %OUT%\research.log. The store copy is kept so a re-run resumes.
)
echo Report: %CD%\%OUT%\REPORT.md
echo Send me REPORT.md and week_review.md ^(or paste them^).
if not "%NUREX_UNATTENDED%"=="1" pause
