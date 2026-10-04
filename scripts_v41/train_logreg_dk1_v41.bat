@echo off
REM ============================================================================
REM Nurex V4.1 - logistic regression for DK1 (2026-10-04)
REM   1. trains the DK1 logistic model -> models_v4_1_batch\v4_1_batch_logreg_DK1.pkl
REM      (12 settings tried on the 90-day validation window, best one kept; ~10-20 min)
REM   2. verification replay of the DK1 logistic model on the research copy of the data
REM      (walk-forward Jul 2025 - Sep 2026; research result to match: about EUR 69k)
REM      -> results\v4_1_batch_eval\logreg\  (~1-2 h). Live reports are not touched.
REM Nothing is sent to the client by this script. Log: logs\v41_logreg.log
REM ============================================================================
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
set LOG=logs\v41_logreg.log
echo ==== %DATE% %TIME% train DK1 logreg ==== >> %LOG%
"%PY%" train_v4_1.py --mode batch train --area DK1 --family logreg >> %LOG% 2>&1
echo ==== %DATE% %TIME% train done (exit %ERRORLEVEL%) ==== >> %LOG%
echo ==== %DATE% %TIME% replay DK1 logreg (research copy) ==== >> %LOG%
"%PY%" train_v4_1.py --mode batch --db "%CD%\Nurex_V4_2\data\nurex42.research_copy.duckdb" replay --area DK1 --family logreg --out results\v4_1_batch_eval\logreg >> %LOG% 2>&1
echo ==== %DATE% %TIME% replay done (exit %ERRORLEVEL%) ==== >> %LOG%
