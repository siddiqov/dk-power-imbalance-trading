@echo off
REM ============================================================================
REM Nurex V4.1 - evaluate the 2026-09-28 model upgrade (does NOT touch the live models)
REM   mFRR volume features, Huber loss + recency weights, spike + flat guards, fallback BUY.
REM   1. private copy of the store (live cycle never blocked)
REM   2. leak test at the batch decision times (must PASS)
REM   3. full walk-forward replay DK1 + DK2  -> results\v4_1_batch_eval\
REM Compare with results\v4_1_batch\replay_*_20260927_2156.md (current live model).
REM Log: logs\v41_enhanced_eval.log   (takes about 2-3 hours; the PC must stay awake)
REM ============================================================================
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
set LOG=logs\v41_enhanced_eval.log
set COPY=%CD%\Nurex_V4_2\data\nurex42.eval_copy.duckdb
echo ==== %DATE% %TIME% enhanced evaluation start ==== >> %LOG%
"%PY%" scripts_v41\copy_store_v41.py "%COPY%" >> %LOG% 2>&1
if errorlevel 1 exit /b 1
"%PY%" train_v4_1.py --db "%COPY%" --mode batch leaktest --samples 6 >> %LOG% 2>&1
if errorlevel 1 (
  echo ==== leak test FAILED - stopped ==== >> %LOG%
  del /q "%COPY%" "%COPY%.wal" 2>nul
  exit /b 1
)
"%PY%" train_v4_1.py --db "%COPY%" --mode batch replay --out results\v4_1_batch_eval >> %LOG% 2>&1
del /q "%COPY%" "%COPY%.wal" 2>nul
echo ==== %DATE% %TIME% enhanced evaluation done ==== >> %LOG%
