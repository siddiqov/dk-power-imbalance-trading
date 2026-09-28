@echo off
REM ============================================================================
REM Nurex V4.1 - test ONE change: today's model + the 16 mFRR activation-volume inputs.
REM Everything else exactly as the live model (no Huber, no guards, no fallback BUY), so the
REM result is directly comparable with results\v4_1_batch\replay_*_20260927_2156.md.
REM Live models, live config and paper trading are not touched.
REM Output: results\v4_1_batch_test_mfrr\   Log: logs\v41_mfrr_test.log   (about 2 hours)
REM ============================================================================
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
set LOG=logs\v41_mfrr_test.log
set COPY=%CD%\Nurex_V4_2\data\nurex42.test_copy.duckdb
set OVR=--set model.enhanced=false --set model.mfrr_volume_features=true --set decision.fallback_buy.enabled=false
echo ==== %DATE% %TIME% mFRR test start ==== >> %LOG%
"%PY%" scripts_v41\copy_store_v41.py "%COPY%" >> %LOG% 2>&1
if errorlevel 1 exit /b 1
"%PY%" train_v4_1.py --db "%COPY%" --mode batch %OVR% leaktest --samples 6 >> %LOG% 2>&1
if errorlevel 1 (
  echo ==== leak test FAILED - stopped ==== >> %LOG%
  del /q "%COPY%" "%COPY%.wal" 2>nul
  exit /b 1
)
"%PY%" train_v4_1.py --db "%COPY%" --mode batch %OVR% replay --out results\v4_1_batch_test_mfrr >> %LOG% 2>&1
del /q "%COPY%" "%COPY%.wal" 2>nul
echo ==== %DATE% %TIME% mFRR test done ==== >> %LOG%
