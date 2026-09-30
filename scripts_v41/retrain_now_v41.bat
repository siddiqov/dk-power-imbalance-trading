@echo off
REM ============================================================================
REM Nurex V4.1 - retrain the live batch models NOW with the current config_v41.yaml
REM   (enhanced: false, mfrr_volume_features: [DK2]).  2026-09-29
REM   Why: the models copied on 28 Sep were trained while the enhanced test config was on
REM   (Huber/recency/spike + mFRR volumes in BOTH zones). That set lost in the replay and
REM   DK1 now gets empty mFRR-volume inputs at prediction time.
REM   1. backs up the current models to backups\models_v4_1_batch_before_retrain
REM   2. retrains DK1 + DK2 (about 25 min). The next locked batch uses the new models.
REM Run it ONLY on the machine that trades live (Denmark). Log: logs\v41_train.log
REM ============================================================================
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
if not exist backups\models_v4_1_batch_before_retrain mkdir backups\models_v4_1_batch_before_retrain
copy /y models_v4_1_batch\*.* backups\models_v4_1_batch_before_retrain\ >nul
echo ==== %DATE% %TIME% retrain now (current config) ==== >> logs\v41_train.log
"%PY%" train_v4_1.py --mode batch train >> logs\v41_train.log 2>&1
findstr /c:"decision params" logs\v41_train.log | more
echo ==== %DATE% %TIME% retrain done ==== >> logs\v41_train.log
echo Done - see logs\v41_train.log
pause
