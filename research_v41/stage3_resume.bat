@echo off
REM Nurex V4.1 research, stage 3 (resume after the 1 Oct pause): remaining models + LSTM/Transformer.
REM Uses the cached dataset only; does not touch live trading. Log: results\research_v41\run.log
cd /d "%~dp0.."
set PY=%CD%\Nurex_V4_2\.venv\Scripts\python.exe
set LOG=%CD%\results\research_v41\run.log
echo ==== %DATE% %TIME% stage 3 start ==== >> "%LOG%"
cd research_v41
"%PY%" rs_run.py models --areas=DK1 --kinds=extratrees,logistic,mlp >> "%LOG%" 2>&1
"%PY%" rs_run.py models --areas=DK2 >> "%LOG%" 2>&1
"%PY%" rs_run.py seq >> "%LOG%" 2>&1
"%PY%" rs_run.py combo --areas=DK1 --drop=live_grid --name=dk1_drop_live_grid >> "%LOG%" 2>&1
echo ==== %DATE% %TIME% stage 3 done ==== >> "%LOG%"
