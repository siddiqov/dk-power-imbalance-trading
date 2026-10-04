@echo off
REM Nurex V4.1 research, stage 2: forecast-drift features, 6 alternative models, targeted
REM feature-group ablation (groups flagged by permutation importance), LSTM + Transformer.
REM Uses the cached dataset only. Log: results\research_v41\run.log
cd /d "%~dp0.."
set PY=%CD%\Nurex_V4_2\.venv\Scripts\python.exe
set LOG=%CD%\results\research_v41\run.log
echo ==== %DATE% %TIME% stage 2 start ==== >> "%LOG%"
cd research_v41
"%PY%" rs_run.py drift >> "%LOG%" 2>&1
"%PY%" rs_run.py ablation --areas=DK1 --groups=outages_umm,system_state,live_grid,calendar >> "%LOG%" 2>&1
"%PY%" rs_run.py ablation --areas=DK2 --groups=other_zone_state,frequency,live_grid >> "%LOG%" 2>&1
"%PY%" rs_run.py models >> "%LOG%" 2>&1
"%PY%" rs_run.py seq >> "%LOG%" 2>&1
echo ==== %DATE% %TIME% stage 2 done ==== >> "%LOG%"
