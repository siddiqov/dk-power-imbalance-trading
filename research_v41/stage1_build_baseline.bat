@echo off
REM Nurex V4.1 research, stage 1: private store copy, dataset build, baseline + permutation importance.
REM Does NOT touch live code, models, journal or scheduled tasks. Log: results\research_v41\run.log
cd /d "%~dp0.."
set PY=%CD%\Nurex_V4_2\.venv\Scripts\python.exe
if not exist results\research_v41 mkdir results\research_v41
set LOG=%CD%\results\research_v41\run.log
echo ==== %DATE% %TIME% stage 1 start ==== >> "%LOG%"
"%PY%" scripts_v41\copy_store_v41.py "%CD%\Nurex_V4_2\data\nurex42.research_copy.duckdb" >> "%LOG%" 2>&1
if errorlevel 1 goto :end
cd research_v41
"%PY%" rs_build.py >> "%LOG%" 2>&1
"%PY%" rs_run.py baseline >> "%LOG%" 2>&1
:end
echo ==== %DATE% %TIME% stage 1 done ==== >> "%LOG%"
