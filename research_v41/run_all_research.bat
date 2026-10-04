@echo off
REM Nurex V4.1 research - ALL stages in one go (about 8-12 hours; keep the PC plugged in and awake).
REM Works on a private copy of the data store. Does NOT touch live code, models, journal or tasks.
REM Progress: results\research_v41\research.log   Results table: results\research_v41\summary.csv
cd /d "%~dp0.."
set PY=%CD%\Nurex_V4_2\.venv\Scripts\python.exe
if not exist results\research_v41 mkdir results\research_v41
set LOG=%CD%\results\research_v41\run.log
echo ==== %DATE% %TIME% research start ==== >> "%LOG%"
"%PY%" scripts_v41\copy_store_v41.py "%CD%\Nurex_V4_2\data\nurex42.research_copy.duckdb" >> "%LOG%" 2>&1
if errorlevel 1 goto :end
cd research_v41
"%PY%" rs_build.py >> "%LOG%" 2>&1
"%PY%" rs_run.py baseline drift >> "%LOG%" 2>&1
"%PY%" rs_run.py models >> "%LOG%" 2>&1
"%PY%" rs_run.py ablation >> "%LOG%" 2>&1
"%PY%" rs_run.py seq >> "%LOG%" 2>&1
:end
echo ==== %DATE% %TIME% research done ==== >> "%LOG%"
