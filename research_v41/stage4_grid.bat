@echo off
cd /d "%~dp0.."
set PY=%CD%\Nurex_V4_2\.venv\Scripts\python.exe
set LOG=%CD%\results\research_v41\stage4_main.log
echo ==== %DATE% %TIME% stage 4 start ==== >> "%LOG%"
cd research_v41
"%PY%" rs_grid.py all >> "%LOG%" 2>&1
echo ==== %DATE% %TIME% stage 4 finished ==== >> "%LOG%"
