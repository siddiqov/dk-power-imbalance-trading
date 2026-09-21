@echo off
REM Nurex V4.1 Intraday - NEW engine only, live/honest view (no legacy High-Alpha tabs)
REM http://127.0.0.1:5006/
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
"%PY%" -m streamlit run dashboard_v41_live.py --server.port 5006
