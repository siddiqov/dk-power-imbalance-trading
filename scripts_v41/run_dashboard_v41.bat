@echo off
REM Nurex V4.1 - start the dashboard on http://127.0.0.1:5005/
cd /d "%~dp0.."
set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
"%PY%" -m streamlit run dashboard_v4_1.py --server.port 5005
