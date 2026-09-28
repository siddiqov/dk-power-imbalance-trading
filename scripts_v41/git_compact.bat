@echo off
REM Compact the git repository (git gc): packs loose objects, keeps all history.
REM Log: logs\git_gc.log
cd /d "%~dp0.."
if not exist logs mkdir logs
set LOG=logs\git_gc.log
echo ==== %DATE% %TIME% git gc start ==== >> %LOG%
where git >> %LOG% 2>&1
if errorlevel 1 (
  echo git.exe not found on PATH >> %LOG%
  exit /b 1
)
git count-objects -vH >> %LOG% 2>&1
git gc >> %LOG% 2>&1
echo exit code %ERRORLEVEL% >> %LOG%
git count-objects -vH >> %LOG% 2>&1
echo ==== %DATE% %TIME% git gc done ==== >> %LOG%
