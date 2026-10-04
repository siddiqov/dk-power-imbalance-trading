@echo off
REM ============================================================================
REM Nurex V4.1 - push ALL code and settings from this PC to GitHub branch
REM v4.1-phase5-cleanup, so the Denmark machine can pull it.   (2026-09-30)
REM
REM Safe for the running system: this folder is NOT changed. No checkout, no pull,
REM no switch of branch. A snapshot of the files is made in a temporary git index
REM on top of the current GitHub branch, shown to you, and pushed only after you
REM answer Y.
REM
REM Pushed: every file git does not ignore (code, scripts, settings, tests, docs).
REM NOT pushed: .env / secrets (ignored), models (*.pkl and models_v* - go by Drive),
REM   data stores (*.db, *.duckdb, *.sqlite, *.parquet, data\, Nurex_V4_2\data\),
REM   results\, backups\, logs\, _drive_transfer\, Nurex_V4_2\batches\.
REM Log: logs\v41_push_to_github.log
REM ============================================================================
setlocal
cd /d "%~dp0.."
set ROOT=%CD%
set BRANCH=v4.1-phase5-cleanup
if not exist logs mkdir logs
set LOG=%ROOT%\logs\v41_push_to_github.log
set GIT_INDEX_FILE=%ROOT%\.git\push_snapshot.index
echo ==== %DATE% %TIME% push to GitHub (%BRANCH%) ==== > "%LOG%"

echo Fetching %BRANCH% from GitHub ...
git fetch origin %BRANCH% >> "%LOG%" 2>&1 || goto :fail
if exist "%GIT_INDEX_FILE%" del /q "%GIT_INDEX_FILE%"
git read-tree origin/%BRANCH% >> "%LOG%" 2>&1 || goto :fail

echo Collecting files (this takes a minute) ...
REM (logs\ and Nurex_V4_2\data\ are already ignored by git, so they are not listed here -
REM  naming an ignored folder makes "git add" fail)
git -c core.autocrlf=true -c core.safecrlf=false -c advice.addIgnoredFile=false add -A -- . ":(exclude)results" ":(exclude)backups" ":(exclude,glob)_backup_*/**" ":(exclude)data" ":(exclude)_drive_transfer" ":(exclude)Nurex_V4_2/batches" ":(exclude,glob)models_v*/**" ":(exclude,glob)**/*.pkl" ":(exclude,glob)**/*.db" ":(exclude,glob)**/*.duckdb*" ":(exclude,glob)**/*.sqlite*" ":(exclude,glob)**/*.parquet" ":(exclude,glob)**/.env*" >> "%LOG%" 2>&1 || goto :fail

git diff --cached --quiet origin/%BRANCH% && (
  echo.
  echo Nothing new: GitHub already has the same code as this PC.
  echo Nothing new >> "%LOG%"
  goto :cleanup
)

REM --- safety checks: no secrets, no file over 50 MB ---
git diff --cached --name-only --diff-filter=AM origin/%BRANCH% > "%ROOT%\.git\push_files.txt"
findstr /i /r "\.env$ \.env\. secret credential token\.json" "%ROOT%\.git\push_files.txt" > nul && (
  echo STOPPED: a file that looks like a secret would be pushed: >> "%LOG%"
  findstr /i /r "\.env$ \.env\. secret credential token\.json" "%ROOT%\.git\push_files.txt" >> "%LOG%"
  goto :fail
)
powershell -NoProfile -Command "$big = Get-Content '%ROOT%\.git\push_files.txt' | Where-Object { (Test-Path -LiteralPath $_) -and (Get-Item -LiteralPath $_).Length -gt 50MB }; if ($big) { 'STOPPED: files over 50 MB:'; $big; exit 1 }" >> "%LOG%" 2>&1 || goto :fail

echo.
echo ===== Changes that will be pushed to GitHub %BRANCH% =====
git diff --cached --ignore-cr-at-eol --stat=120 origin/%BRANCH%
git diff --cached --ignore-cr-at-eol --stat=120 origin/%BRANCH% >> "%LOG%" 2>&1
echo.
choice /C YN /M "Push these changes to GitHub"
if errorlevel 2 (
  echo Cancelled - nothing was pushed.
  echo Cancelled by user >> "%LOG%"
  goto :cleanup
)

for /f %%T in ('git write-tree') do set TREE=%%T
if "%TREE%"=="" goto :fail
> "%ROOT%\.git\push_msg.txt" (
  echo V4.1: sync code and settings from UK PC
  echo.
  echo Co-Authored-By: Claude Opus 5.5 ^<noreply@anthropic.com^>
  echo Claude-Session: https://claude.ai/code/session_01RuX8xnzkFeAKy9Up9ShJw9
)
for /f %%C in ('git commit-tree %TREE% -p origin/%BRANCH% -F "%ROOT%\.git\push_msg.txt"') do set COMMIT=%%C
if "%COMMIT%"=="" goto :fail
git push origin %COMMIT%:refs/heads/%BRANCH% >> "%LOG%" 2>&1 || goto :fail
git fetch origin %BRANCH% >> "%LOG%" 2>&1
echo.
echo PUSHED %COMMIT% to %BRANCH%. On the Denmark machine run scripts_v41\pull_on_denmark_v41.bat
echo PUSHED %COMMIT% >> "%LOG%"
goto :cleanup

:fail
echo.
echo FAILED - nothing was pushed and this folder is unchanged. Details: logs\v41_push_to_github.log
echo FAILED >> "%LOG%"
type "%LOG%"

:cleanup
if exist "%GIT_INDEX_FILE%" del /q "%GIT_INDEX_FILE%"
if exist "%ROOT%\.git\push_files.txt" del /q "%ROOT%\.git\push_files.txt"
if exist "%ROOT%\.git\push_msg.txt" del /q "%ROOT%\.git\push_msg.txt"
endlocal
pause
