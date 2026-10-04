@echo off
REM ============================================================================
REM Nurex V4.1 - update THIS machine (Denmark) from GitHub branch v4.1-phase5-cleanup
REM and (optionally) install model files from Google Drive.   (2026-09-30, updated 2026-10-04)
REM
REM  1. pauses the V4.1 cycle task (the old auto publisher is not touched)
REM  2. fetches the branch; any local file the update would overwrite is first
REM     copied to backups\pre_pull_<time>\ (nothing is lost)
REM  3. moves the branch to exactly what is on GitHub (never merges); this machine's
REM     data files (*.db, *.sqlite, *.duckdb, data\, results\, logs\) are kept as they are
REM  4. only if MODELS_FROM is set: copies the 4 LightGBM model files, after backing up the
REM     current ones, and checks them against SHA256SUMS.txt (restores the backup on mismatch)
REM  5. test forecast DK1 + DK2 with V4.1 LightGBM; if it fails, the cycle stays PAUSED
REM  6. lets the tasks run on battery, resumes the cycle, runs it once
REM  7. trains any recommended model the machine does not have yet (2026-10-04: DK1 logistic);
REM     meanwhile that zone keeps trading with V4.1 LightGBM. Then restarts the dashboard.
REM Kept: .env, data store, journal, results, logs, Nord Pool recordings.
REM Log: logs\v41_pull.log
REM
REM FIRST TIME ONLY (the script is not on this machine yet), in Command Prompt:
REM   git fetch origin v4.1-phase5-cleanup
REM   git checkout origin/v4.1-phase5-cleanup -- scripts_v41\pull_on_denmark_v41.bat
REM ============================================================================
REM The update may replace this very file, so it always runs from a copy in %TEMP%.
if /i not "%~1"=="--run" (
  copy /y "%~f0" "%TEMP%\nurex_pull_on_denmark_run.bat" > nul
  call "%TEMP%\nurex_pull_on_denmark_run.bat" --run "%~dp0.."
  exit /b
)
setlocal EnableDelayedExpansion
cd /d "%~f2"
set ROOT=%CD%
set BRANCH=v4.1-phase5-cleanup

REM >>> Folder on Google Drive with v4_1_batch_DK1/DK2 .pkl/.json + SHA256SUMS.txt.
REM >>> Change it to the real path. Leave it empty (set MODELS_FROM=) to update code only.
set MODELS_FROM=

set PY=Nurex_V4_2\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if not exist logs mkdir logs
set LOG=%ROOT%\logs\v41_pull.log
for /f %%s in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set STAMP=%%s
set BK=%ROOT%\backups\pre_pull_%STAMP%
echo ==== %DATE% %TIME% pull %BRANCH% ==== > "%LOG%"
set CHANGED=0
set HADTASK=0
if not exist "%ROOT%\train_v4_1.py" (
  echo WRONG FOLDER: this script must be run from ^<project folder^>\scripts_v41\ on this machine,
  echo the folder that contains train_v4_1.py and the .git folder. It is now in: %ROOT%
  echo Nothing was changed.
  goto :end
)
git --version > nul 2>&1 || (echo git is not installed or not on PATH - nothing was changed & goto :end)
git rev-parse --is-inside-work-tree > nul 2>&1 || (echo %ROOT% is not a git folder - nothing was changed & goto :end)

REM ---------- 1. pause the cycle (wait if a run is in progress) ----------
schtasks /Query /TN Nurex_V41_Cycle > nul 2>&1 && set HADTASK=1
if "%HADTASK%"=="1" (
  schtasks /Change /TN Nurex_V41_Cycle /DISABLE >> "%LOG%" 2>&1
  for /l %%i in (1,1,20) do (
    schtasks /Query /TN Nurex_V41_Cycle /FO LIST | findstr /i "Running" > nul && (
      echo A cycle run is in progress - waiting 30 s ...
      timeout /t 30 /nobreak > nul
    )
  )
  echo [1] V4.1 cycle paused
) else (
  echo [1] Task Nurex_V41_Cycle not found - V4.1 is not scheduled on this machine yet
)

REM ---------- 2. fetch + protect local changes and data ----------
git fetch origin +refs/heads/%BRANCH%:refs/remotes/origin/%BRANCH% >> "%LOG%" 2>&1 || (echo FAILED: git fetch - git said: & powershell -NoProfile -Command "Get-Content -Tail 6 -LiteralPath '%LOG%'" & goto :fail)
for /f %%h in ('git rev-parse HEAD') do set OLDHEAD=%%h
echo previous code: %OLDHEAD%   (rollback: git checkout -B %BRANCH% %OLDHEAD%) >> "%LOG%"
echo     previous code: %OLDHEAD%
git rev-parse --verify --quiet %BRANCH% > nul && (
  git merge-base --is-ancestor %BRANCH% origin/%BRANCH% || (echo FAILED: local branch %BRANCH% has commits that are not on GitHub - not overwriting them & goto :fail)
)
git diff --name-only HEAD origin/%BRANCH% > "%ROOT%\.git\pull_incoming.txt"
git diff --name-only HEAD > "%ROOT%\.git\pull_local.txt"
git ls-files --others --exclude-standard > "%ROOT%\.git\pull_untracked.txt"
REM data, journals, stores and outputs of THIS machine are always kept as they are on disk
findstr /r /i "\.db$ \.sqlite \.duckdb \.parquet ^data/ ^results/ ^logs/ ^Nurex_V4_2/batches/ ^Nurex_V4_2/data/" "%ROOT%\.git\pull_incoming.txt" > "%ROOT%\.git\pull_keep.txt"
set NKEEP=0
for /f "usebackq delims=" %%f in ("%ROOT%\.git\pull_keep.txt") do (
  set "P=%%f" & set "P=!P:/=\!"
  if exist "!P!" (
    for %%d in ("%BK%\keep\!P!") do if not exist "%%~dpd" mkdir "%%~dpd"
    copy /y "!P!" "%BK%\keep\!P!" > nul
    set /a NKEEP+=1
  )
)
set CHANGED=1
set NBK=0
for /f "usebackq delims=" %%f in (`findstr /x /l /g:"%ROOT%\.git\pull_incoming.txt" "%ROOT%\.git\pull_local.txt"`) do (
  set "P=%%f" & set "P=!P:/=\!"
  if exist "!P!" (
    for %%d in ("%BK%\code\!P!") do if not exist "%%~dpd" mkdir "%%~dpd"
    copy /y "!P!" "%BK%\code\!P!" > nul
  )
  git checkout -- "%%f" >> "%LOG%" 2>&1
  echo backed up and reset local change: %%f >> "%LOG%"
  set /a NBK+=1
)
for /f "usebackq delims=" %%f in (`findstr /x /l /g:"%ROOT%\.git\pull_incoming.txt" "%ROOT%\.git\pull_untracked.txt"`) do (
  set "P=%%f" & set "P=!P:/=\!"
  for %%d in ("%BK%\code\!P!") do if not exist "%%~dpd" mkdir "%%~dpd"
  move /y "!P!" "%BK%\code\!P!" > nul
  echo moved untracked file out of the way: %%f >> "%LOG%"
  set /a NBK+=1
)
echo [2] %NBK% local code file(s) saved to %BK%\code, %NKEEP% data file(s) protected

REM ---------- 3. move to the new code (branch = exactly GitHub, never a merge) ----------
git checkout -B %BRANCH% --track origin/%BRANCH% >> "%LOG%" 2>&1 || (echo FAILED: could not switch to the new code - see log & goto :restorekeep)
for /f "usebackq delims=" %%f in ("%ROOT%\.git\pull_keep.txt") do (
  set "P=%%f" & set "P=!P:/=\!"
  if exist "%BK%\keep\!P!" (
    for %%d in ("!P!") do if not exist "%%~dpd" mkdir "%%~dpd"
    copy /y "%BK%\keep\!P!" "!P!" > nul
    echo kept this machine's data file: %%f >> "%LOG%"
  )
)
for /f "delims=" %%l in ('git log -1 --format^="%%h %%s"') do set HEADLINE=%%l
echo [3] code now at: !HEADLINE!
echo code now at: !HEADLINE! >> "%LOG%"

REM ---------- 4. models from Drive (optional) ----------
if "%MODELS_FROM%"=="" (
  echo [4] MODELS_FROM empty - models not copied ^(this machine trains its own^)
  goto :checks
)
if not exist "%MODELS_FROM%\v4_1_batch_DK1.pkl" (echo FAILED: model files not found in %MODELS_FROM% & goto :fail)
if not exist "%BK%\models" mkdir "%BK%\models"
copy /y models_v4_1_batch\*.* "%BK%\models\" > nul 2>&1
copy /y "%MODELS_FROM%\v4_1_batch_DK1.pkl" models_v4_1_batch\ > nul || goto :modelfail
copy /y "%MODELS_FROM%\v4_1_batch_DK1.json" models_v4_1_batch\ > nul || goto :modelfail
copy /y "%MODELS_FROM%\v4_1_batch_DK2.pkl" models_v4_1_batch\ > nul || goto :modelfail
copy /y "%MODELS_FROM%\v4_1_batch_DK2.json" models_v4_1_batch\ > nul || goto :modelfail
if exist "%MODELS_FROM%\SHA256SUMS.txt" (
  powershell -NoProfile -Command "$bad=0; Get-Content -LiteralPath '%MODELS_FROM%\SHA256SUMS.txt' | Where-Object { $_.Trim() } | ForEach-Object { $h,$n = $_.Trim() -split '\s+',2; $f = Join-Path 'models_v4_1_batch' $n; if ((Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash -ne $h.ToUpper()) { 'MISMATCH ' + $n; $bad++ } else { 'ok ' + $n } }; exit $bad" >> "%LOG%" 2>&1 || goto :modelfail
  echo [4] models installed, all checksums match
) else (
  echo [4] models installed - WARNING: no SHA256SUMS.txt in the Drive folder, files not verified
)
findstr /c:"trained_until" models_v4_1_batch\v4_1_batch_DK1.json

:checks
REM ---------- 5. settings + test forecast ----------
echo [5] settings:
findstr /r /c:"^  mode:" /c:"enhanced:" /c:"mfrr_volume_features:" /c:"intraday_market_features:" v4_1_intraday\config_v41.yaml
echo     test forecast DK1 + DK2 (1-3 minutes) ...
"%PY%" train_v4_1.py --mode batch predict --area DK1 --family lgbm > logs\v41_pull_predict.log 2>&1 || (echo FAILED: test forecast DK1 - see logs\v41_pull_predict.log & goto :fail)
"%PY%" train_v4_1.py --mode batch predict --area DK2 --family lgbm >> logs\v41_pull_predict.log 2>&1 || (echo FAILED: test forecast DK2 - see logs\v41_pull_predict.log & goto :fail)
echo     test forecast OK

REM ---------- 6. resume ----------
if "%HADTASK%"=="1" (
  call scripts_v41\fix_task_power_v41.bat
  schtasks /Change /TN Nurex_V41_Cycle /ENABLE >> "%LOG%" 2>&1
  schtasks /Run /TN Nurex_V41_Cycle >> "%LOG%" 2>&1
  echo [6] V4.1 cycle resumed and started once
) else (
  echo [6] To schedule V4.1 here run:  powershell -ExecutionPolicy Bypass -File scripts_v41\install_tasks_v41.ps1
  echo     then scripts_v41\fix_task_power_v41.bat. It also starts the Nord Pool recorder - run the
  echo     recorder on ONE machine only ^(same Nord Pool login^).
)
REM ---------- 7. train any recommended model this machine does not have yet (2026-10-04) ----------
REM The cycle is already running: until the new model exists, a zone keeps locking with V4.1 LightGBM.
set MISSING=
for /f "usebackq delims=" %%m in (`"%PY%" scripts_v41\missing_models_v41.py 2^>nul`) do set MISSING=%%m
if not "!MISSING!"=="" (
  echo [7] training the recommended model^(s^) not on this machine yet: !MISSING!  ^(10-20 min, see logs\v41_train.log^)
  REM wait until the cycle run started in step 6 has finished - the data store allows one writer
  for /l %%i in (1,1,20) do (
    schtasks /Query /TN Nurex_V41_Cycle /FO LIST 2>nul | findstr /i "Running" > nul && timeout /t 30 /nobreak > nul
  )
  for %%z in (!MISSING!) do (
    for /f "tokens=1,2 delims=:" %%a in ("%%z") do (
      echo ==== %DATE% %TIME% pull: train %%a %%b ==== >> logs\v41_train.log
      "%PY%" train_v4_1.py --mode batch train --area %%a --family %%b >> logs\v41_train.log 2>&1 || echo     WARNING: training %%a %%b failed - %%a keeps trading with V4.1 LightGBM. See logs\v41_train.log
      "%PY%" train_v4_1.py --mode batch predict --area %%a >> logs\v41_pull_predict.log 2>&1 && echo     %%a %%b trained and test forecast OK - used from the next locked batch
    )
  )
) else (
  echo [7] every zone's recommended model is already on this machine
)
call scripts_v41\restart_dashboard_v41.bat
echo.
echo ===== DONE. Dashboard: http://localhost:5005   Log: logs\v41_pull.log =====
echo DONE >> "%LOG%"
goto :end

:restorekeep
for /f "usebackq delims=" %%f in ("%ROOT%\.git\pull_keep.txt") do (
  set "P=%%f" & set "P=!P:/=\!"
  if exist "%BK%\keep\!P!" copy /y "%BK%\keep\!P!" "!P!" > nul
)
goto :fail

:modelfail
echo FAILED: model copy or checksum - restoring the previous models
copy /y "%BK%\models\*.*" models_v4_1_batch\ > nul 2>&1
echo model files restored from %BK%\models >> "%LOG%"

:fail
echo.
if "%CHANGED%"=="0" (
  if "%HADTASK%"=="1" schtasks /Change /TN Nurex_V41_Cycle /ENABLE >> "%LOG%" 2>&1
  echo ===== STOPPED before anything was changed. The V4.1 cycle was resumed - trading continues on the old code. =====
  echo Fix the problem ^(or send logs\v41_pull.log^), then run this script again.
  echo STOPPED, nothing changed, cycle resumed >> "%LOG%"
  goto :end
)
echo ===== STOPPED. The V4.1 cycle is left PAUSED so nothing trades on a half-updated system. =====
echo Fix the problem (or send logs\v41_pull.log), then run this script again.
echo To resume without updating:  schtasks /Change /TN Nurex_V41_Cycle /ENABLE
echo STOPPED >> "%LOG%"

:end
del /q "%ROOT%\.git\pull_incoming.txt" "%ROOT%\.git\pull_local.txt" "%ROOT%\.git\pull_untracked.txt" "%ROOT%\.git\pull_keep.txt" 2> nul
endlocal
pause
