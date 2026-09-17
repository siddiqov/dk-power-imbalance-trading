# Nurex V4.1 - register Windows scheduled tasks (run once in PowerShell as the logged-in user).
#   1) Nurex_V41_Cycle     every 15 minutes: data update + lock upcoming gates + settle
#   2) Nurex_V41_Recorder  at logon: Nord Pool intraday recorder (keeps running, reconnects)
#   3) Nurex_V41_Train     weekly, Sunday 03:00: retrain the models
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$cycle  = Join-Path $here "run_update_v41.bat"
$record = Join-Path $here "run_recorder_v41.bat"
$train  = Join-Path $here "run_train_v41.bat"
schtasks /Delete /TN "Nurex_V41_Update" /F 2>$null | Out-Null
schtasks /Create /F /TN "Nurex_V41_Cycle" /SC MINUTE /MO 15 /TR "`"$cycle`""
schtasks /Create /F /TN "Nurex_V41_Recorder" /SC ONLOGON /TR "`"$record`""
schtasks /Create /F /TN "Nurex_V41_Train" /SC WEEKLY /D SUN /ST 03:00 /TR "`"$train`""
schtasks /Run /TN "Nurex_V41_Recorder"
Write-Host "Tasks installed. Remove with:"
Write-Host "  schtasks /Delete /TN Nurex_V41_Cycle /F ; schtasks /Delete /TN Nurex_V41_Recorder /F ; schtasks /Delete /TN Nurex_V41_Train /F"
