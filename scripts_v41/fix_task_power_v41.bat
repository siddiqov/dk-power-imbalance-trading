@echo off
REM Nurex V4.1 - let the V4.1 scheduled tasks run on battery too (2026-09-29).
REM Why: the tasks were created with the Windows defaults "start only on AC power" and "stop when
REM the PC switches to battery". On battery the 15-min cycle did not run, so whole hours of locked
REM batches were missed ("batch deadline passed before the system ran").
REM Also: run a missed start as soon as possible, and let the cycle wake the PC from sleep.
cd /d "%~dp0.."
if not exist logs mkdir logs
powershell -NoProfile -Command "foreach($n in 'Nurex_V41_Cycle','Nurex_V41_Recorder','Nurex_V41_Train','Nurex_V41_ShadowMonitor'){ $t=Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue; if(-not $t){ $n + ': not found'; continue }; $t.Settings.DisallowStartIfOnBatteries=$false; $t.Settings.StopIfGoingOnBatteries=$false; $t.Settings.StartWhenAvailable=$true; if($n -eq 'Nurex_V41_Cycle'){ $t.Settings.WakeToRun=$true }; try { Set-ScheduledTask -InputObject $t -ErrorAction Stop | Out-Null; $u=Get-ScheduledTask -TaskName $n; '{0}: onBatteryNoStart={1} stopOnBattery={2} startWhenAvailable={3} wake={4}' -f $n,$u.Settings.DisallowStartIfOnBatteries,$u.Settings.StopIfGoingOnBatteries,$u.Settings.StartWhenAvailable,$u.Settings.WakeToRun } catch { $n + ': FAILED ' + $_.Exception.Message } }" > logs\v41_task_fix.log 2>&1
