@echo off
REM Nurex V4.1 - show the power / start settings of the V4.1 scheduled tasks (read only)
cd /d "%~dp0.."
if not exist logs mkdir logs
powershell -NoProfile -Command "foreach($n in 'Nurex_V41_Cycle','Nurex_V41_Recorder','Nurex_V41_Train','Nurex_V41_ShadowMonitor'){ $t=Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue; if($t){ $i=Get-ScheduledTaskInfo -TaskName $n; '{0}: state={1} onBatteryNoStart={2} stopOnBattery={3} startWhenAvailable={4} wake={5} limit={6} lastRun={7} lastResult={8} missed={9}' -f $n,$t.State,$t.Settings.DisallowStartIfOnBatteries,$t.Settings.StopIfGoingOnBatteries,$t.Settings.StartWhenAvailable,$t.Settings.WakeToRun,$t.Settings.ExecutionTimeLimit,$i.LastRunTime,$i.LastTaskResult,$i.NumberOfMissedRuns } else { $n + ': not found' } }; powercfg /a; (Get-CimInstance Win32_Battery | Select-Object -First 1 | ForEach-Object { 'battery status=' + $_.BatteryStatus + ' charge=' + $_.EstimatedChargeRemaining })" > logs\v41_task_check.log 2>&1
