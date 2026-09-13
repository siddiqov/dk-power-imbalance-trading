# ==============================================================================
# scripts/configure_task_scheduler.ps1
# Configures 24/7 Nurex Power Trading Background Service via Windows Task Scheduler
# ==============================================================================

$taskName = "Nurex_PowerTrading_AutoPublisher"
$projectRoot = "C:\Users\Hafeez\Documents\Nurex_Trading\Basic_Approach"
$vbsPath = "$projectRoot\scripts\start_publisher_silent.vbs"
$user = "$env:USERDOMAIN\$env:USERNAME"

Write-Host "======================================================================"
Write-Host "⚙️  CONFIGURING 24/7 NUREX AUTO-PUBLISHER WINDOWS SCHEDULED TASK"
Write-Host "   Task Name:    $taskName"
Write-Host "   Target User:  $user"
Write-Host "   Script Path:  $vbsPath"
Write-Host "======================================================================"

# 1. Check if task already exists and unregister if needed
try {
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Unregistering previous task definition..."
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    }
} catch {
    Write-Host "No existing task found."
}

# 2. Define Action (Silent execution via wscript)
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbsPath`"" -WorkingDirectory $projectRoot

# 3. Define Triggers: At Logon + At Startup / Boot
$triggerLogon = New-ScheduledTaskTrigger -AtLogOn -User $user

# 4. Define Settings: Infinite execution, ignore duplicates, wake machine, auto-restart
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1)

# 5. Define Principal / Register Task
try {
    # Attempt registration with User parameter
    $task = Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger @($triggerLogon) `
        -Settings $settings `
        -User $env:USERNAME `
        -Description "Nurex 24/7 Power Trading Continuous Prediction Publisher to Supabase and Day-Ahead Auction Engine."
} catch {
    Write-Host "Fallback registration without explicit user binding..."
    $task = Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger @($triggerLogon) `
        -Settings $settings `
        -Description "Nurex 24/7 Power Trading Continuous Prediction Publisher to Supabase and Day-Ahead Auction Engine."
}

Write-Host "✅ Scheduled Task '$taskName' registered successfully!"

# 7. Start Task immediately
Write-Host "Starting Scheduled Task '$taskName' now..."
Start-ScheduledTask -TaskName $taskName

Start-Sleep -Seconds 2

# 8. Query Status
$status = Get-ScheduledTask -TaskName $taskName
Write-Host "Current Task Status: $($status.State)"
Write-Host "======================================================================"
