# =============================================================================
# Nurex V4.1 - DENMARK PC ONLY. One go, unattended (2026-10-12):
#   1. waits for the safe update window (minute :50 - :08 Danish time)
#   2. fetches GitHub; checks that this PC's local code edits are also on GitHub
#      (if not: STOPS, changes nothing, writes logs\dk_local_changes.diff)
#   3. backs up and clears the files that blocked the last pull
#   4. runs the (new) pull_on_denmark_v41.bat: pause cycle, update code, test forecast,
#      resume cycle, restart dashboard
#   5. runs run_dk_research_v41.bat (research on a store copy, ~8-12 h, low priority)
# It never retrains or replaces the live models - that needs your go after the report.
# Log: logs\v41_auto_pull_research.log
# Start (PowerShell):  powershell -ExecutionPolicy Bypass -File <this file>
# =============================================================================
param([string]$Root = 'C:\energy-trading')

$ErrorActionPreference = 'Continue'
$Br = 'v4.1-phase5-cleanup'
Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force (Join-Path $Root 'logs') | Out-Null
$Log = Join-Path $Root 'logs\v41_auto_pull_research.log'
$env:NUREX_UNATTENDED = '1'

function Say([string]$m) {
    $l = '{0:yyyy-MM-dd HH:mm:ss} {1}' -f (Get-Date), $m
    Write-Host $l
    Add-Content -LiteralPath $Log -Value $l
}
function GitOut { $o = & git @args 2>$null; return @($o | Where-Object { $_ -and $_.Trim() }) }
function GitRun {
    $o = & git @args 2>&1
    $c = $LASTEXITCODE
    $o | ForEach-Object { Add-Content -LiteralPath $Log -Value ("    git: " + $_) }
    return $c
}
function Stop-Here([string]$m) {
    Say "STOPPED: $m"
    Say 'The live system was left as it is (cycle, models, journals unchanged unless stated above).'
    Read-Host 'Press Enter to close'
    exit 1
}

Say "==== auto pull + DK research start (root $Root) ===="
if (-not (Test-Path -LiteralPath (Join-Path $Root 'train_v4_1.py'))) { Stop-Here "train_v4_1.py not found in $Root" }

# ---------- 1. wait for the safe window ----------
$last = -1
while ($true) {
    $m = (Get-Date).Minute
    if ($m -ge 50 -or $m -le 8) { break }
    if ($last -ne (Get-Date).Hour * 100 + [math]::Floor($m / 10)) {
        $last = (Get-Date).Hour * 100 + [math]::Floor($m / 10)
        Say "waiting for the safe update window (:50 - :08) ... keep this window open"
    }
    Start-Sleep -Seconds 20
}
Say 'window open - starting the update'

# ---------- 2. fetch + check local edits ----------
if ((GitRun fetch origin "+refs/heads/${Br}:refs/remotes/origin/$Br") -ne 0) { Stop-Here 'git fetch failed (internet / GitHub login?)' }
$head = (GitOut rev-parse HEAD)[0]
$target = (GitOut rev-parse "origin/$Br")[0]
Say "this PC: $head   GitHub: $target"

$skipPull = $false
if ($head -eq $target) {
    Say 'code is already up to date - skipping the pull'
    $skipPull = $true
}

if (-not $skipPull) {
    $incoming  = GitOut diff --name-only HEAD "origin/$Br"
    $localMod  = GitOut diff --name-only HEAD
    $untracked = GitOut ls-files --others --exclude-standard
    $clash     = @($localMod | Where-Object { $incoming -contains $_ })
    $clashU    = @($untracked | Where-Object { $incoming -contains $_ })

    $bad = @()
    foreach ($f in $clash) {
        & git diff --quiet --ignore-cr-at-eol "origin/$Br" -- $f 2>$null
        if ($LASTEXITCODE -ne 0) { $bad += $f }
    }
    if ($bad.Count -gt 0) {
        $diff = Join-Path $Root 'logs\dk_local_changes.diff'
        (& git diff --ignore-cr-at-eol "origin/$Br" -- $bad 2>$null) | Set-Content -LiteralPath $diff -Encoding UTF8
        Stop-Here ("this PC has its own edits that are NOT on GitHub: " + ($bad -join ', ') +
                   ". Nothing was changed. Send logs\dk_local_changes.diff to Claude.")
    }

    # ---------- 3. back up + clear the blocking files ----------
    $bk = Join-Path $Root ('backups\pre_pull_auto_{0:yyyyMMdd_HHmm}' -f (Get-Date))
    foreach ($f in $clash) {
        $src = Join-Path $Root ($f -replace '/', '\')
        $dst = Join-Path $bk ($f -replace '/', '\')
        New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
        Copy-Item -LiteralPath $src -Destination $dst -Force
        GitRun checkout -- $f | Out-Null
        Say "local edit identical to GitHub - backed up and cleared: $f"
    }
    foreach ($f in $clashU) {
        $src = Join-Path $Root ($f -replace '/', '\')
        $dst = Join-Path $bk ($f -replace '/', '\')
        New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
        Move-Item -LiteralPath $src -Destination $dst -Force
        Say "untracked file moved out of the way: $f"
    }
    # leftover manual copies from 10 Oct (no longer on GitHub) - tidy into the backup folder
    Get-ChildItem -LiteralPath $Root -Recurse -Filter '*.bak-20261010' -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notlike "$Root\backups\*" } | ForEach-Object {
            $rel = $_.FullName.Substring($Root.Length + 1)
            $dst = Join-Path $bk $rel
            New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
            Move-Item -LiteralPath $_.FullName -Destination $dst -Force
            Say "moved old manual copy: $rel"
        }

    # ---------- 4. run the NEW pull script (taken from GitHub, CRLF, from %TEMP%) ----------
    $pull = Join-Path $env:TEMP 'nurex_pull_on_denmark_new.bat'
    $lines = & git show "origin/${Br}:scripts_v41/pull_on_denmark_v41.bat" 2>$null
    if (-not $lines) { Stop-Here 'could not read the new pull script from GitHub' }
    ($lines -join "`r`n") + "`r`n" | Set-Content -LiteralPath $pull -Encoding ASCII -NoNewline
    Say 'running pull_on_denmark_v41.bat (pause cycle -> update -> test forecast -> resume -> dashboard) ...'
    & cmd.exe /c "`"$pull`" --run `"$Root`""
    $now = (GitOut rev-parse HEAD)[0]
    $pullLog = Join-Path $Root 'logs\v41_pull.log'
    $done = (Test-Path -LiteralPath $pullLog) -and (@(Get-Content -LiteralPath $pullLog -Tail 3 | Where-Object { $_ -match '^DONE' }).Count -gt 0)
    if ($now -ne $target -or -not $done) {
        Stop-Here ("the pull did not finish (code at $now, expected $target). If it says the cycle is PAUSED, " +
                   "send logs\v41_pull.log to Claude; to resume trading on the old code: schtasks /Change /TN Nurex_V41_Cycle /ENABLE")
    }
    Say "pull DONE - code now at $now, cycle resumed"
}

# make sure the cycle is enabled whatever happened above
& schtasks /Query /TN Nurex_V41_Cycle /FO LIST 2>$null | Select-String -Pattern 'Disabled' -Quiet | ForEach-Object {
    if ($_) { & schtasks /Change /TN Nurex_V41_Cycle /ENABLE | Out-Null; Say 'cycle task was disabled - enabled it again' }
}

# ---------- 5. research ----------
$rb = Join-Path $Root 'scripts_v41\run_dk_research_v41.bat'
if (-not (Test-Path -LiteralPath $rb)) { Stop-Here 'run_dk_research_v41.bat not found after the pull' }
Say 'starting the DK research (8-12 h, low priority, live trading continues) ...'
& cmd.exe /c "`"$rb`""
$rc = $LASTEXITCODE
Say "research finished (exit $rc). Report: $Root\results\research_dk\REPORT.md"
Say 'Send REPORT.md and week_review.md to Claude. The live models were NOT changed.'
Read-Host 'Press Enter to close'
