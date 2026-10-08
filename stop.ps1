# PowerShell: .\stop.ps1          (graceful — KILL file, wait up to 60s)
#             .\stop.ps1 -Force   (immediate — pm2 stop now)
#
# Graceful shutdown: drop a KILL sentinel file, wait for the bot to notice
# at its next scan tick (it saves state + closes Supabase cleanly), then
# tell PM2 to stop the process so it doesn't auto-restart.

[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$TraderDir = "C:\Sijoy_2.0\automation\josho-trader"
$KillFile  = Join-Path $TraderDir "KILL"
$AppName   = "josho-trader"
$WaitSecs  = 60

function Get-PM2Status {
    param([string]$Name)
    $json = pm2 jlist 2>$null
    if (-not $json) { return $null }
    try {
        $procs = $json | ConvertFrom-Json
        $match = $procs | Where-Object { $_.name -eq $Name } | Select-Object -First 1
        if ($null -eq $match) { return $null }
        return $match.pm2_env.status
    } catch {
        return $null
    }
}

Write-Host "[stop.ps1] Stopping JoSho Trader ($AppName)" -ForegroundColor Cyan

if ($Force) {
    Write-Host "[stop.ps1] -Force flag set — stopping PM2 process immediately." -ForegroundColor Yellow
    pm2 stop $AppName
    Write-Host "[stop.ps1] Done." -ForegroundColor Green
    exit 0
}

# 1) Drop the KILL sentinel
"stopped via stop.ps1 at $(Get-Date -Format o)" | Out-File -FilePath $KillFile -Encoding utf8 -Force
Write-Host "[stop.ps1] KILL file written: $KillFile"
Write-Host "[stop.ps1] Waiting up to $WaitSecs seconds for graceful shutdown..."

# 2) Poll for graceful exit
$deadline = (Get-Date).AddSeconds($WaitSecs)
$stopped  = $false
while ((Get-Date) -lt $deadline) {
    $status = Get-PM2Status -Name $AppName
    if ($null -eq $status) {
        # Process gone from PM2 list entirely
        $stopped = $true
        break
    }
    if ($status -ne "online") {
        Write-Host "[stop.ps1] PM2 status: $status"
        $stopped = $true
        break
    }
    Start-Sleep -Seconds 2
}

if (-not $stopped) {
    Write-Host "[stop.ps1] Bot did not exit within $WaitSecs s. Forcing PM2 stop." -ForegroundColor Yellow
} else {
    Write-Host "[stop.ps1] Bot acknowledged KILL file — finalizing PM2 stop." -ForegroundColor Green
}

# 3) Tell PM2 to stop (prevents autorestart from kicking back in)
pm2 stop $AppName

Write-Host "[stop.ps1] Done. Use .\start.ps1 to bring it back up." -ForegroundColor Green
