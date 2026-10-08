# PowerShell: .\start.ps1
#
# Starts the JoSho Trader bot under PM2 supervision.
# Verifies prerequisites (python, pm2, vault), clears any stale KILL file,
# then hands off to pm2 + prints status.

$ErrorActionPreference = "Stop"

$TraderDir   = "C:\Sijoy_2.0\automation\josho-trader"
$EcosystemJs = Join-Path $TraderDir "ecosystem.config.js"
$KillFile    = Join-Path $TraderDir "KILL"
$LogsDir     = Join-Path $TraderDir "logs"
$VaultFile   = "C:\Sijoy_2.0\automation\secrets-os\vault.master.env"
$AppName     = "josho-trader"

Write-Host "[start.ps1] JoSho Trader bootstrap" -ForegroundColor Cyan

# 1) Python on PATH
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "[start.ps1] python not found on PATH. Install Python 3.10+ and ensure 'python' resolves."
    exit 1
}

# 2) PM2 installed
if (-not (Get-Command pm2 -ErrorAction SilentlyContinue)) {
    Write-Error @"
[start.ps1] pm2 not found on PATH.
Install with:
  npm install -g pm2 pm2-windows-startup
Then re-run this script.
"@
    exit 1
}

# 3) Vault exists
if (-not (Test-Path $VaultFile)) {
    Write-Error "[start.ps1] Vault not found at $VaultFile. The bot loads secrets from this file via _load_vault_env()."
    exit 1
}

# 4) Ecosystem config exists
if (-not (Test-Path $EcosystemJs)) {
    Write-Error "[start.ps1] Ecosystem config not found at $EcosystemJs."
    exit 1
}

# 5) Ensure logs dir
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
    Write-Host "[start.ps1] Created logs directory: $LogsDir"
}

# 6) Clear stale KILL file (a previous stop.ps1 may have left one behind)
if (Test-Path $KillFile) {
    Remove-Item $KillFile -Force
    Write-Host "[start.ps1] Removed stale KILL file at $KillFile" -ForegroundColor Yellow
}

# 7) Start under PM2
Write-Host "[start.ps1] Starting PM2 app '$AppName'..." -ForegroundColor Cyan
pm2 start $EcosystemJs

# 8) Status
Write-Host ""
pm2 status $AppName

# 9) Echo the 3 things Sijoy needs to know
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " JoSho Trader is supervised by PM2." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host " PM2 dashboard : pm2 monit"
Write-Host " Live logs     : pm2 logs $AppName"
Write-Host " Log files     : $LogsDir\pm2-out.log"
Write-Host "                 $LogsDir\pm2-error.log"
Write-Host " Stop (graceful): .\stop.ps1"
Write-Host " Stop (force)   : .\stop.ps1 -Force"
Write-Host "============================================================" -ForegroundColor Green
