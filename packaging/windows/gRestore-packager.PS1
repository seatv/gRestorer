param(
  [ValidateSet("nvidia","cpu","intel")]
  [string]$Target = "nvidia",

  [string]$Name = "grestorer",

  [switch]$SkipFfmpeg = $false,

  [switch]$SwapPolarsLtsCpu = $true
)

$ErrorActionPreference = "Stop"

function Ensure-Tool($cmd) {
  $p = Get-Command $cmd -ErrorAction SilentlyContinue
  if (-not $p) { throw "Missing required tool '$cmd' on PATH." }
}

Write-Host "== gRestorer packaging ==" -ForegroundColor Cyan
Write-Host "Target: $Target   Name: $Name" -ForegroundColor Cyan
Write-Host "Repo:   $(Get-Location)" -ForegroundColor Cyan

# Basic sanity: must run from repo root
if (-not (Test-Path ".\pyproject.toml")) { throw "Run this from the repo root (pyproject.toml not found)." }
if (-not (Test-Path ".\gRestorer\cli\main.py")) { throw "Repo layout unexpected: .\gRestorer\cli\main.py not found." }
if (-not (Test-Path ".\packaging\windows\grestorer_entrypoint.py")) { throw "Missing packaging entrypoint: .\packaging\windows\grestorer_entrypoint.py" }

# Ensure we're in an activated venv (recommended)
if (-not $env:VIRTUAL_ENV) {
  Write-Warning "No active virtualenv detected. Building from system Python is possible, but not recommended."
}

# Install PyInstaller (keep it simple; you can pin later)
python -m pip install --upgrade pip
python -m pip install --upgrade pyinstaller

# Optional: avoid polars AVX surprises on older CPUs (ultralytics pulls it in sometimes)
if ($SwapPolarsLtsCpu) {
  $polars = (& python -c "import importlib.util; print('1' if importlib.util.find_spec('polars') else '0')").Trim()
  if ($polars.Trim() -eq "1") {
    Write-Host "Swapping polars -> polars-lts-cpu (best effort)..." -ForegroundColor Yellow
    python -m pip uninstall -y polars | Out-Null
    python -m pip install -U polars-lts-cpu | Out-Null
  }
}

# Build args for spec
$specArgs = @("--target=$Target", "--name=$Name")
if ($SkipFfmpeg) { $specArgs += "--skip-ffmpeg" }

Write-Host "Running PyInstaller..." -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean .\packaging\windows\grestorer.spec -- $specArgs

# Add a convenience wrapper so config.json resolves (cwd=exe dir)
$distDir = Join-Path (Resolve-Path ".\dist").Path $Name
$cmdPath = Join-Path $distDir "grestorer.cmd"
$exePath = Join-Path $distDir "$Name.exe"

@"
@echo off
setlocal
cd /d %~dp0
"%~dp0$Name.exe" %*
"@ | Set-Content -Encoding ASCII $cmdPath

# Copy config.json next to the exe (PyInstaller v6 onedir uses _internal for bundled data)
$cfgSrc = Join-Path (Resolve-Path ".").Path "config.json"
$cfgDst = Join-Path $distDir "config.json"
if (Test-Path $cfgSrc) {
  Copy-Item -Force $cfgSrc $cfgDst
  Write-Host "Wrote config.json: $cfgDst" -ForegroundColor Green
} else {
  Write-Warning "config.json not found at repo root: $cfgSrc (skipping copy)"
}


Write-Host "Creating self-extracting installer..." -ForegroundColor Yellow
$7zCheck = Get-Command 7z -ErrorAction SilentlyContinue
if ($7zCheck) {
    $installerName = "gRestorer-install.exe"
    7z a -sfx "$installerName" ".\dist\$Name"
    
    if ($LASTEXITCODE -eq 0) {
        $size = [math]::Round((Get-Item $installerName).Length / 1MB, 2)
        Write-Host "Self-extracting installer created: $installerName (${size}MB)" -ForegroundColor Green
    } else {
        Write-Host "WARNING: Failed to create self-extracting installer" -ForegroundColor Yellow
    }
} else {
    Write-Host "7z not found - skipping self-extracting installer" -ForegroundColor Yellow
    Write-Host "Install 7-Zip to enable this feature: https://www.7-zip.org/" -ForegroundColor Gray
}


Write-Host "Done." -ForegroundColor Green
Write-Host "Output: $distDir" -ForegroundColor Green
Write-Host "Run:    $cmdPath --input ... --output ..." -ForegroundColor Green
