#Requires -Version 5.1
<#
.SYNOPSIS
    Dependency bootstrap helper -- installs all required packages via uv.

.DESCRIPTION
    Creates a local .venv and installs every package listed in
    requirements.txt using uv. Run this once before running the app.

    There is only one venv now: Argus has no model runtime, no OpenCV, and no
    camera dependency of its own (pose and form/exercise classification run
    on each trainee's phone -- see docs/PROTOCOL.md), so it runs on any host
    architecture the same way.

.EXAMPLE
    .\run.ps1
    .\run.ps1 -Python 3.11
#>

[CmdletBinding()]
param(
    [string]$Python = "3.11"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Ok($msg)   { Write-Host "  [OK]    $msg" -ForegroundColor Green }
function Write-Info($msg) { Write-Host "  [INFO]  $msg" }
function Write-Warn($msg) { Write-Host "  [WARN]  $msg" -ForegroundColor Yellow }

$ReqFile = Join-Path $PSScriptRoot "requirements.txt"
$VenvDir = Join-Path $PSScriptRoot ".venv"

if (-not (Test-Path $ReqFile)) {
    Write-Error "requirements.txt not found: $ReqFile"
    exit 1
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Info "uv not found -- installing ..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","User") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH","Machine")
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error "uv install failed. Install manually: https://docs.astral.sh/uv/"
        exit 1
    }
}
Write-Ok "uv $(uv --version)"

if (-not (Test-Path $VenvDir)) {
    Write-Info "Creating .venv (Python $Python) ..."
    uv venv "$VenvDir" --python $Python
    if ($LASTEXITCODE -ne 0) { Write-Error "uv venv failed."; exit 1 }
    Write-Ok ".venv created"
} else {
    Write-Ok ".venv already exists -- skipping creation"
}

Write-Info "Installing dependencies from requirements.txt ..."
uv pip install --system-certs --python "$VenvDir\Scripts\python.exe" -r "$ReqFile"
if ($LASTEXITCODE -ne 0) { Write-Error "uv pip install failed."; exit 1 }

Write-Info "Installing argus itself (editable) ..."
uv pip install --system-certs --python "$VenvDir\Scripts\python.exe" -e "$PSScriptRoot"
if ($LASTEXITCODE -ne 0) { Write-Error "editable install failed."; exit 1 }
Write-Ok "All dependencies installed"

Write-Info "Diagnose the host and the configured ingest port:"
Write-Info "  .venv\Scripts\python.exe -m argus.cli doctor"
Write-Info "Run the ingest server (no phone needed -- generate and replay a fixture):"
Write-Info "  .venv\Scripts\python.exe -m argus.cli demo --ticks 60"
Write-Info "  .venv\Scripts\python.exe -m argus.cli run --http-port 8080 &"
Write-Info "  .venv\Scripts\python.exe demo\replay_client.py"
Write-Info "Run the tests with:"
Write-Info "  .venv\Scripts\python.exe -m pytest tests\ -q"
