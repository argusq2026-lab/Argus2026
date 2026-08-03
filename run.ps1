#Requires -Version 5.1
<#
.SYNOPSIS
    Dependency bootstrap helper -- installs all required packages via uv.

.DESCRIPTION
    Creates a local .venv and installs every package listed in
    requirements.txt using uv. Run this once before running the app.

    The default venv covers the fully-tested path: capture, tracking, triage,
    every output sink, CPU ONNX inference, and the whole pytest suite.

    It is created from an x86-64 interpreter even on a Snapdragon X-Elite
    host. opencv-python has NO win-arm64 wheel and falls back to a from-source
    numpy/meson build that fails without a C toolchain, so the capture path
    and the NPU runtime cannot share one environment. It runs fine under
    emulation.

    Pass -Npu to ALSO create a second, separate native-ARM64 venv (.venv-npu)
    with onnxruntime-qnn / onnxruntime-genai for real Hexagon NPU sessions --
    no opencv, no pytest in that venv.

.EXAMPLE
    .\run.ps1
    .\run.ps1 -Python 3.11
    .\run.ps1 -Npu
#>

[CmdletBinding()]
param(
    [string]$Python = "3.11",
    [switch]$Npu
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

function Get-NativeArm64Python {
    $root = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (-not (Test-Path $root)) { return $null }
    Get-ChildItem $root -Recurse -Filter "python.exe" -ErrorAction SilentlyContinue |
        ForEach-Object {
            $arch = & $_.FullName -c "import platform; print(platform.machine())" 2>$null
            if ($arch -eq "ARM64") { return $_.FullName }
        }
    return $null
}

if (Test-Path $VenvDir) {
    $venvArch = & (Join-Path $VenvDir "Scripts\python.exe") -c "import platform; print(platform.machine())" 2>$null
    if ($venvArch -eq "ARM64") {
        Write-Warn "Existing .venv is ARM64 -- opencv-python has no win-arm64 wheel and will fail to install here. Recreating from x86-64 Python $Python (venv only, system Python untouched)."
        Remove-Item -Recurse -Force $VenvDir
    }
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

if ($Npu) {
    Write-Info "-Npu passed -- provisioning a SEPARATE native-ARM64 venv (.venv-npu) for real Hexagon NPU sessions ..."
    $npuPythonExe = Get-NativeArm64Python
    if (-not $npuPythonExe) {
        Write-Warn "No native ARM64 Python found under $env:LOCALAPPDATA\Programs\Python. Install arm64 Python from python.org and re-run with -Npu. Skipping NPU venv -- the core app still works via .venv above."
    } else {
        Write-Ok "Native ARM64 Python found: $npuPythonExe"
        $NpuVenvDir = Join-Path $PSScriptRoot ".venv-npu"
        if (-not (Test-Path $NpuVenvDir)) {
            uv venv "$NpuVenvDir" --python $npuPythonExe
            if ($LASTEXITCODE -ne 0) { Write-Warn "uv venv (NPU) failed -- skipping NPU extras." }
        }
        if (Test-Path $NpuVenvDir) {
            foreach ($pkg in @("onnxruntime-qnn>=1.17.0", "onnxruntime-genai", "onnx>=1.16", "numpy>=1.24")) {
                uv pip install --system-certs --python "$NpuVenvDir\Scripts\python.exe" $pkg
                if ($LASTEXITCODE -ne 0) {
                    Write-Warn "Failed to install '$pkg' into .venv-npu -- continuing (optional extra)."
                } else {
                    Write-Ok "Installed $pkg into .venv-npu"
                }
            }
            uv pip install --system-certs --python "$NpuVenvDir\Scripts\python.exe" --no-deps -e "$PSScriptRoot"
            Write-Info "Run real-NPU inference with:"
            Write-Info "  .venv-npu\Scripts\python.exe -m argus.cli run --engine qnn-npu"
        }
    }
}

Write-Info "Provision models (a fresh clone has none -- models/ is gitignored):"
Write-Info "  .venv\Scripts\python.exe scripts\fetch_models.py"
Write-Info "Run the app with:"
Write-Info "  .venv\Scripts\python.exe -m argus.cli run --engine mock --max-ticks 60"
Write-Info "Run the tests with:"
Write-Info "  .venv\Scripts\python.exe -m pytest tests\ -q"
