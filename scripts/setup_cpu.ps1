$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required. Install it before running this script.' }
}
& '.venv\Scripts\python.exe' -m pip install -r requirements-cpu.lock
if ($LASTEXITCODE -ne 0) { throw 'CPU dependency installation failed.' }
& '.venv\Scripts\python.exe' -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
& '.venv\Scripts\python.exe' -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Synthetic validation failed.' }
Write-Output 'CPU environment ready. SAM 3 weights, CUDA, and ArcGIS were not installed.'
