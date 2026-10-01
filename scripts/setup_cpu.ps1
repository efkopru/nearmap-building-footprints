# Usage: .\scripts\setup_cpu.ps1 [-Python C:\Python314\python.exe]
# Without -Python, the py launcher's Python 3.12, 3.13 or 3.14 is used, in that order.
param([string]$Python)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$supported = '3.12', '3.13', '3.14'
function Test-PyVersion([string]$Version) {
    # Windows PowerShell 5.1 turns a native command's redirected stderr into an error under 'Stop'.
    $ErrorActionPreference = 'Continue'
    & py "-$Version" -c 'pass' 2>$null | Out-Null
    return $LASTEXITCODE -eq 0
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if ($Python) {
        if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
        & $Python -m venv .venv
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $version = $supported | Where-Object { Test-PyVersion $_ } | Select-Object -First 1
        if (-not $version) { throw 'Python 3.12, 3.13 or 3.14 is required. Install one, or pass -Python C:\path\to\python.exe.' }
        & py "-$version" -m venv .venv
    } else {
        throw 'The py launcher was not found. Pass your Python explicitly, e.g. -Python C:\Python314\python.exe.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Creating .venv failed.' }
}
# No quotes inside the code: Windows PowerShell 5.1 strips embedded double quotes from native arguments.
$venvVersion = & '.venv\Scripts\python.exe' -c 'import sys; print(*sys.version_info[:2], sep=chr(46))'
if ($supported -notcontains $venvVersion) {
    throw ".venv uses Python $venvVersion; this project supports $($supported -join ', '). Delete .venv and run this script again."
}
& '.venv\Scripts\python.exe' -m pip install -r requirements-cpu.lock
if ($LASTEXITCODE -ne 0) { throw 'CPU dependency installation failed.' }
& '.venv\Scripts\python.exe' -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
& '.venv\Scripts\python.exe' -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Synthetic validation failed.' }
Write-Output "CPU environment ready with Python $venvVersion. SAM 3 weights, CUDA, and ArcGIS were not installed."
