$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python310 = $null
$launcher = Get-Command py -ErrorAction SilentlyContinue
if ($launcher) {
    try {
        & py -3.10 --version *> $null
        if ($LASTEXITCODE -eq 0) { $python310 = 'py -3.10' }
    } catch { }
}

if (-not $python310) {
    $candidate = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'
    if (Test-Path -LiteralPath $candidate) { $python310 = $candidate }
}

if (-not $python310) {
    throw 'Python 3.10 was not found. Install Python 3.10, then rerun this script.'
}

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if ($python310 -eq 'py -3.10') {
        & py -3.10 -m venv .venv
    } else {
        & $python310 -m venv .venv
    }
}

$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
& $venvPython -m pip install --upgrade pip==24.0 setuptools==59.5.0 wheel
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare pip/setuptools in .venv.' }

& $venvPython -m pip install torch==2.0.1 torchaudio==2.0.2 numpy==1.26.4 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Could not install CPU PyTorch in .venv.' }

& $venvPython -m pip install fairseq==0.12.2 indic-nlp-library==0.92
if ($LASTEXITCODE -ne 0) { throw 'Could not install Fairseq/Indic NLP in .venv.' }

Write-Host ''
Write-Host 'IndicXlit Windows environment is ready.' -ForegroundColor Green
Write-Host 'In VS Code, select .venv\Scripts\python.exe if prompted.'
Write-Host 'Run the "IndicXlit Marathi beam search" launch configuration with F5.'
