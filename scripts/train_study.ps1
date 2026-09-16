param([string]$Output = 'runs/study14')
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
$config = Join-Path $PSScriptRoot '../configs/study.yaml'
$Output = [System.IO.Path]::GetFullPath($Output)
Push-Location (Join-Path $PSScriptRoot '..')
try {
    foreach ($phase in @('plan','tune','search','train','test','report')) {
        if ($phase -eq 'tune' -and (Test-Path (Join-Path $Output 'frozen_baselines.json'))) { continue }
        if ($phase -eq 'search' -and (Test-Path (Join-Path $Output 'frozen_learning.json'))) { continue }
        & $python -m fleetrl study $phase --config $config --output $Output
        if ($LASTEXITCODE -ne 0) { throw "Study phase $phase failed. Inspect job manifests before resuming." }
    }
}
finally { Pop-Location }
