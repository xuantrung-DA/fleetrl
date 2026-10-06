param([string]$Output = 'runs/study14')
$ErrorActionPreference = 'Stop'
$python = Join-Path $env:USERPROFILE '.conda/envs/fleetrl-env/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Cannot find fleetrl-env or the repository virtual environment.'
}
$config = Join-Path $PSScriptRoot '../configs/study.yaml'
$Output = [System.IO.Path]::GetFullPath($Output)
$resourceScript = Join-Path $PSScriptRoot 'study_resources.py'
$mutex = New-Object System.Threading.Mutex($false, 'Local\FleetRLStudy14')
if (-not $mutex.WaitOne(0)) {
    throw 'A FleetRL study runner is already active.'
}
Push-Location (Join-Path $PSScriptRoot '..')
try {
    foreach ($phase in @('plan','tune','search','train','test','report')) {
        if ($phase -eq 'tune' -and (Test-Path (Join-Path $Output 'frozen_baselines.json'))) { continue }
        if ($phase -eq 'search' -and (Test-Path (Join-Path $Output 'frozen_learning.json'))) { continue }
        if ($phase -eq 'search' -or $phase -eq 'train') {
            $planPath = Join-Path $Output 'plan.json'
            if (-not (Test-Path -LiteralPath $planPath)) { throw "Missing study plan: $planPath" }
            $plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
            $jobCount = if ($phase -eq 'search') { [int]$plan.search_jobs } else { [int]$plan.training_jobs }
            for ($index = 0; $index -lt $jobCount; $index++) {
                & $python $resourceScript --output $Output
                if ($LASTEXITCODE -ne 0) { throw "Resource check failed before $phase job." }
                Write-Host "[PHASE] $phase job $($index + 1)/$jobCount"
                & $python -m fleetrl study $phase --config $config --output $Output --limit 1
                if ($LASTEXITCODE -ne 0) { throw "Study phase $phase failed. Inspect job manifests before resuming." }
                & $python (Join-Path $PSScriptRoot 'study_cleanup.py') --output $Output
                if ($LASTEXITCODE -ne 0) { throw "Checkpoint cleanup failed after $phase job." }
                if ($phase -eq 'search') {
                    $selectionPath = Join-Path $Output 'selected_learning.json'
                    if (Test-Path -LiteralPath $selectionPath) {
                        $selection = Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
                        if ($selection.complete) { break }
                    }
                }
                else {
                    $states = Get-ChildItem -Path (Join-Path $Output 'jobs/train_*/job.json') -ErrorAction SilentlyContinue
                    $complete = @($states | Where-Object {
                        (Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json).status -eq 'complete'
                    }).Count
                    if ($complete -eq $jobCount) { break }
                }
            }
        }
        if ($phase -ne 'plan') {
            & $python $resourceScript --output $Output
            if ($LASTEXITCODE -ne 0) { throw "Resource check failed before $phase." }
        }
        Write-Host "[PHASE] $phase"
        & $python -m fleetrl study $phase --config $config --output $Output
        if ($LASTEXITCODE -ne 0) { throw "Study phase $phase failed. Inspect job manifests before resuming." }
    }
}
finally {
    Pop-Location
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
