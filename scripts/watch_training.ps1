param(
    [string]$RunDir,
    [int]$RefreshSeconds = 5,
    [int]$StaleAfterSeconds = 120
)

$ErrorActionPreference = 'Stop'

function Get-TrainingRun {
    param([string]$RequestedRunDir)

    if ($RequestedRunDir) {
        $resolvedRunDir = Resolve-Path -LiteralPath $RequestedRunDir
        return $resolvedRunDir.Path
    }

    $repoRoot = Split-Path -Parent $PSScriptRoot
    $runsRoot = Join-Path $repoRoot 'runs'
    if (-not (Test-Path -LiteralPath $runsRoot)) {
        throw "Không tìm thấy thư mục runs: $runsRoot"
    }

    $candidate = Get-ChildItem -LiteralPath $runsRoot -Filter 'manifest.json' -File -Recurse |
        ForEach-Object {
            $progressPath = Join-Path $_.DirectoryName 'logs\progress.csv'
            if (Test-Path -LiteralPath $progressPath) {
                Get-Item -LiteralPath $progressPath
            }
        } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $candidate) {
        throw "Không tìm thấy run có manifest.json và logs\progress.csv dưới $runsRoot"
    }

    return $candidate.Directory.Parent.FullName
}

function Get-TrainingStage {
    param($Metadata, [long]$Steps)

    $stages = $Metadata.training_plan.stages
    foreach ($stage in $stages) {
        if ($Steps -lt [long]$stage.end) {
            return $stage.name
        }
    }

    return 'hoàn tất curriculum'
}

if ($RefreshSeconds -lt 1) {
    throw 'RefreshSeconds phải từ 1 trở lên.'
}

$resolvedRunDir = Get-TrainingRun -RequestedRunDir $RunDir
$manifestPath = Join-Path $resolvedRunDir 'manifest.json'
$progressPath = Join-Path $resolvedRunDir 'logs\progress.csv'
$metadataPath = Join-Path $resolvedRunDir 'latest_model.zip.json'

if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Không tìm thấy manifest: $manifestPath"
}
if (-not (Test-Path -LiteralPath $progressPath)) {
    throw "Không tìm thấy progress log: $progressPath"
}

Write-Host "Theo dõi run: $resolvedRunDir"
Write-Host 'Nhấn Ctrl+C để dừng theo dõi.'

while ($true) {
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $progressRows = Import-Csv -LiteralPath $progressPath
        $latestProgress = $progressRows | Select-Object -Last 1

        if (-not $latestProgress) {
            throw 'progress.csv chưa có dòng dữ liệu.'
        }

        $steps = [long]$latestProgress.'time/total_timesteps'
        $requestedSteps = [long]$manifest.requested_timesteps
        $percent = 0
        if ($requestedSteps -gt 0) {
            $percent = [math]::Min(100, [math]::Round(100 * $steps / $requestedSteps, 1))
        }

        $progressTime = (Get-Item -LiteralPath $progressPath).LastWriteTime
        $ageSeconds = [math]::Max(0, ((Get-Date) - $progressTime).TotalSeconds)
        $pythonProcesses = @(Get-Process -Name python, pythonw -ErrorAction SilentlyContinue)
        $status = $manifest.status
        if ($manifest.status -eq 'running') { $status = 'manifest báo running, nhưng không thấy tiến trình Python' }
        if ($manifest.status -eq 'running' -and $ageSeconds -gt $StaleAfterSeconds) { $status = "chưa có log mới trong $([int]$ageSeconds) giây" }
        if ($manifest.status -eq 'running' -and $ageSeconds -le $StaleAfterSeconds -and $pythonProcesses.Count -gt 0) { $status = 'đang train' }

        $stage = 'chưa rõ'
        if (Test-Path -LiteralPath $metadataPath) {
            $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
            $stage = Get-TrainingStage -Metadata $metadata -Steps $steps
        }

        Write-Host "`n--- Cập nhật $(Get-Date -Format 'HH:mm:ss') ---"
        Write-Host "Run: $([System.IO.Path]::GetFileName($resolvedRunDir))"
        Write-Host "Trạng thái: $status"
        Write-Host "Tiến độ: $steps / $requestedSteps bước ($percent%)"
        Write-Host "Giai đoạn: $stage"
        Write-Host "Số update PPO: $($latestProgress.'train/n_updates')"
        Write-Host "Tốc độ gần nhất: $($latestProgress.'time/fps') bước/giây"
        Write-Host "Log cập nhật lúc: $($progressTime.ToString('HH:mm:ss dd/MM/yyyy'))"
        Write-Host "Tự làm mới mỗi $RefreshSeconds giây; nhấn Ctrl+C để thoát."
    }
    catch {
        Write-Host "Không đọc được trạng thái: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    Start-Sleep -Seconds $RefreshSeconds
}
