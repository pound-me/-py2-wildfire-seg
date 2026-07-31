param(
    [Parameter(Mandatory=$true)][string]$BundleRoot,
    [string]$PythonExe = "python",
    [string]$RunName = "flame3_fusion_partial_30e_seed200_retry1"
)

$ErrorActionPreference = "Stop"
$BundleRoot = (Resolve-Path -LiteralPath $BundleRoot).Path
$ProjectRoot = Join-Path $BundleRoot "project_support"
$Config = Join-Path $ProjectRoot "configs\flame3\pidnet_s_fusion_partial_30e.yaml"
$Pretrained = Join-Path $BundleRoot "weights\PIDNet_S_ImageNet.pth.tar"
$BatchRecord = Join-Path $ProjectRoot "audit\flame3_4090_batch_final\flame3_4090_batch_preregistered.json"
$RunDirectory = Join-Path $ProjectRoot "experiments\flame3_pidnet_s_fusion_partial\$RunName"
$ResumeCheckpoint = Join-Path $RunDirectory "last.pth"
$Metrics = Join-Path $RunDirectory "metrics.jsonl"
$Summary = Join-Path $RunDirectory "run_summary.json"
$LogDirectory = Join-Path $ProjectRoot "logs"
$Transcript = Join-Path $LogDirectory "$RunName.resume_100e.transcript.log"

foreach ($Path in @(
    $ProjectRoot,
    $Config,
    $Pretrained,
    $BatchRecord,
    $RunDirectory,
    $ResumeCheckpoint,
    $Metrics,
    $Summary
)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path missing: $Path"
    }
}

$GpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader).Trim()
if ($GpuName -notlike "*4090*") {
    throw "Formal FLAME3 training requires RTX 4090, got: $GpuName"
}

$BatchInfo = Get-Content -LiteralPath $BatchRecord -Raw | ConvertFrom-Json
if ($BatchInfo.status -ne "frozen_before_flame3_accuracy_training") {
    throw "Batch preregistration is not frozen"
}
$BatchSize = [int]$BatchInfo.selected_batch
if ($BatchSize -ne 8) {
    throw "This continuation expects the frozen physical batch 8, got: $BatchSize"
}

$MetricLines = @(Get-Content -LiteralPath $Metrics)
if ($MetricLines.Count -ne 30) {
    throw "Expected exactly 30 completed screening epochs, got: $($MetricLines.Count)"
}
$LastMetric = $MetricLines[-1] | ConvertFrom-Json
if ([int]$LastMetric.epoch -ne 30) {
    throw "Expected the screening run to end at epoch 30, got: $($LastMetric.epoch)"
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
Start-Transcript -LiteralPath $Transcript -Append | Out-Null
try {
    & $PythonExe `
        (Join-Path $ProjectRoot "src\train_baseline.py") `
        --config $Config `
        --root-dataset $BundleRoot `
        --pretrained $Pretrained `
        --batch-size $BatchSize `
        --num-workers 4 `
        --epochs 100 `
        --lr-total-epochs 100 `
        --seed 200 `
        --run-name $RunName `
        --device cuda:0 `
        --resume $ResumeCheckpoint `
        --amp

    if ($LASTEXITCODE -ne 0) {
        throw "FLAME3 100-epoch continuation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Stop-Transcript | Out-Null
}
