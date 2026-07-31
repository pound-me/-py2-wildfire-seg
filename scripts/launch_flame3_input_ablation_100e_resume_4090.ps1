param(
    [Parameter(Mandatory=$true)][string]$BundleRoot,
    [Parameter(Mandatory=$true)][ValidateSet("rgb", "ir")][string]$Mode,
    [Parameter(Mandatory=$true)][string]$RunName,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$BundleRoot = (Resolve-Path -LiteralPath $BundleRoot).Path
$ProjectRoot = Join-Path $BundleRoot "project_support"
if ($Mode -eq "rgb") {
    $ConfigName = "pidnet_s_rgb_partial_30e.yaml"
    $ExperimentGroup = "flame3_pidnet_s_rgb_partial"
} else {
    $ConfigName = "pidnet_s_ir_partial_30e.yaml"
    $ExperimentGroup = "flame3_pidnet_s_ir_partial"
}
$Config = Join-Path $ProjectRoot "configs\flame3\$ConfigName"
$Pretrained = Join-Path $BundleRoot "weights\PIDNet_S_ImageNet.pth.tar"
$BatchRecord = Join-Path $ProjectRoot "audit\flame3_4090_batch_final\flame3_4090_batch_preregistered.json"
$RunDirectory = Join-Path $ProjectRoot "experiments\$ExperimentGroup\$RunName"
$ResumeCheckpoint = Join-Path $RunDirectory "last.pth"
$Metrics = Join-Path $RunDirectory "metrics.jsonl"
$Summary = Join-Path $RunDirectory "run_summary.json"

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
    throw "Input ablation continuation requires physical batch 8, got: $BatchSize"
}
$MetricLines = @(Get-Content -LiteralPath $Metrics)
if ($MetricLines.Count -ne 30) {
    throw "Expected exactly 30 completed screening epochs, got: $($MetricLines.Count)"
}
$LastMetric = $MetricLines[-1] | ConvertFrom-Json
if ([int]$LastMetric.epoch -ne 30) {
    throw "Expected screening to end at epoch 30, got: $($LastMetric.epoch)"
}

$LogDirectory = Join-Path $ProjectRoot "logs"
$Transcript = Join-Path $LogDirectory "$RunName.resume_100e.transcript.log"
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
        throw "FLAME3 $Mode 100-epoch continuation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Stop-Transcript | Out-Null
}
