param(
    [Parameter(Mandatory=$true)][string]$BundleRoot,
    [string]$PythonExe = "python",
    [string]$RunName = "flame3_fusion_partial_30e_seed200"
)

$ErrorActionPreference = "Stop"
$BundleRoot = (Resolve-Path -LiteralPath $BundleRoot).Path
$ProjectRoot = Join-Path $BundleRoot "project_support"
$Config = Join-Path $ProjectRoot "configs\flame3\pidnet_s_fusion_partial_30e.yaml"
$Pretrained = Join-Path $BundleRoot "weights\PIDNet_S_ImageNet.pth.tar"
$BatchRecord = Join-Path $ProjectRoot "audit\flame3_4090_batch_final\flame3_4090_batch_preregistered.json"

foreach ($Path in @($ProjectRoot, $Config, $Pretrained, $BatchRecord)) {
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
if ($BatchSize -notin @(4, 8)) {
    throw "Unexpected preregistered batch: $BatchSize"
}

& $PythonExe `
    (Join-Path $ProjectRoot "src\train_baseline.py") `
    --config $Config `
    --root-dataset $BundleRoot `
    --pretrained $Pretrained `
    --batch-size $BatchSize `
    --num-workers 4 `
    --epochs 30 `
    --lr-total-epochs 100 `
    --seed 200 `
    --run-name $RunName `
    --device cuda:0 `
    --amp

if ($LASTEXITCODE -ne 0) {
    throw "FLAME3 30-epoch training failed with exit code $LASTEXITCODE"
}
