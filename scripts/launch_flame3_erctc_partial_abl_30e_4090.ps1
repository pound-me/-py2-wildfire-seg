param(
    [Parameter(Mandatory=$true)][string]$BundleRoot,
    [string]$PythonExe = "python",
    [string]$RunName = "flame3_erctc_partial_abl_30e_seed200"
)

$ErrorActionPreference = "Stop"
$BundleRoot = (Resolve-Path -LiteralPath $BundleRoot).Path
$ProjectRoot = Join-Path $BundleRoot "project_support"
$Config = Join-Path $ProjectRoot "configs\flame3\pidnet_s_erctc_partial_abl_30e.yaml"
$Pretrained = Join-Path $BundleRoot "weights\PIDNet_S_ImageNet.pth.tar"
$BatchRecord = Join-Path $ProjectRoot "audit\flame3_4090_batch_final\flame3_4090_batch_preregistered.json"
$EngineeringCheck = Join-Path $ProjectRoot "audit\flame3_erctc_partial_abl\engineering_check.json"

foreach ($Path in @($ProjectRoot, $Config, $Pretrained, $BatchRecord, $EngineeringCheck)) {
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
    throw "ERCTC+partial-ABL screening requires the frozen physical batch 8, got: $BatchSize"
}
$Engineering = Get-Content -LiteralPath $EngineeringCheck -Raw | ConvertFrom-Json
if ($Engineering.nonfire_union_bg_smoke_swap_max_abs -ne 0.0) {
    throw "Partial ABL Non-fire union engineering check is not exact"
}
if ($Engineering.partial_abl_trainable_parameters -ne 0 -or
    $Engineering.partial_abl_persistent_buffers -ne 0) {
    throw "Partial ABL must remain training-only state-free supervision"
}
if (-not $Engineering.partial_abl_fire_core_images_only -or
    $Engineering.no_fire_only_partial_abl_loss -ne 0.0) {
    throw "Partial ABL must skip images without a supervised Fire core"
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
    throw "FLAME3 ERCTC+partial-ABL 30-epoch screening failed with exit code $LASTEXITCODE"
}
