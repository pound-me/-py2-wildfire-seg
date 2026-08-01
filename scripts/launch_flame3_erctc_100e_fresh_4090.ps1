param(
    [Parameter(Mandatory=$true)][string]$BundleRoot,
    [string]$PythonExe = "python",
    [Parameter(Mandatory=$true)][ValidateSet(201, 202)][int]$Seed,
    [string]$RunName = ""
)

$ErrorActionPreference = "Stop"
$BundleRoot = (Resolve-Path -LiteralPath $BundleRoot).Path
$ProjectRoot = Join-Path $BundleRoot "project_support"
$Config = Join-Path $ProjectRoot "configs\flame3\pidnet_s_erctc_partial_30e.yaml"
$Pretrained = Join-Path $BundleRoot "weights\PIDNet_S_ImageNet.pth.tar"
$BatchRecord = Join-Path $ProjectRoot "audit\flame3_4090_batch_final\flame3_4090_batch_preregistered.json"

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "flame3_erctc_partial_100e_seed$Seed"
}
foreach ($Path in @($ProjectRoot, $Config, $Pretrained, $BatchRecord, $PythonExe)) {
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
    throw "ERCTC three-seed confirmation requires physical batch 8, got: $BatchSize"
}

$RunDirectory = Join-Path $ProjectRoot "experiments\flame3_pidnet_s_erctc_partial\$RunName"
if (Test-Path -LiteralPath $RunDirectory) {
    throw "Refusing to overwrite an existing run directory: $RunDirectory"
}
$ActiveTraining = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -like "python*" -and
            $_.CommandLine -like "*train_baseline.py*"
        }
)
if ($ActiveTraining.Count -ne 0) {
    throw "A training process is already active: $($ActiveTraining.Count)"
}

& $PythonExe `
    (Join-Path $ProjectRoot "src\train_baseline.py") `
    --config $Config `
    --root-dataset $BundleRoot `
    --pretrained $Pretrained `
    --batch-size $BatchSize `
    --num-workers 4 `
    --epochs 100 `
    --lr-total-epochs 100 `
    --seed $Seed `
    --run-name $RunName `
    --device cuda:0 `
    --amp

if ($LASTEXITCODE -ne 0) {
    throw "FLAME3 ERCTC seed $Seed failed with exit code $LASTEXITCODE"
}
