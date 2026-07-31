param(
    [Parameter(Mandatory=$true)][string]$BundleRoot,
    [Parameter(Mandatory=$true)][ValidateSet("rgb", "ir")][string]$Mode,
    [string]$PythonExe = "python",
    [string]$RunName = ""
)

$ErrorActionPreference = "Stop"
$BundleRoot = (Resolve-Path -LiteralPath $BundleRoot).Path
$ProjectRoot = Join-Path $BundleRoot "project_support"
$ConfigName = if ($Mode -eq "rgb") {
    "pidnet_s_rgb_partial_30e.yaml"
} else {
    "pidnet_s_ir_partial_30e.yaml"
}
$Config = Join-Path $ProjectRoot "configs\flame3\$ConfigName"
$Pretrained = Join-Path $BundleRoot "weights\PIDNet_S_ImageNet.pth.tar"
$BatchRecord = Join-Path $ProjectRoot "audit\flame3_4090_batch_final\flame3_4090_batch_preregistered.json"
if (-not $RunName) {
    $RunName = "flame3_${Mode}_partial_30e_seed200"
}

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
if ($BatchSize -ne 8) {
    throw "Input ablation requires frozen physical batch 8, got: $BatchSize"
}

$LogDirectory = Join-Path $ProjectRoot "logs"
$Transcript = Join-Path $LogDirectory "$RunName.transcript.log"
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
        --epochs 30 `
        --lr-total-epochs 100 `
        --seed 200 `
        --run-name $RunName `
        --device cuda:0 `
        --amp
    if ($LASTEXITCODE -ne 0) {
        throw "FLAME3 $Mode 30-epoch screening failed with exit code $LASTEXITCODE"
    }
}
finally {
    Stop-Transcript | Out-Null
}
