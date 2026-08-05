param(
    [Parameter(Mandatory=$true)][string]$BundleRoot,
    [string]$PythonExe = "python",
    [ValidateSet("mrff", "mrff_nts")][string]$Variant = "mrff",
    [ValidateSet(30, 50, 100)][int]$TargetEpochs = 30,
    [int[]]$Seeds = @(200, 201, 202)
)

$ErrorActionPreference = "Stop"
$BundleRoot = (Resolve-Path -LiteralPath $BundleRoot).Path
$ProjectRoot = Join-Path $BundleRoot "project_support"
$ConfigName = if ($Variant -eq "mrff") {
    "pidnet_s_mrff_partial_30e.yaml"
} else {
    "pidnet_s_mrff_nts_partial_30e.yaml"
}
$Config = Join-Path $ProjectRoot "configs\flame3\$ConfigName"
$Pretrained = Join-Path $BundleRoot "weights\PIDNet_S_ImageNet.pth.tar"
$BatchRecord = Join-Path $ProjectRoot "audit\flame3_4090_batch_final\flame3_4090_batch_preregistered.json"

foreach ($Path in @($ProjectRoot, $Config, $Pretrained, $BatchRecord)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path missing: $Path"
    }
}
$GpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader).Trim()
if ($GpuName -notlike "*4090*") {
    throw "Formal FLAME3 MRFF training requires RTX 4090, got: $GpuName"
}
$BatchInfo = Get-Content -LiteralPath $BatchRecord -Raw | ConvertFrom-Json
if ($BatchInfo.status -ne "frozen_before_flame3_accuracy_training") {
    throw "Batch preregistration is not frozen"
}
$BatchSize = [int]$BatchInfo.selected_batch
if ($BatchSize -ne 8) {
    throw "MRFF preregistration requires physical batch 8, got: $BatchSize"
}
$ExperimentGroup = if ($Variant -eq "mrff") {
    "flame3_pidnet_s_mrff_partial"
} else {
    "flame3_pidnet_s_mrff_nts_partial"
}
foreach ($Seed in $Seeds) {
    if ($Seed -notin @(200, 201, 202)) {
        throw "Only preregistered seeds 200, 201 and 202 are allowed."
    }
    $RunName = "flame3_${Variant}_partial_30e_seed${Seed}"
    $RunDirectory = Join-Path $ProjectRoot "experiments\$ExperimentGroup\$RunName"
    $Arguments = @(
        (Join-Path $ProjectRoot "src\train_baseline.py"),
        "--config", $Config,
        "--root-dataset", $BundleRoot,
        "--pretrained", $Pretrained,
        "--batch-size", "$BatchSize",
        "--num-workers", "4",
        "--epochs", "$TargetEpochs",
        "--lr-total-epochs", "100",
        "--seed", "$Seed",
        "--run-name", $RunName,
        "--device", "cuda:0",
        "--amp"
    )
    if ($TargetEpochs -gt 30) {
        $Resume = Join-Path $RunDirectory "last.pth"
        if (-not (Test-Path -LiteralPath $Resume)) {
            throw "Exact resume checkpoint missing: $Resume"
        }
        $Arguments += @("--resume", $Resume)
    } elseif (Test-Path -LiteralPath $RunDirectory) {
        $Existing = @(Get-ChildItem -LiteralPath $RunDirectory -Force)
        if ($Existing.Count -gt 0) {
            throw "Refusing to overwrite existing MRFF run: $RunDirectory"
        }
    }

    Write-Host "Starting $Variant seed $Seed to epoch $TargetEpochs"
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Variant seed $Seed failed with exit code $LASTEXITCODE"
    }
}
