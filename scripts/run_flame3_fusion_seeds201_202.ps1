$ErrorActionPreference = "Stop"

$BundleRoot = "D:\qianpengcheng\7.31\flame3_4090_bundle_v1_20260731"
$ProjectRoot = Join-Path $BundleRoot "project_support"
$PythonExe = "C:\Users\Admin\anaconda3\envs\cwx\python.exe"
$Config = Join-Path $ProjectRoot "configs\flame3\pidnet_s_fusion_partial_30e.yaml"
$Pretrained = Join-Path $BundleRoot "weights\PIDNet_S_ImageNet.pth.tar"
$BatchRecord = Join-Path $ProjectRoot "audit\flame3_4090_batch_final\flame3_4090_batch_preregistered.json"
$LogDirectory = Join-Path $ProjectRoot "logs"
$Status = Join-Path $LogDirectory "flame3_fusion_seeds201_202_task.status.json"
$Seeds = @(201, 202)

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
foreach ($Path in @($ProjectRoot, $PythonExe, $Config, $Pretrained, $BatchRecord)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path missing: $Path"
    }
}
$GpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader).Trim()
if ($GpuName -notlike "*4090*") {
    throw "Formal FLAME3 training requires RTX 4090, got: $GpuName"
}
$BatchInfo = Get-Content -LiteralPath $BatchRecord -Raw | ConvertFrom-Json
if ($BatchInfo.status -ne "frozen_before_flame3_accuracy_training" -or
    [int]$BatchInfo.selected_batch -ne 8) {
    throw "Frozen FLAME3 physical batch 8 is not available"
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

$Started = Get-Date
@{
    state = "running"
    started_at = $Started.ToString("o")
    seeds = $Seeds
    current_seed = $null
    completed_seeds = @()
} | ConvertTo-Json | Set-Content -LiteralPath $Status -Encoding UTF8

$CompletedSeeds = @()
try {
    foreach ($Seed in $Seeds) {
        $RunName = "flame3_fusion_partial_100e_seed$Seed"
        $RunDirectory = Join-Path $ProjectRoot "experiments\flame3_pidnet_s_fusion_partial\$RunName"
        if (Test-Path -LiteralPath $RunDirectory) {
            throw "Refusing to overwrite existing run: $RunDirectory"
        }
        $Stdout = Join-Path $LogDirectory "$RunName.stdout.log"
        $Stderr = Join-Path $LogDirectory "$RunName.stderr.log"
        @{
            state = "running"
            started_at = $Started.ToString("o")
            current_seed = $Seed
            completed_seeds = $CompletedSeeds
            stdout = $Stdout
            stderr = $Stderr
        } | ConvertTo-Json | Set-Content -LiteralPath $Status -Encoding UTF8

        & $PythonExe -u `
            (Join-Path $ProjectRoot "src\train_baseline.py") `
            --config $Config `
            --root-dataset $BundleRoot `
            --pretrained $Pretrained `
            --batch-size 8 `
            --num-workers 4 `
            --epochs 100 `
            --lr-total-epochs 100 `
            --seed $Seed `
            --run-name $RunName `
            --device cuda:0 `
            --amp `
            1>> $Stdout `
            2>> $Stderr
        if ($LASTEXITCODE -ne 0) {
            throw "Fusion seed $Seed failed with exit code $LASTEXITCODE"
        }
        $CompletedSeeds += $Seed
    }

    @{
        state = "completed"
        exit_code = 0
        started_at = $Started.ToString("o")
        finished_at = (Get-Date).ToString("o")
        completed_seeds = $CompletedSeeds
    } | ConvertTo-Json | Set-Content -LiteralPath $Status -Encoding UTF8
} catch {
    @{
        state = "failed"
        exit_code = 1
        started_at = $Started.ToString("o")
        finished_at = (Get-Date).ToString("o")
        completed_seeds = $CompletedSeeds
        error = $_.Exception.Message
    } | ConvertTo-Json | Set-Content -LiteralPath $Status -Encoding UTF8
    throw
}
