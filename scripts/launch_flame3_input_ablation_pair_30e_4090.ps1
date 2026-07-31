param(
    [Parameter(Mandatory=$true)][string]$BundleRoot,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Join-Path (Resolve-Path -LiteralPath $BundleRoot).Path "project_support"
$Launcher = Join-Path $ProjectRoot "scripts\launch_flame3_input_ablation_30e_4090.ps1"
if (-not (Test-Path -LiteralPath $Launcher)) {
    throw "Required launcher missing: $Launcher"
}

& $Launcher `
    -BundleRoot $BundleRoot `
    -Mode "rgb" `
    -PythonExe $PythonExe `
    -RunName "flame3_rgb_partial_30e_seed200"

& $Launcher `
    -BundleRoot $BundleRoot `
    -Mode "ir" `
    -PythonExe $PythonExe `
    -RunName "flame3_ir_partial_30e_seed200"
