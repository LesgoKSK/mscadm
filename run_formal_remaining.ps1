param(
    [int]$WaitForPid = 0,
    [string]$Python = "python",
    [string]$Config = "repro_configs\paper.json"
)

$ErrorActionPreference = "Stop"
$root = "outputs\full_reproduction"
$logRoot = Join-Path $root "pipeline_logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Invoke-Experiment {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$Artifact
    )
    if ($Artifact -and (Test-Path $Artifact)) {
        Write-Host "[$(Get-Date -Format o)] SKIP $Name -> $Artifact"
        return
    }
    Write-Host "[$(Get-Date -Format o)] START $Name"
    $log = Join-Path $logRoot "$Name.log"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    if ($Artifact -and -not (Test-Path $Artifact)) {
        throw "$Name finished without expected artifact: $Artifact"
    }
    Write-Host "[$(Get-Date -Format o)] DONE $Name"
}

if ($WaitForPid -gt 0) {
    Write-Host "[$(Get-Date -Format o)] Waiting for existing QRGBM PID $WaitForPid"
    while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
}

$qrgbmAll = Join-Path $root "scenarios\qrgbm_test.npz"
if (-not (Test-Path $qrgbmAll)) {
    throw "All-zone QRGBM process ended without $qrgbmAll"
}

Invoke-Experiment "sampling_steps" @(
    "-m", "repro_scripts.sampling_steps", "--config", $Config,
    "--steps", "10", "20", "50", "100", "250", "--scenarios", "100"
) (Join-Path $root "scenarios\sampling_steps\mscadm_250steps.npz")

Invoke-Experiment "zone1_qrgbm" @(
    "-m", "repro_scripts.qrgbm_gpu_single_zone", "--config", $Config,
    "--zone", "1", "--scenarios", "100"
) (Join-Path $root "scenarios\single_zone\zone1_qrgbm.npz")

Invoke-Experiment "zone1_ddpm" @(
    "-m", "repro_scripts.single_zone", "--config", $Config,
    "--model", "ddpm", "--zone", "1", "--scenarios", "100"
) (Join-Path $root "scenarios\single_zone\zone1_ddpm.npz")

Invoke-Experiment "zone1_mscadm" @(
    "-m", "repro_scripts.single_zone", "--config", $Config,
    "--model", "mscadm", "--zone", "1", "--scenarios", "100"
) (Join-Path $root "scenarios\single_zone\zone1_mscadm.npz")

$fullAblation = Join-Path $root "scenarios\ablations\zone1_full.npz"
if (-not (Test-Path $fullAblation)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $fullAblation) | Out-Null
    Copy-Item -LiteralPath (Join-Path $root "scenarios\single_zone\zone1_mscadm.npz") -Destination $fullAblation
}

foreach ($variant in @("no_ce", "no_adaln", "no_lv", "no_rcm")) {
    Invoke-Experiment "ablation_$variant" @(
        "-m", "repro_scripts.ablate", "--config", $Config,
        "--variant", $variant, "--zone", "1", "--scenarios", "100"
    ) (Join-Path $root "scenarios\ablations\zone1_$variant.npz")
}

Invoke-Experiment "suc_all_models" @(
    "-m", "repro_scripts.suc", "--root", $root,
    "--models", "qrgbm", "wgan_reference", "vae_reference", "nf", "ddpm", "mscadm",
    "--days", "7", "--clusters", "10", "--seed", "0", "--time-limit", "120"
) ""

Invoke-Experiment "tables" @("-m", "repro_scripts.tables", "--root", $root) ""
Invoke-Experiment "figures" @("-m", "repro_scripts.figures", "--root", $root) ""
Invoke-Experiment "manifest" @(
    "-m", "repro_scripts.manifest", "--root", $root, "--config", $Config
) ""

Write-Host "[$(Get-Date -Format o)] FORMAL PIPELINE COMPLETE"
