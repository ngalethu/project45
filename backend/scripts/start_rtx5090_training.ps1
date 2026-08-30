[CmdletBinding()]
param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$RunsRoot = "",
    [double]$TotalHours = 5.5,
    [double]$BaseHours = 4.5,
    [double]$FineHours = 0.5,
    [switch]$AllowWithoutRemoteCheckpoint,
    [switch]$LaunchJupyter
)

$ErrorActionPreference = "Stop"

$project = (Resolve-Path -LiteralPath $ProjectRoot).Path
if ([string]::IsNullOrWhiteSpace($RunsRoot)) {
    $RunsRoot = Join-Path $project "backend\outputs\runs_dms_rtx5090"
}
$runs = [IO.Path]::GetFullPath($RunsRoot)

$requiredFiles = @(
    (Join-Path $project "backend\driver_behavior_yolo11m_mediapipe_minimal_stable.ipynb"),
    (Join-Path $project "backend\yolo11m.pt"),
    (Join-Path $project "data\processed\dms_yolo_3class_v3_curated\dms_dataset.yaml")
)
$missing = @($requiredFiles | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missing.Count -gt 0) {
    throw "Thiếu file bắt buộc: $($missing -join ', ')"
}
if ($BaseHours + $FineHours -gt $TotalHours - 0.25) {
    throw "Phải dành ít nhất 0.25 giờ cho test/export/upload."
}

New-Item -ItemType Directory -Path $runs -Force | Out-Null

$env:DMS_PROJECT_ROOT = $project
$env:DMS_RUNS_ROOT = $runs
$env:DMS_TOTAL_BUDGET_HOURS = $TotalHours.ToString([Globalization.CultureInfo]::InvariantCulture)
$env:DMS_BASE_TRAIN_HOURS = $BaseHours.ToString([Globalization.CultureInfo]::InvariantCulture)
$env:DMS_FINE_TRAIN_HOURS = $FineHours.ToString([Globalization.CultureInfo]::InvariantCulture)
$env:DMS_REQUIRE_REMOTE_CHECKPOINT = if ($AllowWithoutRemoteCheckpoint) { "0" } else { "1" }
$env:RCLONE_DRIVE_ROOT_FOLDER_ID = "1RfDV984zjw0Y5yfnxtnd7pPQhJpNczt_"

$probe = @'
import sys
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
assert "RTX 5090" in name.upper(), f"Expected RTX 5090, got {name}"
assert capability >= (12, 0), capability
assert "sm_120" in arches, (
    f"PyTorch {torch.__version__} lacks sm_120. Install a CUDA 12.8+ wheel. arches={arches}"
)
x = torch.randn((256, 256), device="cuda")
value = float((x @ x).mean().item())
print(f"RTX5090 CUDA OK | torch={torch.__version__} | gpu={name} | capability={capability} | probe={value:.6f}")
'@
$probe | python -
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch/CUDA probe thất bại. Cài torch CUDA 13.0 rồi chạy lại script."
}

if (-not $AllowWithoutRemoteCheckpoint) {
    if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) {
        throw "Thiếu rclone. Cài/cấu hình remote gdrive: hoặc dùng -AllowWithoutRemoteCheckpoint với NAS persistent."
    }
    $temporaryRcloneConfig = $null
    try {
        if (-not [string]::IsNullOrWhiteSpace($env:RCLONE_CONFIG_B64)) {
            $temporaryRcloneConfig = Join-Path ([IO.Path]::GetTempPath()) ("dms-rclone-" + [guid]::NewGuid().ToString("N") + ".conf")
            try {
                $configBytes = [Convert]::FromBase64String($env:RCLONE_CONFIG_B64.Trim())
            }
            catch {
                throw "RCLONE_CONFIG_B64 không phải base64 hợp lệ."
            }
            [IO.File]::WriteAllBytes($temporaryRcloneConfig, $configBytes)
            $env:RCLONE_CONFIG = $temporaryRcloneConfig
        }
        & rclone lsf "gdrive:" --dirs-only --max-depth 1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Không truy cập được gdrive:. Không bắt đầu train."
        }
    }
    finally {
        if ($temporaryRcloneConfig -and (Test-Path -LiteralPath $temporaryRcloneConfig)) {
            Remove-Item -LiteralPath $temporaryRcloneConfig -Force
            Remove-Item Env:RCLONE_CONFIG -ErrorAction SilentlyContinue
        }
    }
}

Write-Host "RTX 5090 preflight OK"
Write-Host "Project: $project"
Write-Host "Runs: $runs"
Write-Host "Budget: total=$TotalHours h, base=$BaseHours h, fine=$FineHours h"
Write-Host "Notebook: backend\driver_behavior_yolo11m_mediapipe_minimal_stable.ipynb"

if ($LaunchJupyter) {
    Push-Location (Join-Path $project "backend")
    try {
        & jupyter lab "driver_behavior_yolo11m_mediapipe_minimal_stable.ipynb"
    }
    finally {
        Pop-Location
    }
}
