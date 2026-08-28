[CmdletBinding()]
param(
    [string]$RemoteName = "gdrive",
    [string]$FolderId = "1RfDV984zjw0Y5yfnxtnd7pPQhJpNczt_",
    [string]$LocalRunsRoot = "H:\My Drive\project3_runs",
    [switch]$CopyKaggleSecret
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) {
    throw "Không tìm thấy rclone trong PATH."
}

$remoteRoot = "${RemoteName},root_folder_id=${FolderId}:"
$runNames = @(
    "yolo11m_dms_4class_base",
    "yolo11m_dms_4class_pseudo_finetune",
    "champion_artifacts"
)

# Kiểm tra đúng folder ID từ link, không dựa riêng vào tên thư mục/shortcut.
& rclone lsjson $remoteRoot --max-depth 1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Không truy cập được Google Drive folder ID $FolderId."
}

$existing = @(
    & rclone lsf $remoteRoot --dirs-only --format p |
        ForEach-Object { $_.TrimEnd("/") }
)
foreach ($runName in $runNames) {
    if ($runName -notin $existing) {
        & rclone mkdir "$remoteRoot/$runName"
        if ($LASTEXITCODE -ne 0) {
            throw "Không tạo được thư mục $runName trên Google Drive."
        }
    }
}

if (-not (Test-Path -LiteralPath $LocalRunsRoot -PathType Container)) {
    New-Item -ItemType Directory -Path $LocalRunsRoot -Force | Out-Null
}

Write-Host "Google Drive remote OK: $remoteRoot"
Write-Host "Google Drive Desktop OK: $LocalRunsRoot"

if ($CopyKaggleSecret) {
    # Chỉ xuất section remote được chọn, không lấy credential của remote khác.
    $configText = (& rclone config show $RemoteName | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($configText)) {
        throw "Không đọc được cấu hình rclone remote $RemoteName."
    }
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($configText + "`n"))
    Set-Clipboard -Value $encoded
    Write-Host "Đã copy base64 rclone config vào clipboard. Dán vào Kaggle Secret RCLONE_CONFIG_B64."
    Write-Host "Không dán secret vào notebook, GitHub, Dataset hoặc ảnh chụp màn hình."
}
