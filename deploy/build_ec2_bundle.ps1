$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildRoot = Join-Path $root "_ec2_bundle"
$bundleDir = Join-Path $buildRoot "tesis_v22"
$zipPath = Join-Path $root "output\sicher_ec2_bundle.zip"

if (Test-Path -LiteralPath $buildRoot) {
  Remove-Item -LiteralPath $buildRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $bundleDir | Out-Null
New-Item -ItemType Directory -Path (Split-Path $zipPath -Parent) -Force | Out-Null

$items = @(
  "app",
  "deploy",
  "config",
  "preprocess_cctv_pose.py",
  "train_validate_pose_pipeline.py",
  "requirements-ec2.txt",
  "README.md",
  "users.db",
  "yolo11s-pose.pt",
  "output\pipeline_25fps\hard_negative_mining\models"
)

foreach ($item in $items) {
  $src = Join-Path $root $item
  if (-not (Test-Path -LiteralPath $src)) {
    Write-Warning "No existe: $item"
    continue
  }
  $dst = Join-Path $bundleDir $item
  New-Item -ItemType Directory -Path (Split-Path $dst -Parent) -Force | Out-Null
  Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
}

Get-ChildItem -Path $bundleDir -Recurse -Force -Directory -Include "__pycache__", ".pytest_cache" |
  Remove-Item -Recurse -Force
Get-ChildItem -Path $bundleDir -Recurse -Force -File -Include "*.pyc", "*.log", "*.tmp" |
  Remove-Item -Force

if (Test-Path -LiteralPath $zipPath) {
  Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path (Join-Path $bundleDir "*") -DestinationPath $zipPath -Force

Remove-Item -LiteralPath $buildRoot -Recurse -Force

Write-Host "Bundle creado: $zipPath"
