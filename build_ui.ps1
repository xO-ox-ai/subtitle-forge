param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if ($Clean) {
    foreach ($relative in @("build", "dist")) {
        $target = Join-Path $ProjectRoot $relative
        $resolvedRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
        $resolvedTarget = [System.IO.Path]::GetFullPath($target)
        if (-not $resolvedTarget.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to clean path outside project: $resolvedTarget"
        }
        if (Test-Path -LiteralPath $resolvedTarget) {
            Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
        }
    }
}

python -m pip install --disable-pip-version-check "pyinstaller>=6.10,<7"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller installation failed."
}

python -m PyInstaller --noconfirm subtitle_forge_ui.spec
if ($LASTEXITCODE -ne 0) {
    throw "Subtitle Forge build failed."
}

$ExePath = Join-Path $ProjectRoot "dist\SubtitleForge.exe"
if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "Expected EXE was not created: $ExePath"
}

Write-Host "Built: $ExePath"
