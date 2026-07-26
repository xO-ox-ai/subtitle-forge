param(
    [string]$AssetsDir = "",
    [string]$InnoCompiler = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not $AssetsDir) {
    $AssetsDir = Join-Path $ProjectRoot "portable-assets"
}
$AssetsDir = [System.IO.Path]::GetFullPath($AssetsDir)
$ExpectedAssets = @(
    "runtime\demucs\python.exe",
    "runtime\whisperx\python.exe",
    "runtime\whisper-at\python.exe",
    "runtime\paddleocr\python.exe",
    "tools\ffmpeg\bin\ffmpeg.exe",
    "tools\ffmpeg\bin\ffprobe.exe",
    "tools\llama\llama-server.exe",
    "tools\whisper\whisper-server.exe"
)
foreach ($relative in $ExpectedAssets) {
    $target = Join-Path $AssetsDir $relative
    if (-not (Test-Path -LiteralPath $target)) {
        throw "Portable asset is missing: $target"
    }
}

$CanonicalAssets = Join-Path $ProjectRoot "portable-assets"
if ($AssetsDir -ne [System.IO.Path]::GetFullPath($CanonicalAssets)) {
    throw "Inno Setup source is fixed to $CanonicalAssets. Copy or prepare assets there before building."
}

& (Join-Path $ProjectRoot "build_ui.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "Controller EXE build failed."
}

if (-not $InnoCompiler) {
    $Candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    $InnoCompiler = $Candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
}
if (-not $InnoCompiler -or -not (Test-Path -LiteralPath $InnoCompiler)) {
    throw "Inno Setup 6 was not found. Pass -InnoCompiler with the full path to ISCC.exe."
}

$Version = (Get-Content -LiteralPath (Join-Path $ProjectRoot "VERSION") -Encoding UTF8).Trim()
& $InnoCompiler "/DMyAppVersion=$Version" (Join-Path $ProjectRoot "packaging\SubtitleForge.iss")
if ($LASTEXITCODE -ne 0) {
    throw "Portable installer build failed."
}

Write-Host "Built portable installer in: $(Join-Path $ProjectRoot 'installer-output')"
