param(
    [string]$PythonVersion = "3.10.11",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu124",
    [string]$PaddlePackage = "paddlepaddle-gpu"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$AssetsRoot = Join-Path $ProjectRoot "portable-assets"
$RuntimeRoot = Join-Path $AssetsRoot "runtime"
$DownloadRoot = Join-Path $ProjectRoot "build\runtime-downloads"
New-Item -ItemType Directory -Force -Path $RuntimeRoot, $DownloadRoot | Out-Null

$PythonZip = Join-Path $DownloadRoot "python-$PythonVersion-embed-amd64.zip"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$GetPip = Join-Path $DownloadRoot "get-pip.py"
if (-not (Test-Path -LiteralPath $PythonZip)) {
    Invoke-WebRequest -Uri $PythonUrl -OutFile $PythonZip
}
if (-not (Test-Path -LiteralPath $GetPip)) {
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPip
}

function New-EmbeddedRuntime {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string[]]$Packages,
        [switch]$InstallTorch
    )
    $Target = [System.IO.Path]::GetFullPath((Join-Path $RuntimeRoot $Name))
    $ResolvedRuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
    if (-not $Target.StartsWith($ResolvedRuntimeRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to replace path outside runtime root: $Target"
    }
    if (Test-Path -LiteralPath $Target) {
        Remove-Item -LiteralPath $Target -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
    Expand-Archive -LiteralPath $PythonZip -DestinationPath $Target
    $Pth = Get-ChildItem -LiteralPath $Target -Filter "python*._pth" | Select-Object -First 1
    if (-not $Pth) {
        throw "Embedded Python ._pth file was not found in $Target"
    }
    $PthContent = @(
        "python310.zip",
        ".",
        "Lib\site-packages",
        "import site"
    )
    Set-Content -LiteralPath $Pth.FullName -Value $PthContent -Encoding ASCII
    New-Item -ItemType Directory -Force -Path (Join-Path $Target "Lib\site-packages") | Out-Null
    $PythonExe = Join-Path $Target "python.exe"
    & $PythonExe $GetPip --no-warn-script-location
    if ($LASTEXITCODE -ne 0) {
        throw "pip bootstrap failed for $Name"
    }
    if ($InstallTorch) {
        & $PythonExe -m pip install --no-warn-script-location --index-url $TorchIndexUrl torch torchaudio
        if ($LASTEXITCODE -ne 0) {
            throw "PyTorch installation failed for $Name"
        }
    }
    & $PythonExe -m pip install --no-warn-script-location @Packages
    if ($LASTEXITCODE -ne 0) {
        throw "Package installation failed for $Name"
    }
}

New-EmbeddedRuntime -Name "demucs" -Packages @("demucs") -InstallTorch
New-EmbeddedRuntime -Name "whisperx" -Packages @("whisperx") -InstallTorch
New-EmbeddedRuntime -Name "whisper-at" -Packages @(
    "https://github.com/YuanGongND/whisper-at/archive/refs/heads/main.zip"
) -InstallTorch
New-EmbeddedRuntime -Name "paddleocr" -Packages @($PaddlePackage, "paddleocr")

Write-Host "Portable Python runtimes prepared in: $RuntimeRoot"
Write-Host "Place tested ffmpeg/llama.cpp/whisper.cpp binaries under portable-assets\tools, then run build_portable.ps1."
