# Subtitle Forge Desktop UI

桌面版入口：

```powershell
python .\subtitle_forge_ui.py
```

界面包含：

- 环境检测：GPU、内存、外部工具、四套隔离 Python 运行时、本地模型、HF Token 与云端精修配置。
- 制作字幕：递归扫描视频与子目录，复用 `run_all.py` 的素材分类逻辑，并允许逐视频调整执行、OCR、云端精修和完成后清理。
- 模型中心：根据硬件推荐 Qwen 32B/80B，下载 Qwen 与 Whisper.cpp 模型，支持断点续传。
- 设置：自定义工具、模型、运行时和缓存路径；工具目录仅加入当前应用子进程的 `PATH`。
- 账户直达：Hugging Face 注册、Token、pyannote 模型条款，以及 OpenAI 注册/API Key 页面。

Token 不会明文写入配置文件；Windows 下使用当前用户的 DPAPI 加密。未配置 OpenAI-compatible API Key 时，任务会自动加入 `--no-polish`，Step 9 仍可执行本地合并与过滤，但不会请求云端大模型。

## 构建单文件控制器

```powershell
.\build_ui.ps1 -Clean
```

输出为 `dist\SubtitleForge.exe`。该 EXE 内含 UI、Python 解释器和流水线脚本，可连接设置页指定的隔离运行时与工具。

## 构建无需安装 Python 的单个分发包

完整流水线需要 Demucs、WhisperX、Whisper-AT 和 PaddleOCR。它们的依赖栈互相冲突，因此便携版在一个安装 EXE 中携带四套隔离的嵌入式 Python 运行时：

```powershell
.\packaging\prepare_portable_runtime.ps1
```

然后把经过测试的外部二进制放入：

```text
portable-assets/
  runtime/
    demucs/python.exe
    whisperx/python.exe
    whisper-at/python.exe
    paddleocr/python.exe
  tools/
    ffmpeg/bin/ffmpeg.exe
    ffmpeg/bin/ffprobe.exe
    llama/llama-server.exe
    whisper/whisper-server.exe
```

安装 Inno Setup 6 后执行：

```powershell
.\build_portable.ps1
```

最终得到 `installer-output\SubtitleForge-Setup-<版本>.exe`。用户只运行这一个安装包，无需安装 Python 或 pip；模型仍在模型中心按硬件选择后下载，避免把 3–48GB 模型强制塞进安装包。
