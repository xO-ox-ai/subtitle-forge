# Subtitle Forge

`subtitle-forge` 是一套用于生成中英双语 ASS 字幕的本地流水线。它可以从视频、内嵌字幕、外部单语字幕或已有双语 ASS 出发，完成音频转写、说话人分离、音乐/歌词识别、OCR 屏幕文字翻译、本地大模型翻译、文化注解、ASS 渲染，以及后端大模型最终调优。

目标输出是可直接播放的中英双语 `.ass` 字幕。

当前版本：`1.0.1`。`VERSION` 是版本号的唯一来源；以后每次向远端推送前，将最后一位补丁版本加 1，并在同一提交中同步这里显示的版本号。可运行 `python .\run_all.py --version` 查看当前版本。

## 主入口

在视频或字幕所在目录运行：

```powershell
python .\run_all.py .
```

本项目最便捷的使用方式，是先将仓库拉取到本地，再让 Codex 等具备终端和文件操作能力的大模型代理自动检查并安装运行环境、识别当前素材类型、调用对应脚本生成字幕，最后审查 ASS 成品并按影片语境补译、纠错和完善文化注解。

入口会递归扫描素材目录及其子文件夹，再按每个视频实际拥有的字幕来源自动分批：

| 素材状态 | 自动流程 |
| --- | --- |
| 内嵌英文（文本或 PGS/DVD/DVB 位图）+ 中文文本 | Step 0 优先选择英文 SDH 和简体中文轨，提取并按英文时间轴合并；Step 6 OCR；Step 7 只翻译缺失的中文与 OCR；Step 8 渲染；Step 9 只调优 Qwen/OCR 译文 |
| 只有内嵌英文文本字幕 | Step 0 提取并清理英文 SDH；Step 6 OCR；Step 7 翻译全部缺失中文；Step 8 渲染；Step 9 调优 Qwen/OCR 译文 |
| 只有内嵌英文 PGS/DVD/DVB 位图字幕 | Step 0 调用 Subtitle Edit nOCR 识别，必要时回退 PaddleOCR，再做 SDH 清理；后续与“只有英文文本字幕”相同 |
| 英文可用，但中文只有位图字幕 | 当前不把中文位图轨作为可合并中文；按“只有英文字幕”批次处理，从英文重新翻译中文 |
| 有外部单语字幕 | 导入字幕，Step 6 OCR，Step 7 本地翻译，Step 8 渲染，Step 9 调优 Qwen/OCR 译文 |
| 没有可用字幕 | Step 1 到 Step 8 完整音频识别、OCR 与翻译流程，再跑 Step 9 |
| 已有普通双语 ASS | 只跑 Step 9，允许调优全部中文 |
| 已由本流水线生成的内嵌双语 ASS | 不重复提取；只跑受保护的 Step 9，原有中文不参与调优 |

内嵌双语合并以英文 SDH 时间轴为准，容忍中英文几十毫秒级偏差，也能把同一句被拆成多条的英文重新合并后匹配中文。纯音效、无意义说话人标签等 SDH 冗余会删除；歌曲、吟唱和咒语仍按 SDH 标记保留。内嵌中文缺失的歌曲或咒语会进入 Step 7 翻译，并可优先复用 `embedded_song_translations.json` 中的人工校订译文。

自动分流下，内嵌双语、内嵌英文和外部单语批次默认也执行 OCR，用于补充片名、地点、招牌等画面文字。只想处理新增的非 OCR 批次时可重复使用 `--target-stem` 并加 `--skip-ocr`：

```powershell
python .\run_all.py . `
  --target-stem "Movie.With.Embedded.Bilingual" `
  --target-stem "Movie.With.English.Only" `
  --skip-ocr
```

默认生成 Step 10 质量报告并保留中间文件。Step 11 清理不会默认执行，需要显式传入 `--cleanup`。

默认工作目录为项目下的 `temp/`。OCR JSON、翻译缓存、Step 9 模型缓存、质量报告、运行日志、状态文件、流程清单和其他中间结果都写入这里；正式 `.ass` 仍输出到素材同目录。OCR 抽帧等短期工作文件建立在 `temp/sub_ocr_*/` 中，并在单集处理完成或抛出异常时自动删除。流水线还会把 `TMP`、`TEMP`、`TMPDIR` 和 Python `tempfile` 统一指向项目 `temp/`，因此自身及其子进程不再使用系统 `%TEMP%`。如需隔离不同任务，可以通过 `--work-dir` 改写步骤数据目录；全局通用临时目录仍由 `SUB_TEMP_DIR` 控制，默认就是项目 `temp/`。

只有英文 SDH SRT、没有视频时，可先导入为 Step 7 的字幕 JSON，再按原流程翻译、生成文化注解并渲染 ASS：

```powershell
python .\import_english_srt_to_json.py . --work-dir temp --source '.\Movie.SDH.srt' --force
python .\step07_qwen_all.py . --work-dir temp --source embedded --target-stem 'Movie.SDH' --qwen-profile 80b --dialogue-batch-size 4 --dialogue-context-size 4 --force-dialogue
python .\step08_ass_render.py . --work-dir temp --target-stem 'Movie.SDH'
```

该导入器会删除纯音效和说话人标签，保留真正的画外对白；背景音乐标签不会把对白误标成歌词，斜体外语录音也不会仅因斜体而误标成吟唱。没有视频时不运行 OCR 或画面对照。

## 环境准备

本项目优先面向 Windows + PowerShell。建议使用 Python 3.10 系列环境，并按不同工具拆分虚拟环境，因为 Demucs、WhisperX/pyannote、Whisper-AT、PaddleOCR 的依赖栈经常互相冲突。

需要提前准备的命令行工具：

- `ffmpeg.exe` 和 `ffprobe.exe`
- llama.cpp 的 `llama-server.exe`，用于 Step 7 本地 Qwen 翻译
- whisper.cpp 的 `whisper-server.exe`，用于 Step 3 英文转写
- `codex.exe`，或一个 OpenAI-compatible API，用于 Step 9 后端调优
- 可选的 Subtitle Edit `seconv.exe`，用于识别 PGS/DVD/DVB 位图内嵌英文字幕；未配置时会下载到工作目录

脚本不会再假定工具安装在固定盘符。外部程序优先从当前 `PATH` 查找；先确认这些命令可以直接执行：

```powershell
Get-Command python
Get-Command ffmpeg.exe
Get-Command ffprobe.exe
Get-Command llama-server.exe
Get-Command whisper-server.exe
Get-Command codex.exe
```

解析顺序为：对应的可执行文件环境变量（如 `FFMPEG_EXE`）→ 当前 `PATH` → 项目相对兜底目录。如果不想修改系统 `PATH`，也可以把工具放进项目相对目录：

- `tools/ffmpeg/bin/ffmpeg.exe` 与 `ffprobe.exe`
- `tools/llama/llama-server.exe`
- `tools/whisper/whisper-server.exe`

```powershell
$env:PATH = ".\tools\whisper;.\tools\llama;.\tools\ffmpeg\bin;$env:PATH"
```

所有路径型环境变量都允许使用相对路径，并统一相对于项目根目录解析。只有确实需要覆盖 `PATH` 结果时才设置对应变量。

Python 默认使用 `PATH` 中的 `python`。需要隔离依赖时，推荐把环境放在项目的 `.venvs/` 下；如果相对环境不存在，脚本也会自动查找 PATH Python 同级的 `demucs/`、`whisperx/`、`whisper-at/`、`paddleocr/` 环境，不需要写死盘符：

```powershell
$env:SUB_DEMUCS_SCRIPTS = ".venvs\demucs\Scripts"
$env:SUB_WHISPERX_SCRIPTS = ".venvs\whisperx\Scripts"
$env:SUB_WHISPER_AT_SCRIPTS = ".venvs\whisper-at\Scripts"
$env:SUB_PADDLEOCR_SCRIPTS = ".venvs\paddleocr\Scripts"
```

未指定时，模型与 Python 缓存默认写入项目的 `.cache/`。如需覆盖，也建议继续使用项目相对路径：

```powershell
$env:HF_HOME = ".cache\huggingface"
$env:HF_HUB_CACHE = ".cache\huggingface\hub"
$env:TORCH_HOME = ".cache\torch"
$env:PIP_CACHE_DIR = ".cache\pip"
$env:PIP_FIND_LINKS = ".\wheels"
```

项目会自动设置常见 UTF-8 相关变量；如果想提前固定，也可以这样写：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## 需要提前准备的模型

Step 7 本地翻译模型：

- 默认配置：`80b`，模型别名 `qwen3-next-80b-a3b-instruct`
- 默认 GGUF 文件名：`Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf`
- 低内存回退配置：`32b`，使用 `Qwen3-32B-Q4_K_M.gguf`
- 模型会依次从 `llama-server.exe` 同目录、`tools/llama/`、`models/` 查找；也可分别设置 `QWEN_80B_GGUF`、`QWEN_32B_GGUF`

```powershell
$env:QWEN_PROFILE = "80b"
$env:QWEN_80B_GGUF = ".\models\Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"
$env:QWEN_32B_GGUF = ".\models\Qwen3-32B-Q4_K_M.gguf"
$env:QWEN_BASE_URL = "http://127.0.0.1:8080/v1"
```

也可以每次运行时选择，80B 启动失败或内存不足时无需改代码：

```powershell
python .\run_all.py . --qwen-profile 80b
python .\run_all.py . --qwen-profile 32b
python .\step07_qwen_all.py . --qwen-profile 32b
```

Step 7 只会复用别名与所选配置一致的现有服务。若 80B 服务处理中途退出，本次文件不会以英文原文冒充译文落盘，流程会明确失败；改用 `--qwen-profile 32b` 重跑即可沿用已完成文件继续处理。

Step 3 Whisper.cpp 转写模型：

- 默认文件名：`ggml-large-v3.bin`
- 默认相对路径：`models/ggml-large-v3.bin`
- 也可以放在 `whisper-server.exe` 同目录，或设置相对的 `WHISPER_MODEL` / `WHISPER_CPP_MODEL`

```powershell
$env:WHISPER_MODEL = ".\models\ggml-large-v3.bin"
$env:WHISPER_LANGUAGE = "en"
$env:WHISPER_SERVER_PORT = "8091"
```

Step 4 说话人分离模型：

- 默认模型：`pyannote/speaker-diarization-3.1`
- 如果模型没有提前缓存，或 Hugging Face 模型需要授权，需要设置 `HF_TOKEN`

```powershell
$env:DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
$env:HF_TOKEN = "hf_..."
```

Step 5 音乐/歌词识别：

- 脚本会加载 Whisper-AT `large-v2`
- 缓存目录可用 `WHISPER_AT_CACHE` 指定

```powershell
$env:WHISPER_AT_CACHE = ".cache\huggingface\whisper-at"
```

Step 6 OCR：

- 默认 OCR 版本：`PP-OCRv6`
- 默认语言：`en`
- 默认设备：`gpu`

```powershell
$env:OCR_LANG = "en"
$env:PADDLEOCR_DEVICE = "gpu"
$env:PADDLEOCR_VERSION = "PP-OCRv6"
$env:OCR_MIN_CONFIDENCE = "0.45"
```

这里的 Step 6 OCR 指视频画面文字。Step 0 遇到 PGS、DVD 或 DVB 位图英文字幕时，会另外调用 Subtitle Edit nOCR，必要时回退 PaddleOCR；可用 `SUB_SECONV_EXE` 指定现有的 `seconv.exe`，否则所需程序、Latin nOCR 数据库和英文词典会下载到 `temp/tools/`。

## 常用参数和硬件建议

Qwen / llama.cpp 参数。默认配置已按当前 4090 24GB + 64GB 内存主机测试：

```powershell
$env:QWEN_PROFILE = "80b"
$env:QWEN_80B_CTX_SIZE = "8192"
$env:QWEN_80B_BATCH_SIZE = "1024"
$env:QWEN_80B_UBATCH_SIZE = "256"
$env:QWEN_80B_THREADS = "16"
$env:QWEN_80B_THREADS_BATCH = "16"
$env:QWEN_80B_GPU_LAYERS = "22"
$env:QWEN_80B_FLASH_ATTN = "on"
$env:QWEN_80B_CACHE_TYPE_K = "q8_0"
$env:QWEN_80B_CACHE_TYPE_V = "q8_0"
$env:QWEN_80B_CACHE_RAM = "2048"
$env:QWEN_80B_REASONING = "off"
```

这些参数的含义：

- 配置项使用 `QWEN_80B_*` 或 `QWEN_32B_*` 前缀，互不污染。32B 仍兼容旧的通用 `QWEN_*` 设置；80B 不继承旧的 `QWEN_GPU_LAYERS=all`，防止误把 45GB 模型全量卸载到 24GB 显存。
- `*_CTX_SIZE`：上下文长度。8192 足够容纳当前每批 4 条对白及前后各 4 条只读上下文；调大只会增加内存占用。
- `*_GPU_LAYERS`：GPU 承载层数。80B 默认 22 层是在本机保留约 2GB 显存余量后的稳定上限；32B 默认 `all`。
- `*_BATCH_SIZE` / `*_UBATCH_SIZE`：llama.cpp 的 token 计算批大小，不是一次翻译多少条字幕。显存不足时优先降低 `*_UBATCH_SIZE`。
- `*_THREADS` / `*_THREADS_BATCH`：CPU 线程数。80B 默认 16，给系统和桌面保留调度余量。
- `*_CACHE_TYPE_K` / `*_CACHE_TYPE_V`：KV cache 精度。`q8_0` 比 `f16` 省显存，质量通常够用。
- `*_REASONING`：本流程默认 `off`。字幕翻译更依赖稳定的结构化输出，不启用显式推理模式。

大致硬件参考：

- 当前 4090 24GB + 64GB 内存：80B 的 22 层配置在 6 轮连续四条结构化翻译测试中保持稳定，峰值显存 21,922MiB、最低余量 2,227MiB，最低系统可用内存约 17.5GB。20 层峰值 20,138MiB，可作为更保守的回退；24 层虽完成同样测试，但峰值 23,901MiB、只余 248MiB，不能给桌面和突发分配留足空间，因此不采用。
- 24GB 显存但系统内存不足 64GB：优先使用 32B，或进一步降低 80B 的 GPU 层数前先确认系统内存余量。
- 24GB 以上显存：32B 可全量 GPU offload，`QWEN_32B_GPU_LAYERS=all`。
- 12GB 到 16GB 显存：可以尝试降低 `QWEN_GPU_LAYERS` 和 `QWEN_UBATCH_SIZE`，让一部分层走 CPU；速度会慢。
- 8GB 显存或纯 CPU：仍可运行部分步骤，但 Qwen 32B 会很慢，建议换更小 GGUF 或只对少量字幕测试。
- 内存建议 32GB 起步，64GB 更舒服；如果 Qwen 部分 CPU offload 较多，内存压力会明显增加。

Whisper.cpp 参数：

```powershell
$env:WHISPER_THREADS = "16"
$env:WHISPER_PROCESSORS = "1"
$env:WHISPER_BEST_OF = "5"
$env:WHISPER_BEAM_SIZE = "5"
$env:WHISPER_DEVICE = "CUDA0"
```

- `WHISPER_THREADS`：CPU 线程数，CPU 转写时影响明显。
- `WHISPER_DEVICE`：如果 whisper.cpp 支持 GPU，可指定设备；不设置则按 whisper.cpp 默认行为。
- `WHISPER_BEST_OF` / `WHISPER_BEAM_SIZE`：数值越高可能更稳，但速度更慢。

Demucs 参数：

```powershell
$env:DEMUCS_BATCH_SIZE = "1"
$env:DEMUCS_SHIFTS = "2"
$env:DEMUCS_OVERLAP = "0.25"
$env:DEMUCS_JOBS = "1"
```

- 显存紧张时保持 `DEMUCS_BATCH_SIZE=1`。
- `DEMUCS_SHIFTS` 越高通常越慢，也更耗资源。

PaddleOCR 参数：

```powershell
$env:PADDLEOCR_DEVICE = "gpu"
$env:OCR_MIN_CONFIDENCE = "0.45"
```

- 4GB 到 8GB 显存通常足够处理 OCR，但具体取决于分辨率和 PaddleOCR 版本。
- `OCR_MIN_CONFIDENCE` 提高会减少误识别，也可能漏掉真实屏幕文字；降低则相反。

Step 9 后端调优：

```powershell
$env:SUB_POLISH_PROVIDER = "codex-cli"
$env:SUB_POLISH_CODEX_COMMAND = "codex"
$env:SUB_POLISH_MODEL = "gpt-5.6-sol"
$env:SUB_POLISH_CODEX_REASONING_EFFORT = "high"
$env:SUB_POLISH_BATCH_SIZE = "48"
$env:SUB_POLISH_FILE_BATCH_SIZE = "1"
$env:SUB_POLISH_TIMEOUT = "180"
```

- 默认使用 `codex-cli`、`gpt-5.6-sol` 和 `high` 推理强度；环境变量或命令行参数仍可覆盖。
- `SUB_POLISH_BATCH_SIZE` 越大，请求次数越少，但单次失败影响也更大；网络或后端不稳定时可以降到 25 或 10。
- `SUB_POLISH_FILE_BATCH_SIZE` 默认是 `1`，每集完成后立即写入 ASS 和缓存；不建议为了减少少量请求开销而合并整季文件。
- `SUB_POLISH_TIMEOUT` 是单次后端请求超时秒数。

OpenAI-compatible 后端示例：

```powershell
$env:SUB_POLISH_PROVIDER = "openai"
$env:SUB_POLISH_MODEL = "gpt-4.1-mini"
$env:SUB_POLISH_BASE_URL = "https://api.openai.com/v1"
$env:SUB_POLISH_API_KEY = "sk-..."
```

`run_all.py` 常用参数：

```powershell
python .\run_all.py . --dry-run
python .\run_all.py . --target-stem "Nashville.S01E01"
python .\run_all.py . --qwen-profile 80b
python .\run_all.py . --qwen-profile 32b
python .\run_all.py . --skip-ocr
python .\run_all.py . --force-embedded
python .\run_all.py . --no-polish
python .\run_all.py . --cleanup
python .\run_all.py . --start-at V2_STEP9 --stop-after V2_STEP10
```

- `--dry-run`：只打印计划执行的步骤，不真正跑流程。
- `--target-stem`：只处理指定文件名主干，可重复传入。
- `--qwen-profile`：选择 Step 7 的 `80b` 高质量配置或 `32b` 低内存回退配置；默认 `80b`。
- `--skip-ocr`：当前批次不做画面 OCR 提取、翻译、注解或叠加。
- `--ocr-over-embedded`：在手动 `--source embedded` 流程中也启用 OCR；智能自动分流已经默认对内嵌字幕批次启用 OCR。
- `--force-embedded`：忽略已生成的内嵌字幕结果并重新提取、对齐。
- `--no-polish`：跳过 Step 9 后端大模型调优，只做合并和过滤。
- `--start-at` / `--stop-after`：从指定步骤继续或在指定步骤后停止，例如 `V2_STEP9`。
- `--cleanup`：显式执行清理；默认不会清理中间文件。

## 模型流程

- Step 7 使用本地 Qwen 模型翻译对白、歌词、OCR，并生成文化注解。内嵌双语来源只翻译没有中文的段落，不覆盖片源已有中文。普通对白每批翻译 4 条，前后各附 4 条只读上下文；响应必须逐条返回原 ID，缺失、重复或空结果会按对应 ID 回退为单条重译。歌词和咒语仍按单条处理，避免跨类型合并影响韵律或专门用词。
- Step 8 会把中文来源写入 ASS 事件元数据：片源中文、Qwen 译文、人工译文和 OCR 译文各自可追踪。
- Step 9 使用后端大模型做最终润色、OCR 去噪和可复用 hint 提取。自动分流会按来源传入 `--polish-origins`：内嵌双语成品只调优 `qwen,ocr`，不会改动片源原有中文；普通已有双语 ASS 则按 `all` 处理。
- Step 9 同时支持单条 `中文\\N英文` 事件，以及时间轴完全相同、分别使用 `English`/`Chinese`（或 `Default`）样式的旧式双语事件；后者只更新中文事件，不改英文行和时间轴。
- 旧式分离事件完成调优后，可运行 `python normalize_legacy_ass_layout.py . --glob "剧集匹配式*.ass" --backup-dir temp/ass_layout_backup`，将精确同时间轴的中英文合并为当前 `BILINGUAL` 单事件样式；未配对事件会保留，备份也只写入项目 `temp`。
- Step 9 沉淀出的 hint 会回流给 Step 7；Step 7 只按当前台词检索相关提示，不会全量塞进 prompt。

## 可复用字典文件

- `subtitle_glossary.json`：仅存放需要显示在屏幕上的文化注解；`zh` 是完整注解正文，不参与台词翻译替换
- `subtitle_terminology.json`：固定译名和剧集术语；`preferred_zh` 是可直接用于字幕的标准译名，`note` 只说明适用条件，绝不能写入字幕正文
- `common_mistranslation_hints.json`：常见误译陷阱；`guidance` 是编辑指引，不是直接替换文本
- `common_phrase_correction_hints.json`：口语短语和行业表达倾向；`guidance` 可列出多个候选，必须按上下文选择
- `embedded_song_translations.json`：内嵌双语轨缺失中文时，供歌曲、吟唱和咒语精确复用的校订译文
- `ocr_low_value_short_texts.json`：单独出现时通常应删除的低价值 OCR 碎片

`subtitle_terminology.json` 的条目可用 `series` 按文件名限定剧集，用 `scope` 限定对白、歌词、咒语、OCR 或注解。文化注解词库不会再作为翻译术语提示传入 Step 7/Step 9。

## 注意事项

不要提交生成后的字幕、日志、视频、缓存或 `temp/` 中间文件。这些内容已经在 `.gitignore` 中忽略。
