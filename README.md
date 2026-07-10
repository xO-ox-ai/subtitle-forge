# Subtitle Workflow

Local subtitle production workflow for extracting, translating, rendering, and polishing bilingual TV subtitles.

## Main Entry

Run the full auto workflow from the subtitle/video directory:

```powershell
python .\run_all.py .
```

The auto mode groups inputs by current subtitle state:

- videos with no embedded or external subtitles: Step 1 to Step 8, then Step 9
- videos with embedded subtitles but no external subtitles: Step 0, then Step 6 to Step 8, then Step 9
- videos with external monolingual subtitles: Step 6 to Step 8, then Step 9
- existing bilingual ASS subtitles: Step 9 only

Intermediate files are preserved by default. Step 10 cleanup is opt-in.

## Environment

This workflow is Windows-first and expects PowerShell plus Python 3.10-style virtual environments.

Required command-line tools:

- `ffmpeg.exe` and `ffprobe.exe`
- `llama-server.exe` from llama.cpp, used by Step 7 for local Qwen translation
- `whisper-server.exe` from whisper.cpp, used by Step 3 transcription
- `codex.exe` or an OpenAI-compatible API key, used by Step 9 backend polish

Default tool/model locations can be overridden with environment variables:

```powershell
$env:FFMPEG_DIR = "C:\ffmpeg\bin"
$env:LLAMA_DIR = "C:\llama"
$env:LLAMA_SERVER_EXE = "C:\llama\llama-server.exe"
$env:WHISPER_SERVER_EXE = "C:\whisper.cpp\whisper-server.exe"
```

Python environments are intentionally split because Demucs, WhisperX/pyannote, Whisper-AT, and PaddleOCR often need different dependency stacks:

```powershell
$env:SUB_PYTHON_EXE = "C:\Python\Python310\python.exe"
$env:SUB_DEMUCS_SCRIPTS = "C:\Python\Python310\demucs\Scripts"
$env:SUB_WHISPERX_SCRIPTS = "C:\Python\Python310\whisperx\Scripts"
$env:SUB_WHISPER_AT_SCRIPTS = "C:\Python\Python310\whisper-at\Scripts"
$env:SUB_PADDLEOCR_SCRIPTS = "C:\Python\Python310\paddleocr\Scripts"
```

Cache locations:

```powershell
$env:HF_HOME = "C:\Python\hf-cache"
$env:HF_HUB_CACHE = "C:\Python\hf-cache\hub"
$env:TORCH_HOME = "C:\Python\hf-cache\torch"
$env:PIP_CACHE_DIR = "C:\Python\pip-cache"
$env:PIP_FIND_LINKS = "C:\Python\wheel-cache"
```

If a proxy is needed for model downloads:

```powershell
$env:SUB_PROXY = "http://127.0.0.1:7897"
```

The pipeline sets UTF-8 related Python environment defaults itself, but setting them globally is still harmless:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## Models To Prepare

Local Qwen model for Step 7:

- Default model alias: `qwen3-32b`
- Default GGUF filename: `Qwen3-32B-Q4_K_M.gguf`
- Put it next to `llama-server.exe`, under `LLAMA_DIR`, or set `QWEN_GGUF`.

```powershell
$env:QWEN_GGUF = "C:\llama\models\Qwen3-32B-Q4_K_M.gguf"
$env:QWEN_MODEL = "qwen3-32b"
$env:QWEN_BASE_URL = "http://127.0.0.1:8080/v1"
```

Useful Qwen runtime knobs:

```powershell
$env:QWEN_CTX_SIZE = "8192"
$env:QWEN_THREADS = "20"
$env:QWEN_THREADS_BATCH = "20"
$env:QWEN_GPU_LAYERS = "all"
$env:QWEN_FLASH_ATTN = "on"
$env:QWEN_CACHE_TYPE_K = "q8_0"
$env:QWEN_CACHE_TYPE_V = "q8_0"
$env:QWEN_REASONING = "off"
```

Whisper.cpp model for Step 3:

- Default filename: `ggml-large-v3.bin`
- Put it next to `whisper-server.exe`, or set `WHISPER_MODEL` / `WHISPER_CPP_MODEL`.

```powershell
$env:WHISPER_MODEL = "C:\whisper.cpp\models\ggml-large-v3.bin"
$env:WHISPER_LANGUAGE = "en"
$env:WHISPER_SERVER_PORT = "8091"
$env:WHISPER_THREADS = "16"
```

Pyannote diarization model for Step 4:

- Default model: `pyannote/speaker-diarization-3.1`
- Set `HF_TOKEN` if the model is not already cached or if Hugging Face access is gated.

```powershell
$env:DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
$env:HF_TOKEN = "hf_..."
```

Whisper-AT model for Step 5:

- The script loads Whisper-AT `large-v2`.
- Cache defaults to `C:\Python\hf-cache\whisper-at`, or set `WHISPER_AT_CACHE`.

```powershell
$env:WHISPER_AT_CACHE = "C:\Python\hf-cache\whisper-at"
```

PaddleOCR for Step 6:

- Default OCR version: `PP-OCRv6`
- Default OCR language: `en`
- Default device: `gpu`

```powershell
$env:OCR_LANG = "en"
$env:PADDLEOCR_DEVICE = "gpu"
$env:PADDLEOCR_VERSION = "PP-OCRv6"
$env:OCR_MIN_CONFIDENCE = "0.45"
```

Step 9 backend polish:

- Default provider is `codex-cli`, using the local Codex CLI configuration.
- To use an OpenAI-compatible API instead, set provider/base URL/API key.

```powershell
$env:SUB_POLISH_PROVIDER = "codex-cli"
$env:SUB_POLISH_CODEX_COMMAND = "codex"
$env:SUB_POLISH_CODEX_REASONING_EFFORT = "medium"
```

OpenAI-compatible alternative:

```powershell
$env:SUB_POLISH_PROVIDER = "openai"
$env:SUB_POLISH_MODEL = "gpt-4.1-mini"
$env:SUB_POLISH_BASE_URL = "https://api.openai.com/v1"
$env:SUB_POLISH_API_KEY = "sk-..."
```

## Model Flow

- Step 7 uses the local Qwen model for dialogue, lyrics, OCR translation, and cultural notes.
- Step 9 uses the backend model for final polish, OCR cleanup, and reusable hint extraction.
- Step 9 hint files are fed back into Step 7 by matching only guidance relevant to the current segment.

## Reusable Hint Files

- `subtitle_glossary.json`: names, terms, and cultural-note glossary
- `common_mistranslation_hints.json`: recurring proper-noun, phrase, and mistranslation traps
- `common_phrase_correction_hints.json`: reusable colloquial or industry phrase guidance
- `ocr_low_value_short_texts.json`: OCR fragments that should usually be dropped when shown alone

## Notes

Do not commit generated subtitles, logs, media, or `temp/` cache files. They are ignored by `.gitignore`.
