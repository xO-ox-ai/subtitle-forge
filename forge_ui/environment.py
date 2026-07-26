from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import AppSettings
from .paths import app_data_root, models_dir, source_root


@dataclass
class CheckResult:
    category: str
    name: str
    state: str
    detail: str
    path: str = ""
    required: bool = True


@dataclass
class HardwareInfo:
    ram_gb: float
    gpu_name: str
    vram_gb: float


def _total_ram_gb() -> float:
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / (1024**3)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024**3)
    except (AttributeError, OSError, ValueError):
        return 0.0


def detect_hardware(env: dict[str, str] | None = None) -> HardwareInfo:
    gpu_name = "未检测到 NVIDIA GPU"
    vram_gb = 0.0
    command = shutil.which("nvidia-smi", path=(env or os.environ).get("PATH"))
    if command:
        try:
            result = subprocess.run(
                [command, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                timeout=8,
                env=env,
            )
            if result.returncode == 0 and result.stdout.strip():
                first = result.stdout.strip().splitlines()[0]
                name, memory = first.rsplit(",", 1)
                gpu_name = name.strip()
                vram_gb = float(memory.strip()) / 1024
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return HardwareInfo(ram_gb=_total_ram_gb(), gpu_name=gpu_name, vram_gb=vram_gb)


def recommend_qwen(hardware: HardwareInfo) -> tuple[str, str]:
    if hardware.ram_gb >= 60 and hardware.vram_gb >= 22:
        return "80b", "推荐 80B：系统内存和显存达到高质量配置要求。"
    if hardware.ram_gb >= 30 and hardware.vram_gb >= 10:
        return "32b", "推荐 32B：适合当前内存/显存，速度和质量较均衡。"
    if hardware.ram_gb >= 24:
        return "32b", "可尝试 32B，但会较多使用系统内存，速度可能偏慢。"
    return "32b", "内存低于推荐值；32B 也可能无法稳定运行，建议使用外部翻译服务或升级内存。"


def _configured_or_which(settings: AppSettings, name: str, env: dict[str, str]) -> str:
    configured = settings.tool_paths.get(name, "").strip()
    if configured:
        candidate = Path(configured).expanduser().resolve(strict=False)
        if candidate.is_file():
            return str(candidate)
        if candidate.is_dir():
            filename = name if name.lower().endswith(".exe") else f"{name}.exe"
            child = candidate / filename
            if child.is_file():
                return str(child)
    executable = name if name.lower().endswith(".exe") else f"{name}.exe"
    return shutil.which(executable, path=env.get("PATH", "")) or shutil.which(name, path=env.get("PATH", "")) or ""


def _candidate_python(settings: AppSettings, runtime_name: str) -> Path:
    configured = settings.python_paths.get(runtime_name, "").strip()
    if configured:
        path = Path(configured).expanduser().resolve(strict=False)
        if path.is_dir():
            path = path / "python.exe"
        return path
    candidates = [
        app_data_root() / "runtime" / runtime_name / "python.exe",
        source_root() / ".venvs" / runtime_name / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    return next((path for path in candidates if path.exists()), candidates[-1])


def _module_available(python: Path, modules: tuple[str, ...]) -> tuple[bool, str]:
    if not python.exists():
        return False, f"未找到 Python：{python}"
    if python.resolve(strict=False) == Path(sys.executable).resolve(strict=False):
        missing = []
        for name in modules:
            try:
                if importlib.util.find_spec(name) is None:
                    missing.append(name)
            except (ImportError, ModuleNotFoundError):
                missing.append(name)
        return not missing, f"缺少模块：{', '.join(missing)}" if missing else f"运行时：{python}"
    code = (
        "import importlib.util,json;"
        f"mods={json.dumps(modules)};"
        "missing=[m for m in mods if importlib.util.find_spec(m) is None];"
        "print(json.dumps(missing))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=15,
        )
        missing = json.loads(result.stdout.strip() or "[]") if result.returncode == 0 else list(modules)
        return not missing, f"缺少模块：{', '.join(missing)}" if missing else f"运行时：{python}"
    except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        return False, f"运行时检测失败：{exc}"


def _resolve_model(configured: str, filenames: tuple[str, ...], extra_dirs: tuple[Path, ...] = ()) -> Path:
    if configured:
        return Path(configured).expanduser().resolve(strict=False)
    search_dirs = list(extra_dirs) + [models_dir(), source_root() / "models"]
    for directory in search_dirs:
        for filename in filenames:
            candidate = directory / filename
            if candidate.exists():
                return candidate
    return search_dirs[0] / filenames[0]


def run_environment_checks(settings: AppSettings) -> tuple[HardwareInfo, list[CheckResult]]:
    env = settings.child_environment()
    hardware = detect_hardware(env)
    results: list[CheckResult] = [
        CheckResult("硬件", "系统内存", "ok" if hardware.ram_gb >= 30 else "warning", f"{hardware.ram_gb:.1f} GB"),
        CheckResult(
            "硬件",
            "NVIDIA GPU",
            "ok" if hardware.vram_gb >= 10 else "warning",
            f"{hardware.gpu_name} · {hardware.vram_gb:.1f} GB 显存" if hardware.vram_gb else hardware.gpu_name,
            required=False,
        ),
    ]

    tools = [
        ("ffmpeg", True, "音视频处理"),
        ("ffprobe", True, "媒体/字幕轨探测"),
        ("llama-server", True, "本地 Qwen 翻译"),
        ("whisper-server", True, "无字幕视频语音识别"),
        ("codex", settings.polish_provider == "codex-cli", "Codex 后端润色"),
        ("seconv", False, "位图字幕优先 OCR，可按需自动下载"),
    ]
    for name, required, purpose in tools:
        path = _configured_or_which(settings, name, env)
        state = "ok" if path else ("error" if required else "warning")
        detail = f"{purpose} · {path}" if path else f"{purpose} · 未找到"
        results.append(CheckResult("外部工具", name, state, detail, path, required))

    runtimes = [
        ("demucs", ("torch", "demucs"), "人声分离"),
        ("whisperx", ("torch", "whisperx", "pyannote.audio"), "说话人分离"),
        ("whisper-at", ("torch", "whisper_at"), "音乐/歌词识别"),
        ("paddleocr", ("paddleocr",), "画面 OCR"),
    ]
    for name, modules, purpose in runtimes:
        python = _candidate_python(settings, name)
        available, detail = _module_available(python, modules)
        results.append(
            CheckResult(
                "Python 运行时",
                name,
                "ok" if available else "error",
                f"{purpose} · {detail}",
                str(python),
                True,
            )
        )

    qwen_name = (
        "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"
        if settings.qwen_profile == "80b"
        else "Qwen3-32B-Q4_K_M.gguf"
    )
    llama_tool = _configured_or_which(settings, "llama-server", env)
    llama_dirs = (Path(llama_tool).resolve(strict=False).parent,) if llama_tool else ()
    qwen_path = _resolve_model(settings.selected_qwen_path, (qwen_name,), llama_dirs)
    results.append(
        CheckResult(
            "模型",
            f"Qwen {settings.qwen_profile.upper()}",
            "ok" if qwen_path.is_file() else "error",
            f"{qwen_path}" if qwen_path.is_file() else f"未找到：{qwen_path}",
            str(qwen_path),
            True,
        )
    )
    whisper_tool = _configured_or_which(settings, "whisper-server", env)
    whisper_dirs = (Path(whisper_tool).resolve(strict=False).parent,) if whisper_tool else ()
    whisper_path = _resolve_model(settings.whisper_model_path, ("ggml-large-v3.bin",), whisper_dirs)
    results.append(
        CheckResult(
            "模型",
            "Whisper large-v3",
            "ok" if whisper_path.is_file() else "error",
            f"{whisper_path}" if whisper_path.is_file() else f"未找到：{whisper_path}",
            str(whisper_path),
            True,
        )
    )
    results.append(
        CheckResult(
            "账户",
            "Hugging Face Token",
            "ok" if settings.hf_token else "warning",
            "已填写（用于受限的说话人分离模型）" if settings.hf_token else "未填写；未缓存 pyannote 时将跳过/失败",
            required=False,
        )
    )
    cloud_ok = settings.cloud_polish_available
    results.append(
        CheckResult(
            "账户",
            "云端润色",
            "ok" if cloud_ok else "warning",
            f"{settings.polish_provider} 已配置" if cloud_ok else "未配置 API Key；任务将自动关闭云端润色",
            required=False,
        )
    )
    return hardware, results
