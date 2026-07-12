import os
import subprocess
import sys
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(value: str | Path) -> Path:
    """Resolve relative configuration paths from the repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def env_path(name: str, default: str | Path) -> Path:
    value = os.environ.get(name)
    return resolve_project_path(value if value else default)


def resolve_executable_path(value: str | Path) -> Path:
    configured = str(value)
    resolved = shutil.which(configured)
    if resolved:
        return Path(resolved).resolve()
    configured_path = Path(configured).expanduser()
    if configured_path.is_absolute():
        return configured_path
    if configured_path.parent != Path("."):
        return resolve_project_path(configured_path)
    return configured_path


_CURRENT_PYTHON = Path(sys.executable).resolve()
_CURRENT_ROOT = _CURRENT_PYTHON.parent.parent if _CURRENT_PYTHON.parent.name.lower() == "scripts" else _CURRENT_PYTHON.parent
VENV_ROOT = env_path("SUB_VENV_ROOT", _CURRENT_ROOT)
DEMUCS_SCRIPTS = env_path("SUB_DEMUCS_SCRIPTS", Path(".venvs") / "demucs" / "Scripts")
WHISPERX_SCRIPTS = env_path("SUB_WHISPERX_SCRIPTS", Path(".venvs") / "whisperx" / "Scripts")
WHISPER_AT_SCRIPTS = env_path("SUB_WHISPER_AT_SCRIPTS", Path(".venvs") / "whisper-at" / "Scripts")
PADDLEOCR_SCRIPTS = env_path("SUB_PADDLEOCR_SCRIPTS", Path(".venvs") / "paddleocr" / "Scripts")
if os.environ.get("SUB_PYTHON_EXE"):
    PYTHON_EXE = resolve_executable_path(os.environ["SUB_PYTHON_EXE"])
elif shutil.which("python"):
    PYTHON_EXE = Path(shutil.which("python") or sys.executable).resolve()
else:
    PYTHON_EXE = Path(sys.executable)
HF_HOME = env_path("HF_HOME", Path(".cache") / "huggingface")
HF_HUB_CACHE = env_path("HF_HUB_CACHE", HF_HOME / "hub")
TORCH_HOME = env_path("TORCH_HOME", Path(".cache") / "torch")
NLTK_DATA = env_path("NLTK_DATA", Path(".cache") / "nltk")
PIP_CACHE_DIR = env_path("PIP_CACHE_DIR", Path(".cache") / "pip")
PIP_FIND_LINKS = env_path("PIP_FIND_LINKS", "wheels")
LLAMA_DIR = env_path("LLAMA_DIR", Path("tools") / "llama")
FFMPEG_DIR = env_path("FFMPEG_DIR", Path("tools") / "ffmpeg" / "bin")
WHISPER_DIR = env_path("WHISPER_DIR", Path("tools") / "whisper")
DEFAULT_PROXY = os.environ.get("SUB_PROXY", "http://127.0.0.1:7897")
TORCH_LIB_DIR = VENV_ROOT / "Lib" / "site-packages" / "torch" / "lib"

_DLL_DIRECTORY_HANDLES = []


def configure_windows_utf8_stdio() -> None:
    """Force UTF-8 stdio on Windows regardless of the parent console code page.

    cmd.exe and Windows PowerShell often inherit the active ANSI/OEM code page
    (for example GBK/CP936), which is fine for plain ASCII but can crash on
    subtitle text or symbols like music notes when a child Python process prints
    to the inherited console. We lock Python and child Python processes to UTF-8
    here so launcher/monitor scripts do not depend on the shell encoding.
    """
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if os.name != "nt":
        return
    os.environ.setdefault("PYTHONLEGACYWINDOWSSTDIO", "0")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def lower_process_priority() -> None:
    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            below_normal = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
            kernel32.SetPriorityClass(handle, below_normal)
        elif hasattr(os, "nice"):
            os.nice(5)
    except Exception:
        pass


def configure_environment() -> None:
    configure_windows_utf8_stdio()
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
    os.environ.setdefault("NLTK_DATA", str(NLTK_DATA))
    os.environ.setdefault("PIP_CACHE_DIR", str(PIP_CACHE_DIR))
    if PIP_FIND_LINKS.exists():
        os.environ.setdefault("PIP_FIND_LINKS", str(PIP_FIND_LINKS))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HTTP_PROXY", DEFAULT_PROXY)
    os.environ.setdefault("HTTPS_PROXY", DEFAULT_PROXY)
    os.environ.setdefault("http_proxy", os.environ["HTTP_PROXY"])
    os.environ.setdefault("https_proxy", os.environ["HTTPS_PROXY"])

    # Keep the system PATH first. Tool locations such as ffmpeg, llama-server,
    # and whisper-server are expected to be managed by the host environment now.
    path_parts = [os.environ.get("PATH", "")]
    for extra_dir in (VENV_ROOT / "Scripts", TORCH_LIB_DIR, LLAMA_DIR, FFMPEG_DIR, WHISPER_DIR):
        if extra_dir.exists():
            path_parts.append(str(extra_dir))
    os.environ["PATH"] = os.pathsep.join(part for part in path_parts if part)

    if hasattr(os, "add_dll_directory"):
        for dll_dir in (VENV_ROOT / "Scripts", TORCH_LIB_DIR, FFMPEG_DIR, LLAMA_DIR, WHISPER_DIR):
            if dll_dir.exists():
                try:
                    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dll_dir)))
                except OSError:
                    pass

    lower_process_priority()


SCRIPT_VENV_SCRIPTS: dict[str, Path] = {
    "step02_demucs.py": DEMUCS_SCRIPTS,
    "step04_diarize.py": WHISPERX_SCRIPTS,
    "step05_music_hmm.py": WHISPER_AT_SCRIPTS,
    "step06_ocr_extract.py": PADDLEOCR_SCRIPTS,
}

SCRIPT_PYTHON_ENV: dict[str, str] = {
    "step02_demucs.py": "SUB_PYTHON_DEMUCS_EXE",
    "step04_diarize.py": "SUB_PYTHON_WHISPERX_EXE",
    "step05_music_hmm.py": "SUB_PYTHON_WHISPER_AT_EXE",
    "step06_ocr_extract.py": "SUB_PYTHON_PADDLEOCR_EXE",
}


def scripts_python(scripts_dir: Path) -> Path | None:
    candidate = scripts_dir / "python.exe"
    return candidate if candidate.exists() else None


def python_for_script(script_name: str) -> Path:
    env_name = SCRIPT_PYTHON_ENV.get(Path(script_name).name)
    if env_name and os.environ.get(env_name):
        return resolve_executable_path(os.environ[env_name])
    scripts_dir = SCRIPT_VENV_SCRIPTS.get(Path(script_name).name)
    if scripts_dir:
        candidate = scripts_python(scripts_dir)
        if candidate:
            return candidate
    return PYTHON_EXE


def environment_for_script(script_name: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if os.name == "nt":
        env.setdefault("PYTHONLEGACYWINDOWSSTDIO", "0")
    scripts_dir = SCRIPT_VENV_SCRIPTS.get(Path(script_name).name)
    if scripts_dir and scripts_dir.exists():
        env["PATH"] = str(scripts_dir) + os.pathsep + env.get("PATH", "")
        env.setdefault("VIRTUAL_ENV", str(scripts_dir.parent))
    env.setdefault("HF_HOME", str(HF_HOME))
    env.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
    env.setdefault("PIP_CACHE_DIR", str(PIP_CACHE_DIR))
    if PIP_FIND_LINKS.exists():
        env.setdefault("PIP_FIND_LINKS", str(PIP_FIND_LINKS))
    return env


def exe_path(name: str, env_var: str | None = None, fallback_dirs: tuple[Path, ...] = ()) -> str:
    if env_var and os.environ.get(env_var):
        configured = os.environ[env_var]
        resolved_configured = shutil.which(configured)
        if resolved_configured:
            return resolved_configured
        configured_path = Path(configured).expanduser()
        if configured_path.is_absolute():
            return str(configured_path)
        if configured_path.parent != Path("."):
            return str(resolve_project_path(configured_path))
        return configured
    resolved = shutil.which(name)
    if resolved:
        return resolved
    for directory in fallback_dirs:
        candidate = directory / name
        if candidate.exists():
            return str(candidate)
    return name


def int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def resolve_target_stem(target: str | None) -> str | None:
    if not target:
        return None
    return Path(target).stem
