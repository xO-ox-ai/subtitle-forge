from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import app_data_root, cache_dir, config_path, is_frozen, models_dir, tools_dir
from .secrets import protect_secret, unprotect_secret


TOOL_ENV_NAMES = {
    "ffmpeg": "FFMPEG_EXE",
    "ffprobe": "FFPROBE_EXE",
    "llama-server": "LLAMA_SERVER_EXE",
    "whisper-server": "WHISPER_SERVER_EXE",
    "codex": "SUB_POLISH_CODEX_COMMAND",
    "seconv": "SUB_SECONV_EXE",
}

PYTHON_ENV_NAMES = {
    "demucs": "SUB_PYTHON_DEMUCS_EXE",
    "whisperx": "SUB_PYTHON_WHISPERX_EXE",
    "whisper-at": "SUB_PYTHON_WHISPER_AT_EXE",
    "paddleocr": "SUB_PYTHON_PADDLEOCR_EXE",
}

PYTHON_SCRIPTS_ENV_NAMES = {
    "demucs": "SUB_DEMUCS_SCRIPTS",
    "whisperx": "SUB_WHISPERX_SCRIPTS",
    "whisper-at": "SUB_WHISPER_AT_SCRIPTS",
    "paddleocr": "SUB_PADDLEOCR_SCRIPTS",
}


@dataclass
class AppSettings:
    schema_version: int = 1
    last_media_dir: str = ""
    tool_paths: dict[str, str] = field(default_factory=lambda: {name: "" for name in TOOL_ENV_NAMES})
    python_paths: dict[str, str] = field(default_factory=lambda: {name: "" for name in PYTHON_ENV_NAMES})
    qwen_profile: str = "32b"
    qwen_32b_path: str = ""
    qwen_80b_path: str = ""
    whisper_model_path: str = ""
    hf_token: str = ""
    polish_provider: str = "openai"
    polish_api_key: str = ""
    polish_base_url: str = "https://api.openai.com/v1"
    polish_model: str = "gpt-4.1-mini"
    proxy: str = ""
    ocr_interval: float = 0.5
    cache_root: str = ""

    @property
    def selected_qwen_path(self) -> str:
        return self.qwen_80b_path if self.qwen_profile == "80b" else self.qwen_32b_path

    @property
    def cloud_polish_available(self) -> bool:
        if self.polish_provider == "codex-cli":
            return True
        return bool(self.polish_api_key.strip())

    def resolved_cache_root(self) -> Path:
        return Path(self.cache_root).expanduser() if self.cache_root else cache_dir()

    def resolved_model_dir(self) -> Path:
        candidates = [self.qwen_32b_path, self.qwen_80b_path, self.whisper_model_path]
        for value in candidates:
            if value:
                return Path(value).expanduser().resolve(strict=False).parent
        return models_dir()

    def child_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        path_dirs: list[str] = []
        for name, value in self.tool_paths.items():
            value = value.strip()
            if not value:
                continue
            env_name = TOOL_ENV_NAMES.get(name)
            if env_name:
                env[env_name] = value
            parent = Path(value).expanduser().resolve(strict=False).parent
            if str(parent) not in path_dirs:
                path_dirs.append(str(parent))
        portable_tools = {
            "ffmpeg": tools_dir() / "ffmpeg" / "bin" / "ffmpeg.exe",
            "ffprobe": tools_dir() / "ffmpeg" / "bin" / "ffprobe.exe",
            "llama-server": tools_dir() / "llama" / "llama-server.exe",
            "whisper-server": tools_dir() / "whisper" / "whisper-server.exe",
            "seconv": tools_dir() / "subtitle-edit" / "seconv.exe",
        }
        for name, candidate in portable_tools.items():
            if self.tool_paths.get(name, "").strip() or not candidate.exists():
                continue
            env_name = TOOL_ENV_NAMES[name]
            env[env_name] = str(candidate)
            if str(candidate.parent) not in path_dirs:
                path_dirs.append(str(candidate.parent))
        for name, value in self.python_paths.items():
            value = value.strip()
            if value and name in PYTHON_ENV_NAMES:
                python_path = Path(value).expanduser().resolve(strict=False)
                if python_path.is_dir():
                    python_path = python_path / "python.exe"
            else:
                python_path = app_data_root() / "runtime" / name / "python.exe"
            if python_path.exists():
                env[PYTHON_ENV_NAMES[name]] = str(python_path)
                scripts_dir = python_path.parent / "Scripts"
                if scripts_dir.exists():
                    env[PYTHON_SCRIPTS_ENV_NAMES[name]] = str(scripts_dir)
        if path_dirs:
            env["PATH"] = os.pathsep.join(path_dirs + [env.get("PATH", "")])

        env["SUB_APP_ROOT"] = str(app_data_root())
        if is_frozen():
            # Core pipeline steps re-enter the frozen controller, whose
            # entrypoint dispatches the requested bundled .py file. This keeps
            # the application independent of any system Python installation.
            env["SUB_PYTHON_EXE"] = sys.executable
        env["QWEN_PROFILE"] = self.qwen_profile
        if self.qwen_32b_path:
            env["QWEN_32B_GGUF"] = self.qwen_32b_path
        if self.qwen_80b_path:
            env["QWEN_80B_GGUF"] = self.qwen_80b_path
        if self.whisper_model_path:
            env["WHISPER_MODEL"] = self.whisper_model_path
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token

        cache_root = self.resolved_cache_root().resolve(strict=False)
        env["HF_HOME"] = str(cache_root / "huggingface")
        env["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
        env["TORCH_HOME"] = str(cache_root / "torch")
        env["PIP_CACHE_DIR"] = str(cache_root / "pip")
        env["SUB_TEMP_DIR"] = str(app_data_root() / "temp")
        if self.proxy:
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                env[name] = self.proxy
        else:
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                env.pop(name, None)
            env["SUB_PROXY"] = ""

        env["SUB_POLISH_PROVIDER"] = self.polish_provider
        env["SUB_POLISH_BASE_URL"] = self.polish_base_url
        env["SUB_POLISH_MODEL"] = self.polish_model
        if self.polish_api_key:
            env["SUB_POLISH_API_KEY"] = self.polish_api_key
        else:
            env.pop("SUB_POLISH_API_KEY", None)
        return env


def _merge_settings(payload: dict[str, Any]) -> AppSettings:
    settings = AppSettings()
    for name in settings.__dataclass_fields__:
        if name in payload and name not in {"hf_token", "polish_api_key"}:
            setattr(settings, name, payload[name])
    for field_name in ("tool_paths", "python_paths"):
        defaults = getattr(AppSettings(), field_name)
        defaults.update(payload.get(field_name) or {})
        setattr(settings, field_name, defaults)
    for field_name in ("hf_token", "polish_api_key"):
        protected = str(payload.get(field_name, ""))
        if protected:
            try:
                setattr(settings, field_name, unprotect_secret(protected))
            except Exception:
                setattr(settings, field_name, "")
    if settings.qwen_profile not in {"32b", "80b"}:
        settings.qwen_profile = "32b"
    return settings


def load_settings(path: Path | None = None) -> AppSettings:
    target = path or config_path()
    if not target.exists():
        return AppSettings()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return AppSettings()
        return _merge_settings(payload)
    except (OSError, json.JSONDecodeError):
        return AppSettings()


def save_settings(settings: AppSettings, path: Path | None = None) -> Path:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(settings)
    payload["hf_token"] = protect_secret(settings.hf_token)
    payload["polish_api_key"] = protect_secret(settings.polish_api_key)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)
    return target
