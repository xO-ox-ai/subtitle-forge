from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "SubtitleForge"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def source_root() -> Path:
    return Path(__file__).resolve().parent.parent


def pipeline_root() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS")) / "pipeline"
    return source_root()


def app_data_root() -> Path:
    override = os.environ.get("SUBTITLE_FORGE_HOME")
    if override:
        return Path(override).expanduser().resolve(strict=False)
    if is_frozen():
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_NAME
    return source_root()


def config_path() -> Path:
    override = os.environ.get("SUBTITLE_FORGE_CONFIG")
    if override:
        return Path(override).expanduser().resolve(strict=False)
    return app_data_root() / "config.json"


def models_dir() -> Path:
    return app_data_root() / "models"


def cache_dir() -> Path:
    return app_data_root() / ".cache"


def tools_dir() -> Path:
    return app_data_root() / "tools"


def runtime_dir() -> Path:
    return app_data_root() / "runtime"
