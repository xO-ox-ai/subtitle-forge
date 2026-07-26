from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import AppSettings
from .paths import pipeline_root


FLOW_LABELS = {
    "direct_bilingual_done": "已生成双语字幕 · 仅保护性合并/精修",
    "ready_bilingual": "已有双语 ASS · 仅精修",
    "external_mono": "外部单语字幕 · 导入/翻译/渲染",
    "embedded_bilingual": "内嵌中英字幕 · 对齐/补译/渲染",
    "embedded_only": "内嵌英文字幕 · 提取/翻译/渲染",
    "no_subtitles": "无可用字幕 · 完整语音识别流程",
}


@dataclass
class VideoTask:
    path: Path
    group: str
    selected: bool = True
    use_ocr: bool = True
    use_polish: bool = False
    cleanup: bool = False
    status: str = "等待"

    @property
    def flow(self) -> str:
        return FLOW_LABELS.get(self.group, self.group)


@dataclass
class PipelineCommand:
    task: VideoTask
    argv: list[str]
    env: dict[str, str]

    @property
    def display(self) -> str:
        return subprocess.list2cmdline(self.argv)


def _find_tool(settings: AppSettings, name: str, env: dict[str, str]) -> str:
    configured = settings.tool_paths.get(name, "").strip()
    if configured:
        path = Path(configured).expanduser().resolve(strict=False)
        if path.is_dir():
            path = path / (name if name.endswith(".exe") else f"{name}.exe")
        if path.is_file():
            return str(path)
    return (
        shutil.which(f"{name}.exe", path=env.get("PATH", ""))
        or shutil.which(name, path=env.get("PATH", ""))
        or name
    )


def _embedded_languages(video: Path, settings: AppSettings, env: dict[str, str]) -> tuple[bool, bool]:
    from run_all import BITMAP_SUBTITLE_CODECS, CHINESE_SUBTITLE_LANGUAGES, TEXT_SUBTITLE_CODECS

    command = [
        _find_tool(settings, "ffprobe", env),
        "-v",
        "error",
        "-select_streams",
        "s",
        "-show_streams",
        "-of",
        "json",
        str(video),
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            env=env,
        )
        streams = json.loads(result.stdout or "{}").get("streams", []) if result.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        streams = []
    has_english = False
    has_chinese = False
    for stream in streams:
        codec = str(stream.get("codec_name", "")).lower()
        if codec not in TEXT_SUBTITLE_CODECS | BITMAP_SUBTITLE_CODECS:
            continue
        tags = stream.get("tags") or {}
        language = str(tags.get("language", "")).strip().lower()
        title = str(tags.get("title", "")).strip().lower()
        if language in {"", "eng", "en", "english"} or "english" in title:
            has_english = True
        if language in CHINESE_SUBTITLE_LANGUAGES or "chinese" in title or "中文" in title:
            has_chinese = has_chinese or codec in TEXT_SUBTITLE_CODECS
    return has_english, has_chinese


def scan_video_tasks(base_dir: Path, settings: AppSettings) -> list[VideoTask]:
    base_dir = base_dir.expanduser().resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"目录不存在：{base_dir}")
    root_text = str(pipeline_root())
    inserted = False
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        inserted = True
    try:
        from common import iter_videos
        from run_all import (
            find_bilingual_ass,
            find_external_mono_subtitle,
            is_direct_embedded_bilingual_ass,
        )

        env = settings.child_environment()
        tasks: list[VideoTask] = []
        for video in iter_videos(base_dir):
            existing = find_bilingual_ass(video)
            if existing and is_direct_embedded_bilingual_ass(existing):
                group = "direct_bilingual_done"
            elif existing:
                group = "ready_bilingual"
            elif find_external_mono_subtitle(video):
                group = "external_mono"
            else:
                has_english, has_chinese = _embedded_languages(video, settings, env)
                if has_english and has_chinese:
                    group = "embedded_bilingual"
                elif has_english:
                    group = "embedded_only"
                else:
                    group = "no_subtitles"
            tasks.append(
                VideoTask(
                    path=video,
                    group=group,
                    use_ocr=group not in {"direct_bilingual_done", "ready_bilingual"},
                    use_polish=settings.cloud_polish_available,
                )
            )
        return tasks
    finally:
        if inserted and sys.path and sys.path[0] == root_text:
            sys.path.pop(0)


def _launcher() -> list[str]:
    return [sys.executable, str(pipeline_root() / "run_all.py")]


def build_pipeline_commands(
    base_dir: Path,
    tasks: list[VideoTask],
    settings: AppSettings,
    dry_run: bool = False,
) -> list[PipelineCommand]:
    base_dir = base_dir.expanduser().resolve()
    env = settings.child_environment()
    commands: list[PipelineCommand] = []
    for task in tasks:
        if not task.selected:
            continue
        try:
            selector = str(task.path.resolve().relative_to(base_dir))
        except ValueError:
            selector = str(task.path.resolve())
        argv = _launcher() + [
            str(base_dir),
            "--source",
            "auto",
            "--target-stem",
            selector,
            "--qwen-profile",
            settings.qwen_profile,
            "--ocr-interval",
            str(settings.ocr_interval),
        ]
        if settings.selected_qwen_path:
            argv.extend(["--qwen-gguf", settings.selected_qwen_path])
        if not task.use_ocr:
            argv.append("--skip-ocr")
        if not task.use_polish or not settings.cloud_polish_available:
            argv.append("--no-polish")
        else:
            argv.extend(["--polish-provider", settings.polish_provider])
            if settings.polish_base_url:
                argv.extend(["--polish-base-url", settings.polish_base_url])
            if settings.polish_model:
                argv.extend(["--polish-model", settings.polish_model])
        if task.cleanup:
            argv.append("--cleanup")
        if dry_run:
            argv.append("--dry-run")
        commands.append(PipelineCommand(task=task, argv=argv, env=env.copy()))
    return commands
