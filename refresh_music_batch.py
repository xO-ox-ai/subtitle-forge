"""
refresh_music_batch.py - Rebuild music HMM marks, lyric translations, and ASS files.
Usage: python refresh_music_batch.py "<video_dir>" [--exclude-stem STEM ...]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from pipeline_common import configure_environment, environment_for_script, python_for_script


configure_environment()


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"}
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh music flags and ASS outputs for a batch of videos.")
    parser.add_argument("base_dir", help="Directory containing videos")
    parser.add_argument(
        "--exclude-stem",
        action="append",
        default=[],
        help="Video stem to skip. Can be used multiple times.",
    )
    return parser.parse_args()


def iter_videos(base_dir: Path, excluded_stems: set[str]) -> list[Path]:
    videos = sorted(
        path for path in base_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS and path.stem not in excluded_stems
    )
    return videos


def run_step(base_dir: Path, video: Path, script_name: str, label: str, extra_args: list[str] | None = None) -> None:
    cmd = [str(python_for_script(script_name)), str(SCRIPT_DIR / script_name), str(base_dir), "--target-stem", video.stem]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[step] {label} | {video.name}", flush=True)
    start = time.time()
    result = subprocess.run(cmd, env=environment_for_script(script_name))
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed for {video.name}")
    print(f"[done] {label} | {video.name} | {elapsed:.1f}s", flush=True)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    excluded_stems = set()
    for stem in args.exclude_stem:
        excluded_stems.add(Path(stem).stem)
        excluded_stems.add(Path(stem).name)
    videos = iter_videos(base_dir, excluded_stems)
    if not videos:
        print("[error] no target videos found", flush=True)
        sys.exit(1)

    print(f"[start] videos={len(videos)} excluded={sorted(excluded_stems)}", flush=True)
    batch_start = time.time()
    for index, video in enumerate(videos, start=1):
        print(f"[video] {index}/{len(videos)} | {video.name}", flush=True)
        run_step(base_dir, video, "step05_music_hmm.py", "music-hmm")
        run_step(base_dir, video, "step07_qwen_all.py", "qwen-lyrics", ["--skip-ocr", "--force-dialogue"])
        run_step(base_dir, video, "step08_ass_render.py", "ass-base")
        run_step(base_dir, video, "step09_overlay_filter.py", "ass-overlay")

    total = time.time() - batch_start
    print(f"[complete] videos={len(videos)} total={total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
