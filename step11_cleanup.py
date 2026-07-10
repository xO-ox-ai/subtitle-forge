import argparse
import os
import shutil
import sys
from pathlib import Path

from common import add_common_args, selected_videos, work_dir_for, write_status


TEMP_FILE_DIRS = [
    "embedded_subs",
    "embedded_json",
    "audio",
    "vocals",
    "transcripts",
    "transcripts_raw",
    "diarized",
    "music_marked",
    "translated",
    "ocr_raw",
    "ocr_translated",
    "notes",
    "music_tags",
]


def cleanup_video_temp(work_dir: Path, stem: str) -> int:
    removed = 0
    for dirname in TEMP_FILE_DIRS:
        for suffix in (".wav", ".json", ".pt"):
            path = work_dir / dirname / f"{stem}{suffix}"
            if path.exists():
                path.unlink()
                removed += 1

    for extra in (
        work_dir / "embedded_subs" / f"{stem}.eng.srt",
        work_dir / "embedded_subs" / f"{stem}.eng.sdh.srt",
    ):
        if extra.exists():
            extra.unlink()
            removed += 1

    demucs_item = work_dir / "demucs_raw" / "htdemucs_ft" / stem
    if demucs_item.exists():
        shutil.rmtree(demucs_item)
        removed += 1
    return removed


def remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    dirs = sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True)
    for path in dirs:
        try:
            path.rmdir()
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 STEP11: cleanup intermediate files.")
    add_common_args(parser)
    parser.add_argument("--all-work-dir", action="store_true", help="Remove the whole work directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    if os.environ.get("KEEP_TEMP", "") == "1":
        print("[skip] KEEP_TEMP=1; intermediate files kept")
        return
    if args.all_work_dir:
        if work_dir.exists():
            shutil.rmtree(work_dir)
            write_status(base_dir, "V2_STEP11", work_dir.name, "removed work directory")
            print(f"[done] removed work directory: {work_dir}")
        else:
            print(f"[skip] work directory does not exist: {work_dir}")
        return

    videos = selected_videos(base_dir, args.chunk, args.target_stem)
    if not videos:
        print(f"[error] no target videos found in {base_dir}")
        sys.exit(1)

    total = 0
    for video in videos:
        removed = cleanup_video_temp(work_dir, video.stem)
        total += removed
        print(f"[done] removed {removed} intermediate items for {video.stem}")
    remove_empty_dirs(work_dir)
    write_status(base_dir, "V2_STEP11", f"{len(videos)} files", f"removed {total} intermediate items")
    print(f"v2 step11 cleanup done; removed {total} item(s)")


if __name__ == "__main__":
    main()
