import argparse
import json
import sys
import time
from pathlib import Path

from common import add_common_args, selected_videos, work_dir_for, write_status, write_status_only
import ocr_extract_core as ocr_step


def existing_interval(path: Path) -> float | None:
    try:
        with path.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
        return float(data.get("interval"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def interval_matches(path: Path) -> bool:
    interval = existing_interval(path)
    return interval is not None and abs(interval - ocr_step.OCR_INTERVAL_SECONDS) < 0.001


def write_ocr_progress(base_dir: Path, video: str, index: int, total: int) -> None:
    progress = 100.0 if total <= 0 else index * 100.0 / total
    write_status_only(base_dir, "V2_STEP6", video, f"OCR frame scan {index}/{total} ({progress:.1f}%)")


def ocr_interval_arg(value: str) -> float:
    try:
        interval = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("OCR interval must be a number of seconds") from exc
    if interval < 0.5:
        raise argparse.ArgumentTypeError("OCR interval must be >= 0.5 seconds")
    return interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 STEP6: extract on-screen text into the v2 work directory.")
    add_common_args(parser)
    parser.add_argument(
        "--ocr-interval",
        type=ocr_interval_arg,
        default=ocr_step.OCR_INTERVAL_SECONDS,
        help="OCR frame sampling interval in seconds. Default: 0.5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ocr_step.configure_interval(args.ocr_interval)
    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    out_dir = work_dir / "ocr_raw"
    videos = selected_videos(base_dir, args.chunk, args.target_stem)
    if not videos:
        print(f"[error] no target videos found in {base_dir}")
        sys.exit(1)

    for video in videos:
        out_file = out_dir / f"{video.stem}.json"
        if out_file.exists() and interval_matches(out_file):
            print(f"[skip] {out_file.name} exists at interval={ocr_step.OCR_INTERVAL_SECONDS:g}s")
            continue
        if out_file.exists():
            old_interval = existing_interval(out_file)
            old_text = "unknown" if old_interval is None else f"{old_interval:g}s"
            print(f"[refresh] {out_file.name} interval {old_text} -> {ocr_step.OCR_INTERVAL_SECONDS:g}s")
        write_status(base_dir, "V2_STEP6", video, f"START OCR frame scan interval={ocr_step.OCR_INTERVAL_SECONDS:g}s")
        start = time.time()
        ocr_step.write_status = write_ocr_progress
        ocr_step.scan_video(base_dir, video, out_file)
        elapsed = time.time() - start
        write_status(base_dir, "V2_STEP6", video, f"DONE OCR frame scan ({elapsed:.1f}s)")

    print(f"v2 OCR done; raw OCR files saved in: {out_dir}")


if __name__ == "__main__":
    main()
