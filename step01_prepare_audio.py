import subprocess
import sys
import time
from pathlib import Path

from common import parse_common_args, selected_videos, work_dir_for, write_status


def extract_audio(video: Path, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        str(video),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_file),
        "-y",
        "-loglevel",
        "error",
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"audio extraction failed: {video}")


def main() -> None:
    args = parse_common_args("V2 STEP1: extract audio for selected videos.")
    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    audio_dir = work_dir / "audio"
    videos = selected_videos(base_dir, args.chunk, args.target_stem)
    if not videos:
        print(f"[error] no target videos found in {base_dir}")
        sys.exit(1)

    processed = 0
    for video in videos:
        out_file = audio_dir / f"{video.stem}.wav"
        if out_file.exists():
            print(f"[skip] {out_file.name} exists")
            continue
        print(f"[process] {video.name}")
        write_status(base_dir, "V2_STEP1", video, "START extract audio")
        start = time.time()
        extract_audio(video, out_file)
        elapsed = time.time() - start
        write_status(base_dir, "V2_STEP1", video, f"DONE extract audio ({elapsed:.1f}s)")
        print(f"[done] {out_file}")
        processed += 1

    print(f"v2 step1 done; processed {processed} file(s); audio={audio_dir}")


if __name__ == "__main__":
    main()
