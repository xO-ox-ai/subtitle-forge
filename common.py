import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OLD_SCRIPT_DIR = SCRIPT_DIR / "old"
if OLD_SCRIPT_DIR.exists() and str(OLD_SCRIPT_DIR) not in sys.path:
    sys.path.append(str(OLD_SCRIPT_DIR))

from pipeline_common import configure_environment


configure_environment()


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"}
DEFAULT_WORK_DIR = "temp"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("base_dir", help="Directory containing videos")
    parser.add_argument(
        "--work-dir",
        default=DEFAULT_WORK_DIR,
        help="Intermediate directory. Relative paths are resolved under base_dir.",
    )
    parser.add_argument(
        "--target-stem",
        action="append",
        default=[],
        help="Exact video stem to process. Can be used multiple times.",
    )
    parser.add_argument(
        "--chunk",
        default="0",
        help="0/all = all videos, positive integer = chunk size, other text = episode selector such as s03e01.",
    )


def work_dir_for(base_dir: Path, work_dir: str | Path = DEFAULT_WORK_DIR) -> Path:
    path = Path(work_dir)
    if path.is_absolute():
        return path
    return base_dir / path


def runtime_file_for(base_dir: Path, name: str) -> Path:
    """Return a run-state path inside the default project work directory."""
    runtime_dir = work_dir_for(base_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir / name


def iter_videos(base_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in base_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )


def compact_video_name(video: str | Path | None) -> str:
    if not video:
        return ""
    try:
        return Path(video).name
    except Exception:
        return str(video)


def normalize_selector(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def video_matches_selector(video: Path, selector: str) -> bool:
    selector_key = normalize_selector(Path(selector).stem or selector)
    if not selector_key:
        return True
    haystacks = {
        normalize_selector(video.name),
        normalize_selector(video.stem),
    }
    return any(selector_key in item or item in selector_key for item in haystacks)


def selected_videos(
    base_dir: Path,
    chunk: str = "0",
    target_stems: list[str] | None = None,
) -> list[Path]:
    videos = iter_videos(base_dir)
    raw_targets = [str(stem) for stem in (target_stems or []) if stem]
    if raw_targets:
        selected = []
        for video in videos:
            for target in raw_targets:
                target_path = Path(target)
                candidates = {target, target_path.name}
                if target_path.suffix.lower() in VIDEO_EXTS:
                    candidates.add(target_path.stem)
                if video.stem in candidates or video.name in candidates or video_matches_selector(video, target):
                    selected.append(video)
                    break
        if not selected:
            raise FileNotFoundError(f"target stems not found: {', '.join(raw_targets)}")
        return selected

    chunk_text = str(chunk or "0").strip()
    if chunk_text.lower() in {"0", "all", "*"}:
        return videos
    if chunk_text.isdigit():
        return videos

    selected = [video for video in videos if video_matches_selector(video, chunk_text)]
    if not selected:
        raise FileNotFoundError(f"no video matches selector: {chunk_text}")
    return selected


def chunk_size_from(value: str) -> int:
    text = str(value or "0").strip().lower()
    if text in {"0", "all", "*"}:
        return 0
    if text.isdigit():
        return int(text)
    return 0


def chunked_videos(videos: list[Path], chunk: str) -> list[list[Path]]:
    size = chunk_size_from(chunk)
    if size <= 0:
        return [videos] if videos else []
    return [videos[index : index + size] for index in range(0, len(videos), size)]


def video_stems(videos: list[Path]) -> list[str]:
    return [video.stem for video in videos]


def filter_paths_by_stems(paths: list[Path], stems: set[str] | None) -> list[Path]:
    if not stems:
        return paths
    return [path for path in paths if path.stem in stems]


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        if path.stat().st_size == 0:
            return default
        with path.open(encoding="utf-8-sig") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_log(base_dir: Path, step: str, video: str | Path | None, message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    log_file = runtime_file_for(base_dir, "subtitle_run.log")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{now}] {step} | {compact_video_name(video)} | {message}\n")


def write_status(base_dir: Path, step: str, video: str | Path | None, message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    runtime_file_for(base_dir, "subtitle_status.txt").write_text(
        f"time: {now}\nstep: {step}\nvideo: {compact_video_name(video)}\nmessage: {message}\n",
        encoding="utf-8",
    )
    append_log(base_dir, step, video, message)


def write_status_only(base_dir: Path, step: str, video: str | Path | None, message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    runtime_file_for(base_dir, "subtitle_status.txt").write_text(
        f"time: {now}\nstep: {step}\nvideo: {compact_video_name(video)}\nmessage: {message}\n",
        encoding="utf-8",
    )


class StatusHeartbeat:
    def __init__(
        self,
        base_dir: Path,
        step: str,
        video: str | Path | None,
        message: str,
        interval_seconds: int = 30,
    ) -> None:
        self.base_dir = base_dir
        self.step = step
        self.video = video
        self.message = message
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()
        write_status_only(self.base_dir, self.step, self.video, f"{self.message}; elapsed=0s")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            elapsed = time.time() - self._start
            write_status_only(self.base_dir, self.step, self.video, f"{self.message}; elapsed={elapsed:.0f}s")


def status_preview(text: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", str(text or "").replace("\\N", " ")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    return parser.parse_args()
