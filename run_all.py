import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# The main launcher imports every pipeline helper. Disable per-module bytecode
# before those imports so a normal `python run_all.py` invocation does not
# create a root-level __pycache__; child steps inherit the environment setting
# applied by configure_environment().
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

from pipeline_common import PYTHON_EXE, configure_environment, environment_for_script, python_for_script
from common import chunked_videos, selected_videos, video_stems, work_dir_for, write_status
from ass_polish_helpers import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    DEFAULT_POLISH_FILE_BATCH_SIZE,
)


configure_environment()


SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_CODE_RE = re.compile(r"s\d{2}e\d{2}", re.IGNORECASE)
KNOWN_TARGET_SUFFIXES = {
    ".ass", ".srt", ".vtt", ".ssa", ".sub",
    ".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm", ".ts",
}

PREP_STEPS = [
    ("V2_STEP1", "extract audio", "step01_prepare_audio.py"),
    ("V2_STEP2", "separate vocals", "step02_demucs.py"),
    ("V2_STEP3", "Whisper server transcription", "step03_whisper_server.py"),
    ("V2_STEP4", "speaker diarization", "step04_diarize.py"),
]

POST_STEPS = [
    ("V2_STEP8", "generate base ASS subtitles", "step08_ass_render.py"),
]

STEP_SEQUENCE = [
    "V2_STEP0",
    "V2_STEP1",
    "V2_STEP2",
    "V2_STEP3",
    "V2_STEP4",
    "V2_STEP5",
    "V2_STEP6",
    "V2_STEP7",
    "V2_STEP8",
    "V2_STEP9",
    "V2_STEP10",
    "V2_STEP11",
]

TEXT_SUBTITLE_CODECS = {"subrip", "ass", "ssa", "mov_text", "webvtt"}
EXTERNAL_SUBTITLE_EXTS = {".ass", ".ssa", ".srt", ".vtt"}
ASS_EXTS = {".ass", ".ssa"}
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
ASS_TAG_RE = re.compile(r"\{[^}]*\}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
ASS_TIME_RE = re.compile(r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<cs>\d{2})")
SRT_TIME_RE = re.compile(
    r"(?P<start>\d+:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d+:\d{2}:\d{2}[,.]\d{1,3})"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v2 subtitle generation pipeline.")
    parser.add_argument("base_dir", help="Directory containing videos")
    parser.add_argument(
        "--work-dir",
        default="temp",
        help="Intermediate directory. Relative paths are resolved under base_dir.",
    )
    parser.add_argument(
        "--chunk",
        default="0",
        help="0/all = all videos in one model session, positive integer = chunk size, other text = episode selector.",
    )
    parser.add_argument(
        "--target-stem",
        action="append",
        default=[],
        help="Exact video stem to process. Can be used multiple times.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "audio", "embedded"],
        default="auto",
        help=(
            "Subtitle source. auto groups videos by existing subtitles: bilingual ASS goes to Step9, "
            "external monolingual subtitles are imported then Step6-Step8, embedded subtitles use Step0 then Step6-Step8, "
            "and videos without subtitles use Step1-Step8."
        ),
    )
    parser.add_argument("--force-embedded", action="store_true", help="Regenerate embedded subtitle extraction outputs.")
    parser.add_argument("--keep-sdh-cues", action="store_true", help="Keep SDH-only non-dialogue cues from embedded subtitles.")
    parser.add_argument("--skip-music", action="store_true", help="Skip music HMM tagging on the audio-recognition path.")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR extraction, OCR translation, notes, and overlay.")
    parser.add_argument(
        "--ocr-interval",
        type=float,
        default=0.5,
        help="OCR frame sampling interval in seconds for Step6. Default: 0.5.",
    )
    parser.add_argument(
        "--ocr-over-embedded",
        action="store_true",
        help="Also run OCR extraction + overlay for embedded-subtitle sources (normally skipped because dialogue already comes from SDH).",
    )
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality report and manifest generation.")
    parser.add_argument("--force-dialogue", action="store_true", help="Force Step07 to retranslate dialogue/lyrics.")
    parser.add_argument("--force-ocr", action="store_true", help="Force Step07 to regenerate OCR translations and notes.")
    parser.add_argument(
        "--no-polish",
        action="store_true",
        help="Skip Step09 finished-ASS translation polish (enabled by default).",
    )
    parser.add_argument(
        "--polish-provider",
        choices=["openai", "codex-cli"],
        default="codex-cli",
        help="Step09 polish provider. Default: codex-cli.",
    )
    parser.add_argument(
        "--polish-model",
        default="",
        help="Step09 polish model. Codex CLI defaults to gpt-5.6-sol; other providers use step09's built-in default.",
    )
    parser.add_argument(
        "--polish-base-url",
        default="",
        help="Step09 polish OpenAI-compatible /v1 base URL. Defaults to SUB_POLISH_BASE_URL or OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--polish-batch-size",
        type=int,
        default=0,
        help="Step09 polish batch size. 0 uses step09 default.",
    )
    parser.add_argument(
        "--polish-file-batch-size",
        type=int,
        default=DEFAULT_POLISH_FILE_BATCH_SIZE,
        help="Step09 ASS files per durable polish batch. Default: 1 (write/cache each file before continuing).",
    )
    parser.add_argument(
        "--polish-styles",
        default="",
        help="Comma-separated Step09 ASS styles to polish, or all. Empty uses step09 default.",
    )
    parser.add_argument("--polish-force", action="store_true", help="Ignore Step09 polish cache and re-polish current ASS text.")
    parser.add_argument(
        "--polish-codex-reasoning-effort",
        default="",
        help="Codex CLI reasoning effort for --polish-provider codex-cli. Default: high.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        default=True,
        help="Keep v2 intermediate files after successful run. This is the default.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_false",
        dest="no_cleanup",
        help="Remove v2 intermediate files after a successful run.",
    )
    parser.add_argument("--report-screenshots", action="store_true", help="Capture quality-report screenshots in step10.")
    parser.add_argument("--max-report-screenshots", type=int, default=8, help="Maximum report screenshots per video.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them.")
    parser.add_argument("--start-at", default="", help="Start at a step id, e.g. V2_STEP3 or V2_STEP5.")
    parser.add_argument("--stop-after", default="", help="Stop after a step id, e.g. V2_STEP6.")
    return parser.parse_args()


def run_step(
    base_dir: Path,
    work_dir: str,
    step: str,
    label: str,
    script_name: str,
    stems: list[str],
    extra_args: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    step_python = python_for_script(script_name)
    cmd = [str(step_python), str(SCRIPT_DIR / script_name), str(base_dir), "--work-dir", work_dir, "--chunk", "0"]
    for stem in stems:
        cmd.extend(["--target-stem", stem])
    if extra_args:
        cmd.extend(extra_args)

    pretty = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print(f"[{step}] {label}")
    print(f"  {pretty}")
    if dry_run:
        return

    start = time.time()
    result = subprocess.run(cmd, env=environment_for_script(script_name))
    elapsed = time.time() - start
    if result.returncode != 0:
        write_status(base_dir, step, f"{len(stems)} files", f"FAILED {label} ({elapsed:.1f}s)")
        raise RuntimeError(f"step failed: {step} ({label})")


def should_run(step: str, start_at: str, stop_after: str, seen: dict[str, bool]) -> bool:
    if start_at and not seen["started"]:
        if step == start_at:
            seen["started"] = True
        else:
            return False
    if not start_at:
        seen["started"] = True
    if seen.get("stopped"):
        return False
    return True


def mark_stop(step: str, stop_after: str, seen: dict[str, bool]) -> None:
    if stop_after and step == stop_after:
        seen["stopped"] = True


def embedded_stems(work_dir: Path, stems: list[str]) -> list[str]:
    embedded_dir = work_dir / "embedded_json"
    return [stem for stem in stems if (embedded_dir / f"{stem}.json").exists()]


def without_stems(stems: list[str], excluded: list[str]) -> list[str]:
    excluded_set = set(excluded)
    return [stem for stem in stems if stem not in excluded_set]


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def has_han(text: str) -> bool:
    return bool(HAN_RE.search(str(text or "")))


def has_latin(text: str) -> bool:
    return bool(LATIN_RE.search(str(text or "")))


def external_subtitle_candidates(video: Path) -> list[Path]:
    candidates = [
        path
        for path in video.parent.iterdir()
        if path.is_file()
        and path.suffix.lower() in EXTERNAL_SUBTITLE_EXTS
        and (path.stem == video.stem or path.name.startswith(f"{video.stem}."))
    ]
    priority = {".ass": 0, ".ssa": 1, ".srt": 2, ".vtt": 3}
    return sorted(candidates, key=lambda path: (priority.get(path.suffix.lower(), 99), path.name.lower()))


def ass_visible_text(line: str) -> str:
    text = str(line or "")
    if line.startswith("Dialogue:"):
        parts = line.split(",", 9)
        if len(parts) == 10:
            text = parts[9]
    text = ASS_TAG_RE.sub("", text)
    return text.replace(r"\N", " ").replace(r"\h", " ").strip()


def is_bilingual_ass(path: Path) -> bool:
    if path.suffix.lower() not in ASS_EXTS:
        return False
    text = read_text(path)
    visible = "\n".join(ass_visible_text(line) for line in text.splitlines() if line.startswith("Dialogue:"))
    if not has_han(visible) or not has_latin(visible):
        return False
    if re.search(r"Style:\s*BILINGUAL\b|,BILINGUAL(?:_MUSIC|_CHANT)?\s*,", text):
        return True
    return bool(re.search(r"\\N\s*\{\\fs\d+", text))


def find_bilingual_ass(video: Path) -> Path | None:
    return next((path for path in external_subtitle_candidates(video) if is_bilingual_ass(path)), None)


def find_external_mono_subtitle(video: Path) -> Path | None:
    return next((path for path in external_subtitle_candidates(video) if not is_bilingual_ass(path)), None)


def video_has_embedded_text_subtitle(video: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "s",
        "-show_streams",
        "-of",
        "json",
        str(video),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(f"  [warning] ffprobe subtitle scan failed for {video.name}: {result.stderr.strip()}")
        return False
    try:
        streams = json.loads(result.stdout or "{}").get("streams", [])
    except json.JSONDecodeError:
        return False
    for stream in streams:
        codec = str(stream.get("codec_name", "")).lower()
        if codec not in TEXT_SUBTITLE_CODECS:
            continue
        tags = stream.get("tags") or {}
        lang = str(tags.get("language", "")).strip().lower()
        title = str(tags.get("title", "")).strip().lower()
        if lang in {"", "eng", "en", "english"} or "english" in title:
            return True
    return False


def parse_ass_time(value: str) -> float | None:
    match = ASS_TIME_RE.fullmatch(str(value or "").strip())
    if not match:
        return None
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + int(match.group("cs")) / 100.0
    )


def parse_srt_time(value: str) -> float:
    head, fraction = value.replace(",", ".").split(".", 1)
    hours, minutes, seconds = [int(part) for part in head.split(":")]
    millis = int((fraction + "000")[:3])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def clean_external_text(text: str) -> str:
    text = HTML_TAG_RE.sub("", str(text or ""))
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(r"\N", "\n").replace(r"\h", " ")
    return " ".join(part.strip() for part in text.splitlines() if part.strip()).strip()


def make_external_segment(start: float, end: float, text: str, index: int, source: str) -> dict | None:
    text = clean_external_text(text)
    if not text or not has_latin(text):
        return None
    is_music = "♪" in text or "♫" in text
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "text": text,
        "en_wrap": text,
        "source": source,
        "kind": "lyric" if is_music else "dialogue",
        "is_music": is_music,
        "is_chant": False,
        "external_index": index,
    }


def parse_external_ass(path: Path) -> list[dict]:
    segments: list[dict] = []
    for line in read_text(path).splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) != 10:
            continue
        style = parts[3].strip().upper()
        if style in {"OCR_TRANSLATION", "EXPLANATION_NOTE"}:
            continue
        start = parse_ass_time(parts[1])
        end = parse_ass_time(parts[2])
        if start is None or end is None or end <= start:
            continue
        text = ass_visible_text(parts[9])
        segment = make_external_segment(start, end, text, len(segments) + 1, "external_ass")
        if segment:
            segments.append(segment)
    return segments


def parse_external_srt_or_vtt(path: Path) -> list[dict]:
    text = read_text(path).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*WEBVTT[^\n]*\n+", "", text, flags=re.IGNORECASE)
    blocks = re.split(r"\n\s*\n", text.strip())
    segments: list[dict] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_index = next((index for index, line in enumerate(lines) if "-->" in line), -1)
        if time_index < 0:
            continue
        match = SRT_TIME_RE.search(lines[time_index])
        if not match:
            continue
        body = "\n".join(lines[time_index + 1 :])
        segment = make_external_segment(
            parse_srt_time(match.group("start")),
            parse_srt_time(match.group("end")),
            body,
            len(segments) + 1,
            f"external_{path.suffix.lower().lstrip('.')}",
        )
        if segment:
            segments.append(segment)
    return segments


def parse_external_subtitle(path: Path) -> list[dict]:
    if path.suffix.lower() in ASS_EXTS:
        return parse_external_ass(path)
    return parse_external_srt_or_vtt(path)


def import_external_subtitles(base_dir: Path, work_dir: Path, videos: list[Path], force: bool, dry_run: bool) -> list[str]:
    imported: list[str] = []
    embedded_json_dir = work_dir / "embedded_json"
    for video in videos:
        subtitle = find_external_mono_subtitle(video)
        if not subtitle:
            continue
        out_file = embedded_json_dir / f"{video.stem}.json"
        if out_file.exists() and not force:
            print(f"[external] reuse {out_file.name} from existing embedded_json")
            imported.append(video.stem)
            continue
        if dry_run:
            print(f"[external] would import {subtitle.name} -> {out_file}")
            imported.append(video.stem)
            continue
        segments = parse_external_subtitle(subtitle)
        if not segments:
            raise RuntimeError(f"external subtitle has no usable English cues: {subtitle.name}")
        embedded_json_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "video": video.name,
            "stem": video.stem,
            "source": "external_subtitle",
            "external_subtitle": subtitle.name,
            "counts": {
                "segments": len(segments),
                "music_segments": sum(1 for segment in segments if segment.get("is_music")),
                "chant_segments": 0,
            },
            "segments": segments,
        }
        out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[external] imported {subtitle.name}: segments={len(segments)}")
        imported.append(video.stem)
    if imported and not dry_run:
        write_status(base_dir, "V2_EXTERNAL", f"{len(imported)} files", "imported external monolingual subtitles")
    return imported


def run_qwen_step(
    base_dir: Path,
    work_dir: str,
    stems: list[str],
    source: str,
    skip_ocr: bool,
    force_dialogue: bool,
    force_ocr: bool,
    ocr_over_embedded: bool,
    dry_run: bool,
) -> None:
    if not stems:
        return
    extra = ["--source", source]
    # For embedded sources we normally skip OCR entirely (dialogue comes from
    # SDH). With --ocr-over-embedded we instead keep OCR on and let step07
    # translate the OCR files alongside the dialogue.
    # skip_ocr reflects the user's --skip-ocr flag. The embedded branch
    # normally skips OCR (dialogue comes from SDH), but --ocr-over-embedded
    # re-enables it so title/location/ad cards get translated too.
    if skip_ocr or (source == "embedded" and not ocr_over_embedded):
        extra.append("--skip-ocr")
    if ocr_over_embedded and source == "embedded":
        extra.append("--ocr-over-embedded")
    if force_dialogue:
        extra.append("--force-dialogue")
    if force_ocr and (source != "embedded" or ocr_over_embedded):
        extra.append("--force-ocr")
    label = "Qwen embedded subtitle translation" if source == "embedded" else "single Qwen dialogue/OCR/notes session"
    run_step(base_dir, work_dir, "V2_STEP7", label, "step07_qwen_all.py", stems, extra, dry_run)


def step9_extra_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if not args.no_polish:
        extra.append("--polish-backend")
        extra.extend(["--polish-provider", args.polish_provider])
        effective_model = args.polish_model or (DEFAULT_CODEX_MODEL if args.polish_provider == "codex-cli" else "")
        if effective_model:
            extra.extend(["--polish-model", effective_model])
        if args.polish_base_url:
            extra.extend(["--polish-base-url", args.polish_base_url])
        if args.polish_batch_size > 0:
            extra.extend(["--polish-batch-size", str(args.polish_batch_size)])
        if args.polish_file_batch_size > 0:
            extra.extend(["--polish-file-batch-size", str(args.polish_file_batch_size)])
        if args.polish_styles:
            extra.extend(["--polish-styles", args.polish_styles])
        if args.polish_force:
            extra.append("--polish-force")
        effective_reasoning = args.polish_codex_reasoning_effort or DEFAULT_CODEX_REASONING_EFFORT
        if args.polish_provider == "codex-cli":
            extra.extend(["--polish-codex-reasoning-effort", effective_reasoning])
    return extra


def step_index(step: str) -> int:
    try:
        return STEP_SEQUENCE.index((step or "").upper())
    except ValueError:
        return -1


def allows_ass_only_run(start_at: str) -> bool:
    start = (start_at or "").strip().upper()
    if not start:
        return False
    return step_index(start) >= step_index("V2_STEP9")


def episode_code(value: str | Path) -> str:
    match = EPISODE_CODE_RE.search(str(value))
    return match.group(0).upper() if match else ""


def target_selector_text(target: str) -> str:
    target_path = Path(target)
    if target_path.suffix.lower() in KNOWN_TARGET_SUFFIXES:
        return target_path.stem
    return target_path.name or target


def ass_files_for_auto(base_dir: Path, target_stems: list[str] | None = None, chunk: str = "0") -> list[Path]:
    ass_files = sorted(path for path in base_dir.glob("*.ass") if path.is_file())
    raw_targets = [str(stem) for stem in (target_stems or []) if stem]
    if raw_targets:
        selected: list[Path] = []
        for path in ass_files:
            for target in raw_targets:
                target_ep = episode_code(target)
                if target_ep:
                    matched = target_ep == episode_code(path.name)
                else:
                    selector = target_selector_text(target).lower()
                    matched = selector == path.name.lower() or selector == path.stem.lower() or selector in path.stem.lower()
                if matched:
                    selected.append(path)
                    break
        return selected

    chunk_text = str(chunk or "0").strip().lower()
    if chunk_text in {"0", "all", "*"} or chunk_text.isdigit():
        return ass_files
    return [path for path in ass_files if chunk_text in path.name.lower() or chunk_text in path.stem.lower()]


def auto_groups(videos: list[Path]) -> dict[str, list[Path]]:
    groups = {
        "ready_bilingual": [],
        "external_mono": [],
        "embedded_only": [],
        "no_subtitles": [],
    }
    for video in videos:
        if find_bilingual_ass(video):
            groups["ready_bilingual"].append(video)
        elif find_external_mono_subtitle(video):
            groups["external_mono"].append(video)
        elif video_has_embedded_text_subtitle(video):
            groups["embedded_only"].append(video)
        else:
            groups["no_subtitles"].append(video)
    return groups


def print_group(label: str, videos: list[Path]) -> None:
    print(f"{label}: {len(videos)}")
    for video in videos:
        print(f"  {video.name}")


def run_audio_generation_group(
    args: argparse.Namespace,
    base_dir: Path,
    videos: list[Path],
    seen: dict[str, bool],
) -> None:
    stems = video_stems(videos)
    if not stems:
        return
    for step, label, script in PREP_STEPS:
        if should_run(step, args.start_at, args.stop_after, seen):
            run_step(base_dir, args.work_dir, step, label, script, stems, dry_run=args.dry_run)
            mark_stop(step, args.stop_after, seen)
    if not args.skip_music and should_run("V2_STEP5", args.start_at, args.stop_after, seen):
        run_step(base_dir, args.work_dir, "V2_STEP5", "music HMM tagging", "step05_music_hmm.py", stems, dry_run=args.dry_run)
        mark_stop("V2_STEP5", args.stop_after, seen)
    if not args.skip_ocr and should_run("V2_STEP6", args.start_at, args.stop_after, seen):
        run_step(
            base_dir,
            args.work_dir,
            "V2_STEP6",
            "extract on-screen OCR text",
            "step06_ocr_extract.py",
            stems,
            ["--ocr-interval", str(args.ocr_interval)],
            dry_run=args.dry_run,
        )
        mark_stop("V2_STEP6", args.stop_after, seen)
    if should_run("V2_STEP7", args.start_at, args.stop_after, seen):
        run_qwen_step(
            base_dir,
            args.work_dir,
            stems,
            "audio",
            args.skip_ocr,
            args.force_dialogue,
            args.force_ocr,
            False,
            args.dry_run,
        )
        mark_stop("V2_STEP7", args.stop_after, seen)
    if should_run("V2_STEP8", args.start_at, args.stop_after, seen):
        run_step(base_dir, args.work_dir, "V2_STEP8", "generate base ASS subtitles", "step08_ass_render.py", stems, dry_run=args.dry_run)
        mark_stop("V2_STEP8", args.stop_after, seen)


def run_embedded_generation_group(
    args: argparse.Namespace,
    base_dir: Path,
    videos: list[Path],
    seen: dict[str, bool],
) -> None:
    stems = video_stems(videos)
    if not stems:
        return
    if should_run("V2_STEP0", args.start_at, args.stop_after, seen):
        extra = ["--prefer-sdh"]
        if args.force_embedded:
            extra.append("--force")
        if args.keep_sdh_cues:
            extra.append("--keep-sdh-cues")
        run_step(
            base_dir,
            args.work_dir,
            "V2_STEP0",
            "extract embedded English subtitles, preferring SDH",
            "step00_extract_embedded_subs.py",
            stems,
            extra,
            args.dry_run,
        )
        mark_stop("V2_STEP0", args.stop_after, seen)
    run_external_or_embedded_post_steps(args, base_dir, stems, seen)


def run_external_or_embedded_post_steps(
    args: argparse.Namespace,
    base_dir: Path,
    stems: list[str],
    seen: dict[str, bool],
) -> None:
    if not stems:
        return
    if not args.skip_ocr and should_run("V2_STEP6", args.start_at, args.stop_after, seen):
        run_step(
            base_dir,
            args.work_dir,
            "V2_STEP6",
            "extract on-screen OCR text",
            "step06_ocr_extract.py",
            stems,
            ["--ocr-interval", str(args.ocr_interval)],
            dry_run=args.dry_run,
        )
        mark_stop("V2_STEP6", args.stop_after, seen)
    if should_run("V2_STEP7", args.start_at, args.stop_after, seen):
        run_qwen_step(
            base_dir,
            args.work_dir,
            stems,
            "embedded",
            args.skip_ocr,
            args.force_dialogue,
            args.force_ocr,
            True,
            args.dry_run,
        )
        mark_stop("V2_STEP7", args.stop_after, seen)
    if should_run("V2_STEP8", args.start_at, args.stop_after, seen):
        run_step(base_dir, args.work_dir, "V2_STEP8", "generate base ASS subtitles", "step08_ass_render.py", stems, dry_run=args.dry_run)
        mark_stop("V2_STEP8", args.stop_after, seen)


def run_smart_auto_flow(
    args: argparse.Namespace,
    base_dir: Path,
    work_dir: Path,
    videos: list[Path],
    seen: dict[str, bool],
) -> None:
    groups = auto_groups(videos)
    print()
    print("========================================")
    print("Auto subtitle grouping")
    print_group("Already has bilingual ASS; Step09 only", groups["ready_bilingual"])
    print_group("Has external monolingual subtitles; import then Step6-Step8", groups["external_mono"])
    print_group("Has embedded subtitles only; Step0 then Step6-Step8", groups["embedded_only"])
    print_group("No embedded/external subtitles; Step1-Step8", groups["no_subtitles"])
    print("========================================")

    if groups["no_subtitles"]:
        print("\n[auto] Group 1: no subtitles -> Step1-Step8")
        run_audio_generation_group(args, base_dir, groups["no_subtitles"], seen)

    if groups["embedded_only"] and not seen.get("stopped"):
        print("\n[auto] Group 2: embedded subtitles -> Step0, Step6-Step8")
        run_embedded_generation_group(args, base_dir, groups["embedded_only"], seen)

    if groups["external_mono"] and not seen.get("stopped"):
        print("\n[auto] Group 3: external monolingual subtitles -> import, Step6-Step8")
        imported = import_external_subtitles(base_dir, work_dir, groups["external_mono"], args.force_embedded, args.dry_run)
        run_external_or_embedded_post_steps(args, base_dir, imported, seen)

    all_stems = video_stems(videos)
    if all_stems and not seen.get("stopped"):
        step9_extra = step9_extra_args(args)
        if should_run("V2_STEP9", args.start_at, args.stop_after, seen):
            label = "polish all ASS subtitles with backend model" if not args.no_polish else "merge OCR/notes overlays and filter ASS"
            run_step(
                base_dir,
                args.work_dir,
                "V2_STEP9",
                label,
                "step09_overlay_filter.py",
                all_stems,
                step9_extra,
                dry_run=args.dry_run,
            )
            mark_stop("V2_STEP9", args.stop_after, seen)

    if all_stems and not args.skip_quality and not seen.get("stopped") and should_run("V2_STEP10", args.start_at, args.stop_after, seen):
        extra = ["--max-screenshots", str(args.max_report_screenshots)]
        if args.report_screenshots:
            extra.append("--screenshots")
        run_step(base_dir, args.work_dir, "V2_STEP10", "quality report and manifest", "step10_quality_report.py", all_stems, extra, args.dry_run)
        mark_stop("V2_STEP10", args.stop_after, seen)

    if all_stems and not args.no_cleanup and not seen.get("stopped") and should_run("V2_STEP11", args.start_at, args.stop_after, seen):
        run_step(base_dir, args.work_dir, "V2_STEP11", "remove v2 intermediate files", "step11_cleanup.py", all_stems, dry_run=args.dry_run)
        mark_stop("V2_STEP11", args.stop_after, seen)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    if not base_dir.exists():
        print(f"[error] directory not found: {base_dir}")
        sys.exit(1)

    work_dir = work_dir_for(base_dir, args.work_dir)
    auto_ass_files: list[Path] = []
    try:
        videos = selected_videos(base_dir, args.chunk, args.target_stem)
    except FileNotFoundError:
        if allows_ass_only_run(args.start_at):
            videos = []
        else:
            raise
    chunks = chunked_videos(videos, args.chunk)
    if not chunks:
        auto_ass_files = ass_files_for_auto(base_dir, args.target_stem, args.chunk)
        if args.source == "auto" and not args.start_at and auto_ass_files:
            print("[info] no target videos found; ASS-only auto flow will run Step09")
            chunks = [[]]
        elif allows_ass_only_run(args.start_at):
            print("[info] no target videos found; continuing from ASS-only flow")
            chunks = [[]]
        else:
            print(f"[error] no target videos found in {base_dir}")
            sys.exit(1)

    print("========================================")
    print("Subtitle pipeline v2")
    print(f"Directory: {base_dir}")
    print(f"Work dir: {work_dir}")
    print(f"Python: {PYTHON_EXE}")
    print(f"Videos: {len(videos)}")
    print(f"ASS-only files: {len(auto_ass_files)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Source: {args.source}")
    print(f"OCR: {'disabled' if args.skip_ocr else 'enabled'}")
    print(f"Music HMM: {'disabled' if args.skip_music else 'enabled'}")
    print(f"Quality: {'disabled' if args.skip_quality else 'enabled'}")
    print(f"Backend polish: {'disabled' if args.no_polish else 'enabled'}")
    print(f"Cleanup: {'disabled' if args.no_cleanup else 'enabled'}")
    print(f"Auto grouping: {'enabled' if args.source == 'auto' and not args.start_at and (bool(videos) or bool(auto_ass_files)) else 'disabled'}")
    print("========================================")

    if args.dry_run:
        print("[dry-run] no commands will be executed")

    seen = {"started": False, "stopped": False}
    start = time.time()
    try:
        if args.source == "auto" and not args.start_at and auto_ass_files and not videos:
            step9_extra = step9_extra_args(args)
            if should_run("V2_STEP9", args.start_at, args.stop_after, seen):
                run_step(
                    base_dir,
                    args.work_dir,
                    "V2_STEP9",
                    "polish ASS-only subtitles with backend model",
                    "step09_overlay_filter.py",
                    [path.stem for path in auto_ass_files],
                    step9_extra,
                    dry_run=args.dry_run,
                )
                mark_stop("V2_STEP9", args.stop_after, seen)
            if not args.no_cleanup and not seen.get("stopped") and should_run("V2_STEP11", args.start_at, args.stop_after, seen):
                run_step(base_dir, args.work_dir, "V2_STEP11", "remove v2 intermediate files", "step11_cleanup.py", [path.stem for path in auto_ass_files], dry_run=args.dry_run)
                mark_stop("V2_STEP11", args.stop_after, seen)
        elif args.source == "auto" and not args.start_at and videos:
            run_smart_auto_flow(args, base_dir, work_dir, videos, seen)
        else:
            for chunk_index, chunk_videos in enumerate(chunks, start=1):
                stems = video_stems(chunk_videos)
                print()
                print("========================================")
                print(f"Chunk {chunk_index}/{len(chunks)} | videos={len(stems)}")
                for video in chunk_videos:
                    print(f"  {video.name}")
                if not chunk_videos:
                    print("  [ASS-only chunk]")
                print("========================================")

                if should_run("V2_STEP0", args.start_at, args.stop_after, seen):
                    extra = []
                    if args.force_embedded:
                        extra.append("--force")
                    if args.keep_sdh_cues:
                        extra.append("--keep-sdh-cues")
                    run_step(base_dir, args.work_dir, "V2_STEP0", "extract embedded English subtitles", "step00_extract_embedded_subs.py", stems, extra, args.dry_run)
                    mark_stop("V2_STEP0", args.stop_after, seen)

                embedded_ready = embedded_stems(work_dir, stems)
                audio_stems = stems
                if args.source == "embedded":
                    audio_stems = []
                elif args.source == "auto":
                    audio_stems = without_stems(stems, embedded_ready)

                # Stems that receive on-screen OCR extraction + overlay. Normally
                # only the audio path (no embedded SDH); --ocr-over-embedded also
                # OCRs the embedded-subtitle stems so title/location/ad cards get
                # translated too.
                ocr_stems = list(audio_stems)
                if args.ocr_over_embedded and not args.skip_ocr:
                    for stem in embedded_ready:
                        if stem not in ocr_stems:
                            ocr_stems.append(stem)

                if audio_stems:
                    for step, label, script in PREP_STEPS:
                        if should_run(step, args.start_at, args.stop_after, seen):
                            run_step(base_dir, args.work_dir, step, label, script, audio_stems, dry_run=args.dry_run)
                            mark_stop(step, args.stop_after, seen)

                    if not args.skip_music and should_run("V2_STEP5", args.start_at, args.stop_after, seen):
                        run_step(base_dir, args.work_dir, "V2_STEP5", "music HMM tagging", "step05_music_hmm.py", audio_stems, dry_run=args.dry_run)
                        mark_stop("V2_STEP5", args.stop_after, seen)

                if not args.skip_ocr and ocr_stems and should_run("V2_STEP6", args.start_at, args.stop_after, seen):
                    run_step(
                        base_dir,
                        args.work_dir,
                        "V2_STEP6",
                        "extract on-screen OCR text",
                        "step06_ocr_extract.py",
                        ocr_stems,
                        ["--ocr-interval", str(args.ocr_interval)],
                        dry_run=args.dry_run,
                    )
                    mark_stop("V2_STEP6", args.stop_after, seen)

                if should_run("V2_STEP7", args.start_at, args.stop_after, seen):
                    if args.source == "embedded":
                        run_qwen_step(
                            base_dir,
                            args.work_dir,
                            stems,
                            "embedded",
                            args.skip_ocr,
                            args.force_dialogue,
                            False,
                            args.ocr_over_embedded,
                            args.dry_run,
                        )
                    elif args.source == "audio":
                        run_qwen_step(
                            base_dir,
                            args.work_dir,
                            stems,
                            "audio",
                            args.skip_ocr,
                            args.force_dialogue,
                            args.force_ocr,
                            args.ocr_over_embedded,
                            args.dry_run,
                        )
                    else:
                        if embedded_ready:
                            run_qwen_step(
                                base_dir,
                                args.work_dir,
                                embedded_ready,
                                "embedded",
                                args.skip_ocr,
                                args.force_dialogue,
                                False,
                                args.ocr_over_embedded,
                                args.dry_run,
                            )
                        if audio_stems:
                            run_qwen_step(
                                base_dir,
                                args.work_dir,
                                audio_stems,
                                "audio",
                                args.skip_ocr,
                                args.force_dialogue,
                                args.force_ocr,
                                args.ocr_over_embedded,
                                args.dry_run,
                            )
                    mark_stop("V2_STEP7", args.stop_after, seen)

                for step, label, script in POST_STEPS:
                    if should_run(step, args.start_at, args.stop_after, seen):
                        run_step(base_dir, args.work_dir, step, label, script, stems, dry_run=args.dry_run)
                        mark_stop(step, args.stop_after, seen)

                step9_stems = stems if (not args.skip_ocr or not args.no_polish) else []
                step9_extra = step9_extra_args(args)
                if not step9_stems and allows_ass_only_run(args.start_at):
                    # step09 can select .ass files directly when there are no videos,
                    # but still needs the user's original selector for single-file runs.
                    step9_stems = list(args.target_stem)
                if (step9_stems or allows_ass_only_run(args.start_at)) and should_run("V2_STEP9", args.start_at, args.stop_after, seen):
                    label = "merge OCR/notes overlays, filter ASS, and polish translations" if not args.no_polish else "merge OCR/notes overlays and filter ASS"
                    run_step(
                        base_dir,
                        args.work_dir,
                        "V2_STEP9",
                        label,
                        "step09_overlay_filter.py",
                        step9_stems,
                        step9_extra,
                        dry_run=args.dry_run,
                    )
                    mark_stop("V2_STEP9", args.stop_after, seen)

                if not args.skip_quality and should_run("V2_STEP10", args.start_at, args.stop_after, seen):
                    extra = ["--max-screenshots", str(args.max_report_screenshots)]
                    if args.report_screenshots:
                        extra.append("--screenshots")
                    run_step(base_dir, args.work_dir, "V2_STEP10", "quality report and manifest", "step10_quality_report.py", stems, extra, args.dry_run)
                    mark_stop("V2_STEP10", args.stop_after, seen)

                if not args.no_cleanup and should_run("V2_STEP11", args.start_at, args.stop_after, seen):
                    run_step(base_dir, args.work_dir, "V2_STEP11", "remove v2 intermediate files", "step11_cleanup.py", stems, dry_run=args.dry_run)
                    mark_stop("V2_STEP11", args.stop_after, seen)

        elapsed = time.time() - start
        if not args.dry_run:
            write_status(base_dir, "V2_DONE", "", f"all done; total={elapsed:.1f}s")
        print("========================================")
        print(f"V2 done; total elapsed: {elapsed:.1f}s")
        print("========================================")
    except Exception as exc:
        if not args.dry_run:
            write_status(base_dir, "V2_ERROR", "", str(exc))
        print(f"[error] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
