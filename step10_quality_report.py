import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from common import add_common_args, filter_paths_by_stems, load_json, save_json, selected_videos, work_dir_for, write_status
from pipeline_common import FFMPEG_DIR, exe_path
from step07_qwen_all import TRADITIONAL_TO_SIMPLIFIED
from step08_ass_render import display_parts, normalize_plain_text, semantic_display_parts


BASE_STYLES = {"BILINGUAL", "BILINGUAL_MUSIC", "BILINGUAL_CHANT"}
SCRIPT_FILES = [
    "run_all.py",
    "common.py",
    "pipeline_common.py",
    "qwen_common.py",
    "override_utils.py",
    "step00_extract_embedded_subs.py",
    "step01_prepare_audio.py",
    "step02_demucs.py",
    "step03_whisper_server.py",
    "step04_diarize.py",
    "step05_music_hmm.py",
    "step06_ocr_extract.py",
    "ocr_extract_core.py",
    "step07_qwen_all.py",
    "ocr_translate_helpers.py",
    "step08_ass_render.py",
    "ass_overlay_helpers.py",
    "step09_overlay_filter.py",
    "ass_filter_helpers.py",
    "step10_quality_report.py",
    "step11_cleanup.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 STEP10: quality report, manifest, and optional screenshots.")
    add_common_args(parser)
    parser.add_argument("--screenshots", action="store_true", help="Capture report screenshots with final ASS burned in.")
    parser.add_argument("--max-screenshots", type=int, default=8, help="Maximum screenshots per video.")
    return parser.parse_args()


def fmt_time(value: float) -> str:
    value = max(0.0, float(value))
    return f"{int(value // 3600):02d}:{int((value % 3600) // 60):02d}:{value % 60:05.2f}"


def parse_ass_time(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, centiseconds = rest.split(".")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(centiseconds) / 100.0


def clean_ass_text(text: str) -> list[str]:
    clean = re.sub(r"\{[^}]*\}", "", str(text or ""))
    return clean.split(r"\N")


def parse_ass_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    fields: list[str] = []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("Format:"):
            fields = [item.strip() for item in line.split(":", 1)[1].split(",")]
        elif line.startswith("Dialogue:") and fields:
            parts = line.split(":", 1)[1].lstrip().split(",", len(fields) - 1)
            if len(parts) != len(fields):
                continue
            event = dict(zip(fields, parts))
            try:
                event["start_seconds"] = parse_ass_time(event.get("Start", "0:00:00.00"))
                event["end_seconds"] = parse_ass_time(event.get("End", "0:00:00.00"))
            except Exception:
                continue
            event["lines"] = clean_ass_text(event.get("Text", ""))
            event["visible"] = " | ".join(event["lines"])
            events.append(event)
    return events


def ass_stats(events: list[dict]) -> dict:
    base = [event for event in events if event.get("Style") in BASE_STYLES]
    overlaps = []
    previous = None
    for event in base:
        if previous and event["start_seconds"] < previous["end_seconds"]:
            overlaps.append(
                {
                    "start": event["start_seconds"],
                    "overlap": round(previous["end_seconds"] - event["start_seconds"], 3),
                    "previous": previous["visible"],
                    "current": event["visible"],
                }
            )
        previous = event

    hard_line_events = [event for event in base if len(event.get("lines", [])) > 2]
    very_long = []
    for event in base:
        lines = event.get("lines", [])
        zh = lines[0] if lines else ""
        en = lines[1] if len(lines) > 1 else ""
        if len(zh) > 38 or len(en) > 96:
            very_long.append(
                {
                    "start": event["start_seconds"],
                    "end": event["end_seconds"],
                    "zh_len": len(zh),
                    "en_len": len(en),
                    "text": event["visible"],
                }
            )

    ocr_events = [event for event in events if event.get("Style") == "OCR_TRANSLATION"]
    note_events = [event for event in events if event.get("Style") == "EXPLANATION_NOTE"]
    trad_chars = {chr(key): value for key, value in TRADITIONAL_TO_SIMPLIFIED.items() if chr(key) != value}
    visible = "\n".join(event["visible"] for event in events)
    traditional_hits = {char: visible.count(char) for char in trad_chars if char in visible}

    return {
        "total_events": len(events),
        "base_events": len(base),
        "ocr_events": len(ocr_events),
        "note_events": len(note_events),
        "base_overlaps": overlaps,
        "hard_line_events": [
            {"start": event["start_seconds"], "end": event["end_seconds"], "text": event["visible"]}
            for event in hard_line_events
        ],
        "very_long_logical_lines": very_long,
        "traditional_hits": traditional_hits,
    }


def display_stats(segments: list[dict]) -> dict:
    aligned = []
    changed = []
    failed = []
    for index, seg in enumerate(segments, start=1):
        if seg.get("display_units_source") != "qwen":
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        is_music = bool(seg.get("is_music"))
        qwen_parts = semantic_display_parts(seg, start, end, is_music)
        if not qwen_parts:
            failed.append({"index": index, "start": start, "end": end, "text": normalize_plain_text(seg.get("text", ""))})
            continue
        aligned.append({"index": index, "start": start, "end": end, "parts": len(qwen_parts)})
        fallback_seg = copy.deepcopy(seg)
        fallback_seg.pop("display_units", None)
        fallback_seg.pop("display_units_source", None)
        fallback_parts = display_parts(
            fallback_seg,
            start,
            end,
            str(seg.get("zh", "")),
            str(seg.get("en_wrap", seg.get("text", ""))),
        )
        q_sig = [(round(part["start"], 2), round(part["end"], 2), normalize_plain_text(part["en"])) for part in qwen_parts]
        f_sig = [(round(part["start"], 2), round(part["end"], 2), normalize_plain_text(part["en"])) for part in fallback_parts]
        if q_sig != f_sig:
            changed.append(
                {
                    "index": index,
                    "start": start,
                    "end": end,
                    "is_music": is_music,
                    "qwen_parts": [
                        {
                            "start": part["start"],
                            "end": part["end"],
                            "zh": normalize_plain_text(part["zh"]),
                            "en": normalize_plain_text(part["en"]),
                        }
                        for part in qwen_parts
                    ],
                    "fallback_parts": len(fallback_parts),
                }
            )
    return {
        "stored": len(aligned) + len(failed),
        "aligned": len(aligned),
        "failed": failed,
        "changed": changed,
    }


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(base_dir: Path, work_dir: Path, videos: list[Path]) -> Path:
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": str(base_dir),
        "work_dir": str(work_dir),
        "videos": [video.name for video in videos],
        "scripts": {
            name: {
                "sha256": sha256_file(base_dir / name),
                "mtime": (base_dir / name).stat().st_mtime if (base_dir / name).exists() else None,
            }
            for name in SCRIPT_FILES
        },
        "thresholds": {
            "display_max_duration": 5.8,
            "en_display_max_chars": 78,
            "zh_display_max_chars": 34,
            "ass_event_gap": 0.03,
        },
    }
    path = work_dir / "pipeline_manifest.json"
    save_json(path, manifest)
    return path


def ffmpeg_exe() -> str:
    return exe_path("ffmpeg.exe", "FFMPEG_EXE", (FFMPEG_DIR,))


def ffmpeg_filter_path(path: Path) -> str:
    text = path.resolve().as_posix()
    return text.replace(":", r"\:").replace("'", r"\'")


def capture_screenshot(video: Path, ass_file: Path, timestamp: float, out_file: Path) -> bool:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    vf = f"ass='{ffmpeg_filter_path(ass_file)}'"
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, timestamp):.3f}",
        "-i",
        str(video),
        "-vf",
        vf,
        "-frames:v",
        "1",
        str(out_file),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0 and out_file.exists()


def screenshot_candidates(stats: dict, display: dict) -> list[tuple[float, str]]:
    candidates: list[tuple[float, str]] = []
    for item in stats["base_overlaps"]:
        candidates.append((item["start"], "overlap"))
    for item in stats["hard_line_events"]:
        candidates.append((item["start"], "hard_lines"))
    for item in stats["very_long_logical_lines"]:
        candidates.append((item["start"], "long_line"))
    for item in display["changed"]:
        candidates.append((item["start"], "semantic_split"))
    seen = set()
    unique = []
    for timestamp, reason in candidates:
        key = round(timestamp, 1)
        if key in seen:
            continue
        seen.add(key)
        unique.append((timestamp, reason))
    return unique


def write_text_report(path: Path, summary: dict, display: dict, stats: dict, screenshots: list[str]) -> None:
    lines = [
        f"video: {summary['video']}",
        f"translated_segments: {summary['translated_segments']}",
        f"music_segments: {summary['music_segments']}",
        f"chant_segments: {summary.get('chant_segments', 0)}",
        f"qwen_display_units: stored={display['stored']} aligned={display['aligned']} changed={len(display['changed'])} failed={len(display['failed'])}",
        f"ass_events: base={stats['base_events']} total={stats['total_events']} ocr={stats['ocr_events']} notes={stats['note_events']}",
        f"issues: overlaps={len(stats['base_overlaps'])} hard_lines_gt2={len(stats['hard_line_events'])} very_long={len(stats['very_long_logical_lines'])} traditional_hits={sum(stats['traditional_hits'].values())}",
        "",
        "semantic display changes:",
    ]
    for item in display["changed"]:
        lines.append(f"- {fmt_time(item['start'])}-{fmt_time(item['end'])} idx={item['index']} parts={len(item['qwen_parts'])}")
        for part in item["qwen_parts"]:
            lines.append(f"  {fmt_time(part['start'])}-{fmt_time(part['end'])} | {part['zh']} | {part['en']}")
    if display["failed"]:
        lines.append("")
        lines.append("failed qwen display alignment:")
        for item in display["failed"]:
            lines.append(f"- {fmt_time(item['start'])}-{fmt_time(item['end'])} idx={item['index']} | {item['text']}")
    if stats["very_long_logical_lines"]:
        lines.append("")
        lines.append("very long logical lines:")
        for item in stats["very_long_logical_lines"]:
            lines.append(f"- {fmt_time(item['start'])}-{fmt_time(item['end'])} zh={item['zh_len']} en={item['en_len']} | {item['text']}")
    if screenshots:
        lines.append("")
        lines.append("screenshots:")
        lines.extend(f"- {path}" for path in screenshots)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_video(base_dir: Path, work_dir: Path, video: Path, screenshots_enabled: bool, max_screenshots: int) -> dict:
    translated = load_json(work_dir / "translated" / f"{video.stem}.json", {"segments": []})
    music_marked = load_json(work_dir / "music_marked" / f"{video.stem}.json", {"segments": []})
    ocr = load_json(work_dir / "ocr_translated" / f"{video.stem}.json", {"items": []})
    notes = load_json(work_dir / "notes" / f"{video.stem}.json", {"notes": []})
    ass_file = base_dir / f"{video.stem}.ass"
    events = parse_ass_events(ass_file)
    stats = ass_stats(events)
    translated_segments = translated.get("segments", [])
    display = display_stats(translated_segments)
    summary = {
        "video": video.name,
        "stem": video.stem,
        "translated_segments": len(translated_segments),
        "music_segments": sum(1 for seg in translated_segments if seg.get("is_music"))
        or sum(1 for seg in music_marked.get("segments", []) if seg.get("is_music")),
        "chant_segments": sum(
            1
            for seg in translated_segments
            if str(seg.get("kind", "") or "").lower() == "chant" or bool(seg.get("is_chant"))
        ),
        "ocr_items": len(ocr.get("items", [])),
        "notes": len(notes.get("notes", [])),
        "display_units_stored": display["stored"],
        "display_units_aligned": display["aligned"],
        "display_units_changed": len(display["changed"]),
        "display_units_failed": len(display["failed"]),
        "base_events": stats["base_events"],
        "total_events": stats["total_events"],
        "ocr_events": stats["ocr_events"],
        "note_events": stats["note_events"],
        "base_overlaps": len(stats["base_overlaps"]),
        "hard_line_events": len(stats["hard_line_events"]),
        "very_long_logical_lines": len(stats["very_long_logical_lines"]),
        "traditional_hits": sum(stats["traditional_hits"].values()),
    }

    report_dir = work_dir / "reports"
    screen_dir = report_dir / f"{video.stem}_screens"
    screenshots: list[str] = []
    if screenshots_enabled and ass_file.exists():
        for shot_index, (timestamp, reason) in enumerate(screenshot_candidates(stats, display)[:max_screenshots], start=1):
            out_file = screen_dir / f"{shot_index:02d}_{reason}_{int(timestamp * 1000):08d}.jpg"
            if capture_screenshot(video, ass_file, timestamp, out_file):
                screenshots.append(str(out_file))
    summary["screenshots"] = screenshots

    save_json(report_dir / f"{video.stem}.quality.json", {"summary": summary, "ass": stats, "display": display})
    write_text_report(report_dir / f"{video.stem}.quality.txt", summary, display, stats, screenshots)
    return summary


def write_summary(report_dir: Path, summaries: list[dict]) -> None:
    totals = {
        "videos": len(summaries),
        "translated_segments": sum(item["translated_segments"] for item in summaries),
        "music_segments": sum(item["music_segments"] for item in summaries),
        "chant_segments": sum(item.get("chant_segments", 0) for item in summaries),
        "display_units_changed": sum(item["display_units_changed"] for item in summaries),
        "display_units_failed": sum(item["display_units_failed"] for item in summaries),
        "base_overlaps": sum(item["base_overlaps"] for item in summaries),
        "hard_line_events": sum(item["hard_line_events"] for item in summaries),
        "very_long_logical_lines": sum(item["very_long_logical_lines"] for item in summaries),
        "traditional_hits": sum(item["traditional_hits"] for item in summaries),
        "screenshots": sum(len(item.get("screenshots", [])) for item in summaries),
    }
    save_json(report_dir / "summary.json", {"totals": totals, "videos": summaries})
    lines = [
        "# Subtitle Pipeline Quality Summary",
        "",
        f"- videos: {totals['videos']}",
        f"- translated_segments: {totals['translated_segments']}",
        f"- music_segments: {totals['music_segments']}",
        f"- chant_segments: {totals['chant_segments']}",
        f"- qwen_semantic_display_changes: {totals['display_units_changed']}",
        f"- qwen_display_alignment_failed: {totals['display_units_failed']}",
        f"- base_overlaps: {totals['base_overlaps']}",
        f"- hard_line_events_gt2: {totals['hard_line_events']}",
        f"- very_long_logical_lines: {totals['very_long_logical_lines']}",
        f"- traditional_hits: {totals['traditional_hits']}",
        f"- screenshots: {totals['screenshots']}",
        "",
        "| video | segs | music | chant | qwen changed | qwen failed | overlaps | long | trad |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['video']} | {item['translated_segments']} | {item['music_segments']} | {item.get('chant_segments', 0)} | "
            f"{item['display_units_changed']} | {item['display_units_failed']} | "
            f"{item['base_overlaps']} | {item['very_long_logical_lines']} | {item['traditional_hits']} |"
        )
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    report_dir = work_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    videos = selected_videos(base_dir, args.chunk, args.target_stem)
    if not videos:
        print(f"[error] no target videos found in {base_dir}")
        sys.exit(1)

    write_status(base_dir, "V2_STEP10", f"{len(videos)} files", "START quality report")
    manifest_path = write_manifest(base_dir, work_dir, videos)
    summaries = []
    for video in videos:
        print(f"[report] {video.name}")
        summaries.append(report_video(base_dir, work_dir, video, args.screenshots, max(0, args.max_screenshots)))
    write_summary(report_dir, summaries)
    write_status(base_dir, "V2_STEP10", f"{len(videos)} files", "DONE quality report")
    print(f"v2 step10 done; manifest={manifest_path}; summary={report_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
