import argparse
import re
import sys
from pathlib import Path

from common import add_common_args, selected_videos, work_dir_for, write_status
from ass_polish_helpers import (
    add_polish_args,
    build_polish_client,
    curate_static_hint_files,
    curate_glossary_file,
    parse_polish_styles,
    polish_ass_file,
    polish_ass_files,
)
from override_utils import apply_ocr_overrides, load_overrides
from ass_overlay_helpers import ass_escape, break_zh, load_json, make_note_events, ocr_position, seconds_to_ass, split_ass_sections
from ass_filter_helpers import ass_visible_text, has_chinese
from fix_ocr_timeline import repair_ass_file as repair_ocr_timeline


OCR_TIME_SHIFT_SECONDS = 0.0
OCR_STYLE_V2 = (
    "Style: OCR_TRANSLATION,Microsoft YaHei UI,40,&H00E8FFFF,&H000000FF,"
    "&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,4.0,1.2,5,30,30,30,1"
)
NOTE_STYLE_V2 = (
    "Style: EXPLANATION_NOTE,Microsoft YaHei UI,34,&H00FFFFFF,&H000000FF,"
    "&H00333000,&H90000000,-1,0,0,0,100,100,0,0,1,3.0,0.8,8,60,60,34,1"
)

# SDH broadcast captioner / sponsor disclaimer lines that leaked into the
# bilingual dialogue as if they were story subtitles. These are end-credit
# boilerplate ("Captioning sponsored by ...", "Captioned by Media Access Group
# at WGBH access.wgbh.org") and must be stripped from the finished subtitles.
CREDIT_ZH_RE = re.compile(r"^\s*字幕由|赞助商|本字幕|本节目由.*提供")
CREDIT_EN_RE = re.compile(
    r"caption(?:ed|ing)\s+(?:by|sponsored)|media access group|access\.wgbh\.org|"
    r"subtitled by|closed caption|viewers like you|contributions to your",
    re.IGNORECASE,
)
DEDUP_OVERLAY_STYLES = ("OCR_TRANSLATION", "EXPLANATION_NOTE")
DEDUP_KEY_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
EPISODE_CODE_RE = re.compile(r"s\d{2}e\d{2}", re.IGNORECASE)
KNOWN_TARGET_SUFFIXES = {
    ".ass", ".srt", ".vtt", ".ssa", ".sub",
    ".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm", ".ts",
}
ASS_TIME_RE = re.compile(r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<cs>\d{2})")
OCR_MAX_DISPLAY_SECONDS = 6.0
OCR_SUBSUME_TIME_EPSILON = 0.06
SDH_SEPARATOR_RE = re.compile(r"^[\s\-\u2013\u2014\u2015]+$")


def episode_code(value: str | Path) -> str:
    match = EPISODE_CODE_RE.search(str(value))
    return match.group(0).upper() if match else ""


def normalize_selector_text(value: str | Path) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def path_matches_selector(path: Path, selector: str) -> bool:
    selector_ep = episode_code(selector)
    if selector_ep:
        return selector_ep == episode_code(path.name)
    selector_path = Path(selector)
    selector_text = selector_path.stem if selector_path.suffix.lower() in KNOWN_TARGET_SUFFIXES else selector_path.name
    selector_key = normalize_selector_text(selector_text or selector)
    if not selector_key:
        return True
    path_keys = {normalize_selector_text(path.name), normalize_selector_text(path.stem)}
    return any(selector_key in item or item in selector_key for item in path_keys)


def selected_ass_files(base_dir: Path, chunk: str, target_stems: list[str] | None) -> list[Path]:
    ass_files = sorted(base_dir.glob("*.ass"))
    raw_targets = [str(item) for item in (target_stems or []) if item]
    if raw_targets:
        selected = []
        for path in ass_files:
            for target in raw_targets:
                target_path = Path(target)
                candidates = {target, target_path.name, target_path.stem}
                if path.stem in candidates or path.name in candidates:
                    selected.append(path)
                    break
        if selected:
            return selected
    try:
        videos = selected_videos(base_dir, chunk, target_stems)
    except FileNotFoundError:
        videos = []
    if videos:
        stems = {video.stem for video in videos}
        eps = {episode_code(video.name) for video in videos if episode_code(video.name)}
        return [path for path in ass_files if path.stem in stems or episode_code(path.name) in eps]

    selectors = [str(item) for item in (target_stems or []) if item]
    chunk_text = str(chunk or "0").strip()
    if chunk_text.lower() not in {"0", "all", "*"} and not chunk_text.isdigit():
        selectors.append(chunk_text)
    if selectors:
        return [path for path in ass_files if any(path_matches_selector(path, selector) for selector in selectors)]
    return ass_files


def find_sidecar_file(directory: Path, ass_file: Path) -> Path:
    exact = directory / f"{ass_file.stem}.json"
    if exact.exists():
        return exact
    ass_ep = episode_code(ass_file.name)
    if ass_ep:
        matches = sorted(path for path in directory.glob("*.json") if episode_code(path.name) == ass_ep)
        if matches:
            return matches[0]
    return exact


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


def note_reading_units(text: str) -> int:
    visible = ass_visible_text(text)
    hans = len(re.findall(r"[\u4e00-\u9fff]", visible))
    latin_words = len(re.findall(r"[A-Za-z0-9]+", visible))
    return hans + latin_words


def ocr_display_seconds(text: str) -> float:
    units = note_reading_units(text)
    if units <= 2:
        return 2.5
    return min(OCR_MAX_DISPLAY_SECONDS, max(2.5, units / 4.0))


def cap_ocr_durations(path: Path) -> dict[str, int]:
    """Shorten stale OCR overlays in finished ASS files.

    When no OCR sidecar is available, step09 cannot rebuild OCR timing from the
    500ms samples. This keeps existing ASS overlays from lingering far past the
    on-screen text while preserving already-short OCR flashes.
    """
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    changed = 0
    for index, line in enumerate(lines):
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) != 10 or parts[3].strip() != "OCR_TRANSLATION":
            continue
        start = parse_ass_time(parts[1])
        end = parse_ass_time(parts[2])
        if start is None or end is None:
            continue
        target_end = start + ocr_display_seconds(parts[9])
        if end > target_end + 0.01:
            parts[2] = seconds_to_ass(target_end)
            lines[index] = ",".join(parts)
            changed += 1
    if changed:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return {"changed": changed}


def expand_note_durations(path: Path) -> dict[str, int]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    changed = 0
    for index, line in enumerate(lines):
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) != 10 or parts[3].strip() != "EXPLANATION_NOTE":
            continue
        start = parse_ass_time(parts[1])
        end = parse_ass_time(parts[2])
        if start is None or end is None:
            continue
        target_duration = max(5.5, min(9.0, note_reading_units(parts[9]) / 5.0))
        target_end = start + target_duration
        if end + 0.01 < target_end:
            parts[2] = seconds_to_ass(target_end)
            lines[index] = ",".join(parts)
            changed += 1
    if changed:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return {"changed": changed}


def is_credit_disclaimer(visible_text: str) -> bool:
    """True for SDH captioner/sponsor disclaimer lines.

    Carefully excludes real dialogue that merely mentions sponsoring something
    (e.g. "我在赞助一支NASCAR车队" / "we can sponsor the police softball team"):
    those do NOT start with "字幕由" and are not "Captioning/Captioned by ...".
    """
    if not visible_text:
        return False
    # Any line whose visible text matches the SDH disclaimer shape.
    if CREDIT_ZH_RE.search(visible_text) or CREDIT_EN_RE.search(visible_text):
        return True
    return False


def make_dialogue(start: float, end: float, style: str, text: str, layer: int = 5) -> str:
    return f"Dialogue: {layer},{seconds_to_ass(start)},{seconds_to_ass(end)},{style},,0,0,0,,{text}"


def replace_style(lines: list[str], style_name: str, style_line: str) -> tuple[list[str], bool]:
    replaced = False
    new_lines = []
    for line in lines:
        if line.startswith(f"Style: {style_name},"):
            if not replaced:
                new_lines.append(style_line)
                replaced = True
            continue
        new_lines.append(line)
    return new_lines, replaced


def insert_styles_v2(ass_text: str) -> str:
    lines = ass_text.splitlines()
    lines, has_ocr = replace_style(lines, "OCR_TRANSLATION", OCR_STYLE_V2)
    lines, has_note = replace_style(lines, "EXPLANATION_NOTE", NOTE_STYLE_V2)
    new_lines = []
    inserted = False
    for line in lines:
        if not inserted and line.startswith("[Events]"):
            if not has_ocr:
                new_lines.append(OCR_STYLE_V2)
            if not has_note:
                new_lines.append(NOTE_STYLE_V2)
            inserted = True
        new_lines.append(line)
    return "\n".join(new_lines) + "\n"


def make_ocr_events_v2(
    data: dict,
    overrides: dict | None = None,
    play_res_x: int = 1920,
    play_res_y: int = 1080,
) -> list[str]:
    if overrides:
        apply_ocr_overrides(data, overrides)
    width = int(data.get("width") or 1920)
    height = int(data.get("height") or 1080)
    # OCR bbox coordinates are in the video's native resolution, but the ASS
    # subtitle is rendered against PlayResX/PlayResY. Scale positions so the
    # overlay lands next to the on-screen English text instead of drifting
    # toward the bottom-right when the video is larger than PlayRes.
    scale_x = play_res_x / width if width else 1.0
    scale_y = play_res_y / height if height else 1.0
    events = []
    for item in data.get("items", []):
        zh = break_zh(item.get("zh") or "")
        if not zh:
            continue
        raw_start = float(item.get("start", 0))
        raw_end = max(float(item.get("end", raw_start + 1.5)), raw_start + 0.8)
        start = max(0.0, raw_start - OCR_TIME_SHIFT_SECONDS)
        end = max(start + 0.3, raw_end - OCR_TIME_SHIFT_SECONDS)
        x, y = ocr_position(item, width, height)
        x = int(round(x * scale_x))
        y = int(round(y * scale_y))
        # ocr_position returns the bbox center, so an \an5 (center-aligned) overlay
        # sits right on top of the English text. Shift it up by half the bbox height
        # plus a small gap so the Chinese reads above the original instead of
        # covering it. If there's no room at the top, fall back to below.
        bbox = item.get("bbox") or item.get("box") or []
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                by1, by2 = float(bbox[1]), float(bbox[3])
                if by2 < by1:
                    by1, by2 = by2, by1
                half_h = int(round((by2 - by1) / 2 * scale_y))
                shift = half_h + 8
                if y - shift >= 8:
                    y -= shift
                else:
                    y += shift
            except (TypeError, ValueError):
                pass
        text = rf"{{\an5\pos({x},{y})}}{ass_escape(zh)}"
        events.append(make_dialogue(start, end, "OCR_TRANSLATION", text, layer=6))
    return events


def _ass_play_res(ass_text: str) -> tuple[int, int]:
    """Extract PlayResX/PlayResY from the [Script Info] header (default 1920x1080)."""
    play_x, play_y = 1920, 1080
    for line in ass_text.splitlines():
        low = line.strip().lower()
        if low.startswith("playresx:"):
            try:
                play_x = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif low.startswith("playresy:"):
            try:
                play_y = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return play_x, play_y


def overlay_ass(ass_file: Path, ocr_file: Path, notes_file: Path) -> int:
    if not ocr_file.exists() and not notes_file.exists():
        return 0
    original_text = ass_file.read_text(encoding="utf-8-sig")
    ass_text = original_text
    ass_text = insert_styles_v2(ass_text)
    header, base_events = split_ass_sections(ass_text)
    # Make the overlay idempotent: drop any previously-merged OCR/NOTE overlay
    # events so re-running this step never stacks duplicates on top of itself.
    overlay_styles = ("OCR_TRANSLATION", "EXPLANATION_NOTE")
    base_events = [
        line for line in base_events
        if not (line.startswith("Dialogue:") and any(f",{style}," in line for style in overlay_styles))
    ]
    overlay_events = []
    overrides = load_overrides(ass_file.parent, ass_file.stem)
    play_x, play_y = _ass_play_res(ass_text)
    overlay_events.extend(make_ocr_events_v2(load_json(ocr_file, {}), overrides, play_x, play_y))
    overlay_events.extend(make_note_events(load_json(notes_file, {})))
    all_events = base_events + overlay_events
    all_events.sort(key=lambda line: line.split(",", 3)[1] if line.startswith("Dialogue:") else "")
    output = "\n".join(header + all_events) + "\n"
    if output != original_text:
        ass_file.write_text(output, encoding="utf-8-sig")
    return len(overlay_events)


def is_separator_only_bilingual(line: str) -> bool:
    """Return true for empty SDH speaker separators such as "- -" / "——"."""
    parts = line.split(",", 9) if line.startswith("Dialogue:") else []
    if len(parts) != 10 or parts[3] != "BILINGUAL":
        return False
    fragments = parts[9].split(r"\N")
    visible_fragments = [ass_visible_text(fragment).strip() for fragment in fragments]
    return bool(visible_fragments) and all(
        not fragment or SDH_SEPARATOR_RE.fullmatch(fragment)
        for fragment in visible_fragments
    )


def filter_ass_file(path: Path) -> tuple[int, int, int]:
    """Filter one ASS file.

    Returns (ocr_total, removed_non_chinese, removed_credits).
    - OCR_TRANSLATION events without Chinese are dropped.
    - BILINGUAL dialogue lines that are SDH captioner/sponsor disclaimer
      boilerplate ("字幕由…赞助", "Captioning sponsored by …",
      "Captioned by Media Access Group at WGBH …") are dropped. Real dialogue
      that merely mentions sponsoring something is preserved.
    """
    text = path.read_text(encoding="utf-8-sig")
    kept_lines: list[str] = []
    removed_non_chinese = 0
    removed_credits = 0
    total_ocr = 0
    for line in text.splitlines():
        if not line.startswith("Dialogue:"):
            kept_lines.append(line)
            continue
        if ",OCR_TRANSLATION," in line:
            total_ocr += 1
            if not has_chinese(ass_visible_text(line)):
                removed_non_chinese += 1
                continue
        elif ",BILINGUAL," in line:
            if is_separator_only_bilingual(line):
                removed_non_chinese += 1
                continue
            if is_credit_disclaimer(ass_visible_text(line)):
                removed_credits += 1
                continue
        kept_lines.append(line)
    if removed_non_chinese or removed_credits:
        path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8-sig")
    return total_ocr, removed_non_chinese, removed_credits


def overlay_dedup_key(text: str) -> str:
    visible = ass_visible_text(text)
    key = DEDUP_KEY_RE.sub("", visible).lower()
    return key


def _overlay_text_subsumed(key: str, seen_keys: set[str], style: str) -> bool:
    if style != "OCR_TRANSLATION" or len(key) < 2:
        return False
    for other in seen_keys:
        if len(other) <= len(key):
            continue
        if key in other:
            return True
    return False


def _overlay_times_match(start_a: str, end_a: str, start_b: str, end_b: str) -> bool:
    start1 = parse_ass_time(start_a)
    end1 = parse_ass_time(end_a)
    start2 = parse_ass_time(start_b)
    end2 = parse_ass_time(end_b)
    if None in {start1, end1, start2, end2}:
        return False
    return abs(start1 - start2) <= OCR_SUBSUME_TIME_EPSILON and abs(end1 - end2) <= OCR_SUBSUME_TIME_EPSILON


def dedupe_overlay_events(path: Path) -> dict[str, int]:
    """Drop repeated OCR/note overlay text within one ASS file, keeping first use.

    OCR and cultural notes are auxiliary context; if the same overlay text appears
    repeatedly in an episode, the first instance is enough and later repeats add
    clutter. This runs after overlay generation and again after long-OCR repair,
    because repair can split one long OCR into repeated short events.
    """
    text = path.read_text(encoding="utf-8-sig")
    seen: dict[str, set[str]] = {style: set() for style in DEDUP_OVERLAY_STYLES}
    removed = {style: 0 for style in DEDUP_OVERLAY_STYLES}
    kept_lines: list[str] = []
    changed = False
    kept_overlay_parts: list[list[str]] = []
    for line in text.splitlines():
        parts = line.split(",", 9) if line.startswith("Dialogue:") else []
        if len(parts) == 10 and parts[3] in seen:
            key = overlay_dedup_key(parts[9])
            min_key_len = 1 if parts[3] == "OCR_TRANSLATION" else 2
            if len(key) >= min_key_len:
                if key in seen[parts[3]]:
                    removed[parts[3]] += 1
                    changed = True
                    continue
                for prev in kept_overlay_parts:
                    if prev[3] != parts[3]:
                        continue
                    if not _overlay_times_match(parts[1], parts[2], prev[1], prev[2]):
                        continue
                    if _overlay_text_subsumed(key, {overlay_dedup_key(prev[9])}, parts[3]):
                        removed[parts[3]] += 1
                        changed = True
                        break
                else:
                    seen[parts[3]].add(key)
                    kept_overlay_parts.append(parts)
                    kept_lines.append(line)
                    continue
                continue
        kept_lines.append(line)
    if changed:
        path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8-sig")
    return {
        "ocr_duplicates": removed["OCR_TRANSLATION"],
        "note_duplicates": removed["EXPLANATION_NOTE"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V2 STEP9: merge OCR/notes overlays, filter unusable OCR subtitles, and optionally polish ASS text."
    )
    add_common_args(parser)
    add_polish_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    raw_ocr_dir = work_dir / "ocr_raw"
    ocr_dir = work_dir / "ocr_translated"
    notes_dir = work_dir / "notes"
    polish_styles = parse_polish_styles(args.polish_styles)
    polish_cache_dir = Path(args.polish_cache_dir) if args.polish_cache_dir else work_dir / "polish_cache"
    if not polish_cache_dir.is_absolute():
        polish_cache_dir = work_dir / polish_cache_dir
    if args.polish_backend:
        try:
            build_polish_client(
                args.polish_provider,
                args.polish_base_url,
                args.polish_api_key,
                args.polish_timeout,
                cache_dir=polish_cache_dir,
                cwd=base_dir,
                codex_command=args.polish_codex_command,
                codex_reasoning_effort=args.polish_codex_reasoning_effort,
            )
        except RuntimeError as exc:
            print(f"[error] {exc}")
            sys.exit(1)
        write_status(base_dir, "V2_STEP9", "subtitle_glossary.json", "review new glossary entries")
        glossary_stats = curate_glossary_file(
            base_dir,
            work_dir,
            provider=args.polish_provider,
            model=args.polish_model,
            base_url=args.polish_base_url,
            api_key=args.polish_api_key,
            batch_size=max(args.polish_batch_size, 0),
            timeout=args.polish_timeout,
            temperature=args.polish_temperature,
            cache_dir=polish_cache_dir,
            codex_command=args.polish_codex_command,
            codex_reasoning_effort=args.polish_codex_reasoning_effort,
        )
        if glossary_stats["targets"]:
            print(
                "[glossary] "
                f"targets={glossary_stats['targets']}, reviewed={glossary_stats['reviewed']}, "
                f"changed={glossary_stats['changed']}, removed={glossary_stats['removed']}, "
                f"failed={glossary_stats['failed']}"
            )
    ass_files = selected_ass_files(base_dir, args.chunk, args.target_stem)
    if not ass_files:
        print(f"[error] no ASS files found in {base_dir}")
        sys.exit(1)

    # Every ASS file is a candidate for overlay. OCR overlays only apply to the
    # audio path (embedded subtitles have no OCR text), but cultural notes now
    # apply to both paths, so we no longer skip embedded sources wholesale.
    # overlay_ass() decides per file based on whether ocr_file/notes_file exist.
    if not ass_files:
        print(f"[error] no ASS files found in {base_dir}")
        sys.exit(1)

    total_overlay = 0
    total_ocr = 0
    total_removed = 0
    total_credits = 0
    total_polish_targets = 0
    total_polish_requested = 0
    total_polish_cache = 0
    total_polish_already = 0
    total_polish_unchanged = 0
    total_polish_deleted = 0
    total_polish_changed = 0
    total_polish_failed = 0
    total_ocr_repair_changed = 0
    total_ocr_repair_split = 0
    total_ocr_repair_added = 0
    total_ocr_duration_changed = 0
    total_dedup_ocr = 0
    total_dedup_notes = 0
    total_note_duration_changed = 0
    file_stats = []
    for ass_file in ass_files:
        raw_ocr_file = find_sidecar_file(raw_ocr_dir, ass_file)
        ocr_file = find_sidecar_file(ocr_dir, ass_file)
        notes_file = find_sidecar_file(notes_dir, ass_file)
        write_status(base_dir, "V2_STEP9", ass_file.name, "merge overlays")
        overlay_count = overlay_ass(ass_file, ocr_file, notes_file)
        ocr_count, removed, credits = filter_ass_file(ass_file)
        dedup_stats = dedupe_overlay_events(ass_file)
        ocr_count_after, removed_after, credits_after = filter_ass_file(ass_file)
        ocr_count = ocr_count_after
        removed += removed_after
        credits += credits_after
        polish_stats = {
            "targets": 0,
            "requested": 0,
            "cache_hits": 0,
            "already_polished": 0,
            "unchanged": 0,
            "deleted": 0,
            "changed": 0,
            "failed": 0,
        }
        file_stats.append(
            {
                "ass_file": ass_file,
                "raw_ocr_file": raw_ocr_file,
                "ocr_file": ocr_file,
                "overlay_count": overlay_count,
                "ocr_count": ocr_count,
                "removed": removed,
                "credits": credits,
                "dedup_stats": dedup_stats,
                "polish_stats": polish_stats,
            }
        )

    if args.polish_backend:
        file_batch_size = max(int(getattr(args, "polish_file_batch_size", 0) or 0), 1)
        if file_batch_size > len(file_stats):
            file_batch_size = len(file_stats) or 1
        for batch_offset in range(0, len(file_stats), file_batch_size):
            batch_records = file_stats[batch_offset : batch_offset + file_batch_size]
            for record in batch_records:
                write_status(base_dir, "V2_STEP9", record["ass_file"].name, "polish ASS translations")
            batch_results = polish_ass_files(
                [record["ass_file"] for record in batch_records],
                base_dir,
                work_dir,
                provider=args.polish_provider,
                model=args.polish_model,
                base_url=args.polish_base_url,
                api_key=args.polish_api_key,
                batch_size=args.polish_batch_size,
                timeout=args.polish_timeout,
                temperature=args.polish_temperature,
                style_names=polish_styles,
                cache_dir=polish_cache_dir,
                force=args.polish_force,
                codex_command=args.polish_codex_command,
                codex_reasoning_effort=args.polish_codex_reasoning_effort,
            )
            for record in batch_records:
                ass_file = record["ass_file"]
                record["polish_stats"] = batch_results.get(ass_file, record["polish_stats"])
                ocr_count_after, removed_after, credits_after = filter_ass_file(ass_file)
                record["ocr_count"] = ocr_count_after
                record["removed"] += removed_after
                record["credits"] += credits_after

        polish_failures = sum(record["polish_stats"].get("failed", 0) for record in file_stats)
        if polish_failures:
            print(f"[static-hints] skipped because polish_failed={polish_failures}")
        else:
            static_hint_stats = curate_static_hint_files(
                [polish_cache_dir / f"{record['ass_file'].stem}.json" for record in file_stats],
                provider=args.polish_provider,
                model=args.polish_model,
                base_url=args.polish_base_url,
                api_key=args.polish_api_key,
                timeout=args.polish_timeout,
                temperature=args.polish_temperature,
                cache_dir=polish_cache_dir,
                cwd=base_dir,
                codex_command=args.polish_codex_command,
                codex_reasoning_effort=args.polish_codex_reasoning_effort,
            )
            if static_hint_stats["samples"]:
                print(
                    "[static-hints] "
                    f"samples={static_hint_stats['samples']}, "
                    f"mistranslation_added={static_hint_stats['mistranslation_added']}, "
                    f"phrase_added={static_hint_stats['phrase_added']}, "
                    f"ocr_low_value_added={static_hint_stats['ocr_low_value_added']}, "
                    f"failed={static_hint_stats['failed']}, "
                    f"skipped={static_hint_stats.get('skipped', 0)}"
                )

    for record in file_stats:
        ass_file = record["ass_file"]
        raw_ocr_file = record["raw_ocr_file"]
        ocr_file = record["ocr_file"]
        overlay_count = record["overlay_count"]
        ocr_count = record["ocr_count"]
        removed = record["removed"]
        credits = record["credits"]
        dedup_stats = record["dedup_stats"]
        polish_stats = record["polish_stats"]
        repair_stats = {"changed": 0, "split": 0, "added": 0}
        if raw_ocr_file.exists() and ocr_file.exists():
            repair_stats = repair_ocr_timeline(ass_file, raw_ocr_file, ocr_file)
            ocr_count_after, removed_after, credits_after = filter_ass_file(ass_file)
            ocr_count = ocr_count_after
            removed += removed_after
            credits += credits_after
            repair_dedup_stats = dedupe_overlay_events(ass_file)
            dedup_stats["ocr_duplicates"] += repair_dedup_stats["ocr_duplicates"]
            dedup_stats["note_duplicates"] += repair_dedup_stats["note_duplicates"]
            ocr_count_after, removed_after, credits_after = filter_ass_file(ass_file)
            ocr_count = ocr_count_after
            removed += removed_after
            credits += credits_after
        ocr_duration_stats = cap_ocr_durations(ass_file)
        final_dedup_stats = dedupe_overlay_events(ass_file)
        dedup_stats["ocr_duplicates"] += final_dedup_stats["ocr_duplicates"]
        dedup_stats["note_duplicates"] += final_dedup_stats["note_duplicates"]
        ocr_count_after, removed_after, credits_after = filter_ass_file(ass_file)
        ocr_count = ocr_count_after
        removed += removed_after
        credits += credits_after
        note_duration_stats = expand_note_durations(ass_file)
        total_overlay += overlay_count
        total_ocr += ocr_count
        total_removed += removed
        total_credits += credits
        total_polish_targets += polish_stats["targets"]
        total_polish_requested += polish_stats["requested"]
        total_polish_cache += polish_stats["cache_hits"]
        total_polish_already += polish_stats["already_polished"]
        total_polish_unchanged += polish_stats.get("unchanged", 0)
        total_polish_deleted += polish_stats.get("deleted", 0)
        total_polish_changed += polish_stats["changed"]
        total_polish_failed += polish_stats["failed"]
        total_ocr_repair_changed += repair_stats["changed"]
        total_ocr_repair_split += repair_stats["split"]
        total_ocr_repair_added += repair_stats["added"]
        total_ocr_duration_changed += ocr_duration_stats["changed"]
        total_dedup_ocr += dedup_stats["ocr_duplicates"]
        total_dedup_notes += dedup_stats["note_duplicates"]
        total_note_duration_changed += note_duration_stats["changed"]
        polish_suffix = ""
        if args.polish_backend:
            polish_suffix = (
                f", polish_targets={polish_stats['targets']}, "
                f"polish_requested={polish_stats['requested']}, "
                f"polish_cache={polish_stats['cache_hits']}, "
                f"polish_already={polish_stats['already_polished']}, "
                f"polish_unchanged={polish_stats.get('unchanged', 0)}, "
                f"polish_deleted={polish_stats.get('deleted', 0)}, "
                f"polish_changed={polish_stats['changed']}, "
                f"polish_failed={polish_stats['failed']}"
            )
        repair_suffix = ""
        if repair_stats["changed"]:
            repair_suffix = (
                f", ocr_repair_changed={repair_stats['changed']}, "
                f"ocr_repair_split={repair_stats['split']}, "
                f"ocr_repair_added={repair_stats['added']}"
            )
        dedup_suffix = ""
        if dedup_stats["ocr_duplicates"] or dedup_stats["note_duplicates"]:
            dedup_suffix = (
                f", dedup_ocr={dedup_stats['ocr_duplicates']}, "
                f"dedup_notes={dedup_stats['note_duplicates']}"
            )
        ocr_duration_suffix = ""
        if ocr_duration_stats["changed"]:
            ocr_duration_suffix = f", ocr_duration_changed={ocr_duration_stats['changed']}"
        note_duration_suffix = ""
        if note_duration_stats["changed"]:
            note_duration_suffix = f", note_duration_changed={note_duration_stats['changed']}"
        print(
            f"[done] {ass_file.name}: overlay={overlay_count}, "
            f"OCR_TRANSLATION={ocr_count}, removed_non_chinese={removed}, removed_credits={credits}"
            f"{polish_suffix}{repair_suffix}{dedup_suffix}{ocr_duration_suffix}{note_duration_suffix}"
        )

    polish_done = ""
    if args.polish_backend:
        polish_done = (
            f", polish_targets={total_polish_targets}, polish_requested={total_polish_requested}, "
            f"polish_cache={total_polish_cache}, polish_already={total_polish_already}, "
            f"polish_unchanged={total_polish_unchanged}, "
            f"polish_deleted={total_polish_deleted}, "
            f"polish_changed={total_polish_changed}, polish_failed={total_polish_failed}"
        )
    repair_done = ""
    if total_ocr_repair_changed:
        repair_done = (
            f", ocr_repair_changed={total_ocr_repair_changed}, "
            f"ocr_repair_split={total_ocr_repair_split}, "
            f"ocr_repair_added={total_ocr_repair_added}"
        )
    ocr_duration_done = ""
    if total_ocr_duration_changed:
        ocr_duration_done = f", ocr_duration_changed={total_ocr_duration_changed}"
    dedup_done = ""
    if total_dedup_ocr or total_dedup_notes:
        dedup_done = f", dedup_ocr={total_dedup_ocr}, dedup_notes={total_dedup_notes}"
    note_duration_done = ""
    if total_note_duration_changed:
        note_duration_done = f", note_duration_changed={total_note_duration_changed}"
    write_status(
        base_dir,
        "V2_STEP9",
        f"{len(ass_files)} files",
        f"DONE overlay={total_overlay}, ocr={total_ocr}, removed={total_removed}, credits={total_credits}{polish_done}{repair_done}{ocr_duration_done}{dedup_done}{note_duration_done}",
    )
    print(
        f"v2 step9 done; files={len(ass_files)}, "
        f"overlay={total_overlay}, OCR_TRANSLATION={total_ocr}, "
        f"removed_non_chinese={total_removed}, removed_credits={total_credits}"
        f"{polish_done}{repair_done}{ocr_duration_done}{dedup_done}{note_duration_done}"
    )


if __name__ == "__main__":
    main()
