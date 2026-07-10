import argparse
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

from ass_polish_helpers import (
    BILINGUAL_STYLES,
    CHANT_SYMBOL,
    MUSIC_SYMBOL,
    OVERLAY_STYLES,
    split_bilingual_text,
    split_dialogue,
    strip_presentation_tags,
    visible_ass_text,
    with_english_size_tag,
)


STYLE_PRIORITY = {
    "BILINGUAL": 1,
    "BILINGUAL_MUSIC": 2,
    "BILINGUAL_CHANT": 3,
    "EXPLANATION_NOTE": 4,
    "OCR_TRANSLATION": 5,
}
BILINGUAL_FONT_SIZE = "56"
BILINGUAL_STYLE_NAMES = {"BILINGUAL", "BILINGUAL_MUSIC", "BILINGUAL_CHANT"}
MOJIBAKE_MUSIC_MARK_RE = re.compile(
    r"(?<![0-9A-Za-z])(?:[jJ][“”\"']|[jJ]n(?=\s))|[jJ][“”\"'](?=\s|$)"
)


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", visible_ass_text(strip_presentation_tags(text))).strip()


def norm_style_text(text: str) -> str:
    value = norm_text(text)
    value = value.strip(MUSIC_SYMBOL).strip(CHANT_SYMBOL).strip()
    value = re.sub(r"^[jJ][“”\"']\s*", "", value).strip()
    value = re.sub(r"\s*[jJ][“”\"']$", "", value).strip()
    return value


def has_music_marks(text: str) -> bool:
    value = norm_text(text)
    return any(symbol in value for symbol in (MUSIC_SYMBOL, "\u266b", "\u266c")) or bool(
        MOJIBAKE_MUSIC_MARK_RE.search(value)
    )


def has_chant_marks(text: str) -> bool:
    value = norm_text(text)
    return CHANT_SYMBOL in value


def prefer_style(current: str | None, candidate: str) -> str:
    if not current:
        return candidate
    return candidate if STYLE_PRIORITY.get(candidate, 0) > STYLE_PRIORITY.get(current, 0) else current


def add_mapping(mapping: dict, key, style: str) -> None:
    if not key or style not in STYLE_PRIORITY:
        return
    mapping[key] = prefer_style(mapping.get(key), style)


def load_cache_styles(cache_file: Path) -> dict:
    pair_map: dict[tuple[str, str], str] = {}
    zh_map: dict[str, str] = {}
    en_special: dict[str, set[str]] = defaultdict(set)
    if not cache_file.exists():
        return {"pair": pair_map, "zh": zh_map, "en_special": en_special}
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {"pair": pair_map, "zh": zh_map, "en_special": en_special}

    entries = data.get("entries", {}) if isinstance(data, dict) else {}
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        style = str(entry.get("style") or "")
        if style not in STYLE_PRIORITY:
            continue
        en = norm_text(entry.get("en", ""))
        zh_values = {
            norm_text(entry.get("source_zh", "")),
            norm_text(entry.get("polished_zh", "")),
            norm_style_text(entry.get("source_zh", "")),
            norm_style_text(entry.get("polished_zh", "")),
        }
        for zh in zh_values:
            if not zh:
                continue
            if en:
                add_mapping(pair_map, (en, zh), style)
            else:
                add_mapping(zh_map, zh, style)
        if en and style != "BILINGUAL":
            en_special[en].add(style)
    return {"pair": pair_map, "zh": zh_map, "en_special": en_special}


def style_from_cache(cache_styles: dict, zh: str, en: str, allow_overlay: bool) -> str | None:
    en_key = norm_text(en)
    zh_keys = {norm_text(zh), norm_style_text(zh)}
    if en_key:
        for zh_key in zh_keys:
            style = cache_styles["pair"].get((en_key, zh_key))
            if style and (allow_overlay or style in BILINGUAL_STYLES):
                return style
        special = cache_styles["en_special"].get(en_key, set())
        if len(special) == 1:
            style = next(iter(special))
            if allow_overlay or style in BILINGUAL_STYLES:
                return style
    if allow_overlay:
        for zh_key in zh_keys:
            style = cache_styles["zh"].get(zh_key)
            if style:
                return style
    return None


def inferred_bilingual_style(zh: str, en: str, fallback: str) -> str:
    if has_chant_marks(zh) or has_chant_marks(en):
        return "BILINGUAL_CHANT"
    if has_music_marks(zh) or has_music_marks(en):
        return "BILINGUAL_MUSIC"
    return fallback if fallback in BILINGUAL_STYLES else "BILINGUAL"


def clean_bilingual_field(zh_fragment: str, en_fragment: str) -> str:
    zh = strip_presentation_tags(zh_fragment).strip()
    en = with_english_size_tag(en_fragment)
    return zh + r"\N" + en if en else zh


def repair_style_header(line: str) -> str:
    if not line.startswith("Style: "):
        return line
    parts = line.split(",")
    if len(parts) < 3:
        return line
    style_name = parts[0].replace("Style:", "", 1).strip()
    if style_name in BILINGUAL_STYLE_NAMES:
        parts[2] = BILINGUAL_FONT_SIZE
        return ",".join(parts)
    return line


def repair_ass_file(ass_file: Path, cache_file: Path) -> dict:
    cache_styles = load_cache_styles(cache_file)
    lines = ass_file.read_text(encoding="utf-8-sig").splitlines()
    changed = False
    stats = {
        "dialogues": 0,
        "style_changed": 0,
        "presentation_tags_removed": 0,
        "style_header_changed": 0,
        "cache_style": 0,
        "inferred_music": 0,
        "inferred_chant": 0,
    }

    for index, line in enumerate(lines):
        repaired_header = repair_style_header(line)
        if repaired_header != line:
            lines[index] = repaired_header
            changed = True
            stats["style_header_changed"] += 1
            continue
        parts = split_dialogue(line)
        if not parts:
            continue
        stats["dialogues"] += 1
        original_style = parts[3].strip()
        original_text = parts[9]
        split = split_bilingual_text(original_text) if original_style in BILINGUAL_STYLES else None

        if split:
            zh_fragment, en_fragment = split
            cache_style = style_from_cache(cache_styles, zh_fragment, en_fragment, allow_overlay=False)
            inferred_style = inferred_bilingual_style(zh_fragment, en_fragment, original_style)
            new_style = prefer_style(inferred_style, cache_style) if cache_style else inferred_style
            new_text = clean_bilingual_field(zh_fragment, en_fragment)
            if cache_style and cache_style != original_style:
                stats["cache_style"] += 1
            if not cache_style and new_style == "BILINGUAL_MUSIC" and original_style != new_style:
                stats["inferred_music"] += 1
            if not cache_style and new_style == "BILINGUAL_CHANT" and original_style != new_style:
                stats["inferred_chant"] += 1
        else:
            cache_style = style_from_cache(cache_styles, original_text, "", allow_overlay=True)
            new_style = cache_style or original_style
            new_text = strip_presentation_tags(original_text)
            if cache_style and cache_style != original_style:
                stats["cache_style"] += 1

        if original_text != new_text:
            stats["presentation_tags_removed"] += 1
        if original_style != new_style:
            stats["style_changed"] += 1
        parts[3] = new_style
        parts[9] = new_text
        new_line = ",".join(parts)
        if new_line != line:
            lines[index] = new_line
            changed = True

    if changed:
        ass_file.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair ASS event styles from polish cache and remove presentation override tags."
    )
    parser.add_argument("base_dir", nargs="?", default=".", help="Directory containing finished ASS files.")
    parser.add_argument(
        "--cache-dir",
        default="temp/polish_cache",
        help="Directory containing per-episode polish cache JSON files.",
    )
    parser.add_argument(
        "--backup-dir",
        default="",
        help="Optional backup directory for ASS files before modifying them.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = base_dir / cache_dir
    backup_dir = Path(args.backup_dir).resolve() if args.backup_dir else None
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)

    total = defaultdict(int)
    ass_files = sorted(base_dir.glob("*.ass"))
    for ass_file in ass_files:
        if backup_dir:
            shutil.copy2(ass_file, backup_dir / ass_file.name)
        stats = repair_ass_file(ass_file, cache_dir / f"{ass_file.stem}.json")
        for key, value in stats.items():
            total[key] += value
        print(
            f"[repair] {ass_file.name}: "
            f"style_changed={stats['style_changed']}, "
            f"presentation_tags_removed={stats['presentation_tags_removed']}, "
            f"style_header_changed={stats['style_header_changed']}, "
            f"cache_style={stats['cache_style']}, "
            f"inferred_music={stats['inferred_music']}, "
            f"inferred_chant={stats['inferred_chant']}"
        )

    print(
        f"repair done at {time.strftime('%Y-%m-%d %H:%M:%S')}; "
        f"files={len(ass_files)}, "
        f"dialogues={total['dialogues']}, "
        f"style_changed={total['style_changed']}, "
        f"presentation_tags_removed={total['presentation_tags_removed']}, "
        f"style_header_changed={total['style_header_changed']}, "
        f"cache_style={total['cache_style']}, "
        f"inferred_music={total['inferred_music']}, "
        f"inferred_chant={total['inferred_chant']}"
    )


if __name__ == "__main__":
    main()
