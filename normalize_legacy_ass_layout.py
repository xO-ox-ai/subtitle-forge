import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

from ass_filter_helpers import has_chinese
from ass_polish_helpers import (
    CHANT_SYMBOL,
    LEGACY_PAIRED_SOURCE_STYLES,
    LEGACY_PAIRED_ZH_STYLES,
    MUSIC_SYMBOL,
    split_dialogue,
    strip_presentation_tags,
    visible_ass_text,
    with_english_size_tag,
)
from step08_ass_render import ASS_HEADER


LATIN_RE = re.compile(r"[A-Za-z]")
MUSIC_SYMBOLS = (MUSIC_SYMBOL, "♫", "♬")


def _bilingual_style(zh: str, en: str) -> str:
    visible = f"{visible_ass_text(zh)} {visible_ass_text(en)}"
    if CHANT_SYMBOL in visible:
        return "BILINGUAL_CHANT"
    if any(symbol in visible for symbol in MUSIC_SYMBOLS):
        return "BILINGUAL_MUSIC"
    return "BILINGUAL"


def _clean_text(text: str) -> str:
    return strip_presentation_tags(text).strip()


def _combined_text(zh: str, en: str) -> str:
    zh_clean = _clean_text(zh)
    en_clean = with_english_size_tag(en)
    if zh_clean and en_clean:
        return zh_clean + r"\N" + en_clean
    return zh_clean or en_clean


def _canonical_header() -> list[str]:
    return ASS_HEADER.rstrip().splitlines()


def normalize_legacy_layout(lines: list[str]) -> tuple[list[str], dict[str, int]]:
    """Convert exact-time legacy language pairs to current bilingual events.

    Pair selection intentionally matches ``extract_legacy_paired_targets``:
    exact start/end groups only, with the last events preferred when malformed
    groups contain extra fragments. Unpaired events are retained.
    """
    groups: dict[tuple[str, str], dict[str, list[tuple[int, list[str]]]]] = defaultdict(
        lambda: {"sources": [], "targets": []}
    )
    dialogue_by_index: dict[int, list[str]] = {}

    for index, line in enumerate(lines):
        parts = split_dialogue(line)
        if not parts:
            continue
        dialogue_by_index[index] = parts
        style = parts[3].strip()
        if style not in (LEGACY_PAIRED_SOURCE_STYLES | LEGACY_PAIRED_ZH_STYLES):
            continue
        visible = visible_ass_text(parts[9])
        if not visible:
            continue
        group = groups[(parts[1], parts[2])]
        if style in LEGACY_PAIRED_ZH_STYLES and has_chinese(visible):
            group["targets"].append((index, parts))
        elif style in LEGACY_PAIRED_SOURCE_STYLES and LATIN_RE.search(visible):
            group["sources"].append((index, parts))

    pair_by_index: dict[int, tuple[int, int, list[str], list[str]]] = {}
    pair_count = 0
    for group in groups.values():
        count = min(len(group["sources"]), len(group["targets"]))
        if count <= 0:
            continue
        for (source_index, source_parts), (zh_index, zh_parts) in zip(
            group["sources"][-count:], group["targets"][-count:]
        ):
            pair = (min(source_index, zh_index), max(source_index, zh_index), source_parts, zh_parts)
            pair_by_index[source_index] = pair
            pair_by_index[zh_index] = pair
            pair_count += 1

    events_index = next((i for i, line in enumerate(lines) if line.strip() == "[Events]"), None)
    body_start = None
    if events_index is not None:
        body_start = next(
            (i + 1 for i in range(events_index + 1, len(lines)) if lines[i].startswith("Format:")),
            events_index + 1,
        )
    body = lines[body_start:] if body_start is not None else lines

    # Body indices must still point to the original line list while converting.
    body_index_offset = body_start or 0
    output = _canonical_header()
    legacy_singles = 0
    for relative_index, line in enumerate(body):
        index = body_index_offset + relative_index
        parts = dialogue_by_index.get(index)
        if not parts:
            output.append(line)
            continue

        pair = pair_by_index.get(index)
        if pair:
            emit_index, _other_index, source_parts, zh_parts = pair
            if index != emit_index:
                continue
            merged = zh_parts.copy()
            merged[3] = _bilingual_style(zh_parts[9], source_parts[9])
            merged[9] = _combined_text(zh_parts[9], source_parts[9])
            output.append(",".join(merged))
            continue

        style = parts[3].strip()
        if style in (LEGACY_PAIRED_SOURCE_STYLES | LEGACY_PAIRED_ZH_STYLES):
            visible = visible_ass_text(parts[9])
            normalized = parts.copy()
            normalized[3] = _bilingual_style(parts[9] if has_chinese(visible) else "", parts[9])
            if has_chinese(visible):
                normalized[9] = _clean_text(parts[9])
            elif LATIN_RE.search(visible):
                normalized[9] = with_english_size_tag(parts[9])
            else:
                normalized[9] = _clean_text(parts[9])
            output.append(",".join(normalized))
            legacy_singles += 1
        else:
            output.append(line)

    stats = {
        "dialogues_before": len(dialogue_by_index),
        "pairs_merged": pair_count,
        "legacy_singles": legacy_singles,
        "dialogues_after": len(dialogue_by_index) - pair_count,
    }
    return output, stats


def normalize_ass_file(path: Path, backup_dir: Path | None = None) -> dict[str, int]:
    original = path.read_text(encoding="utf-8-sig").splitlines()
    normalized, stats = normalize_legacy_layout(original)
    rendered = "\n".join(normalized) + "\n"
    current = "\n".join(original) + "\n"
    stats["changed"] = int(rendered != current)
    if rendered != current:
        if backup_dir is not None:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_dir / path.name)
        path.write_text(rendered, encoding="utf-8-sig")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy split-language ASS events to current layout.")
    parser.add_argument("base_dir", nargs="?", default=".")
    parser.add_argument("--glob", default="*.ass")
    parser.add_argument("--backup-dir", default="")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    backup_dir = Path(args.backup_dir).resolve() if args.backup_dir else None
    files = sorted(base_dir.glob(args.glob))
    totals = defaultdict(int)
    for path in files:
        stats = normalize_ass_file(path, backup_dir)
        for key, value in stats.items():
            totals[key] += value
    print(
        "files={files} changed={changed} dialogues_before={before} pairs_merged={pairs} "
        "legacy_singles={singles} dialogues_after={after}".format(
            files=len(files),
            changed=totals["changed"],
            before=totals["dialogues_before"],
            pairs=totals["pairs_merged"],
            singles=totals["legacy_singles"],
            after=totals["dialogues_after"],
        )
    )


if __name__ == "__main__":
    main()
