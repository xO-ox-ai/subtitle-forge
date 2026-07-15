import argparse
import html
import re
from dataclasses import dataclass
from pathlib import Path

from ass_filter_helpers import has_chinese
from step08_ass_render import (
    ASS_HEADER,
    CHANT_SYMBOL,
    MUSIC_SYMBOL,
    make_bilingual_text,
    make_dialogue,
)


TIMESTAMP_RE = re.compile(
    r"(?m)^(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{3})[^\r\n]*$"
)
TAG_RE = re.compile(r"<[^>]+>")
INDEX_RE = re.compile(r"^\d+$")


@dataclass(frozen=True)
class BilingualCue:
    start: float
    end: float
    en: str
    zh: str
    style: str


def parse_srt_time(value: str) -> float:
    head, fraction = value.replace(",", ".").split(".", 1)
    hours, minutes, seconds = [int(part) for part in head.split(":")]
    millis = int((fraction + "000")[:3])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def clean_line(value: str) -> str:
    return html.unescape(TAG_RE.sub("", str(value or ""))).strip().replace(r"\N", " ")


def cue_style(en: str, zh: str) -> str:
    visible = f"{en} {zh}"
    if CHANT_SYMBOL in visible:
        return "BILINGUAL_CHANT"
    if MUSIC_SYMBOL in visible:
        return "BILINGUAL_MUSIC"
    return "BILINGUAL"


def parse_bilingual_srt(text: str) -> list[BilingualCue]:
    """Parse standard SRT and files whose cues omit separator blank lines."""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    matches = list(TIMESTAMP_RE.finditer(normalized))
    cues: list[BilingualCue] = []
    for index, match in enumerate(matches):
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        lines = [clean_line(line) for line in normalized[match.end() : body_end].split("\n")]
        lines = [line for line in lines if line]
        if lines and INDEX_RE.fullmatch(lines[-1]):
            lines.pop()
        if not lines:
            continue

        zh_lines = [line for line in lines if has_chinese(line)]
        en_lines = [line for line in lines if not has_chinese(line)]
        zh = " ".join(zh_lines).strip()
        en = " ".join(en_lines).strip()
        if not zh and not en:
            continue

        start = parse_srt_time(match.group("start"))
        end = parse_srt_time(match.group("end"))
        if end <= start:
            continue
        cues.append(BilingualCue(start=start, end=end, en=en, zh=zh, style=cue_style(en, zh)))
    return cues


def render_ass(cues: list[BilingualCue]) -> str:
    lines = [ASS_HEADER.rstrip()]
    for cue in cues:
        # A punctuation-only placeholder lets Step 9 recognize a genuinely
        # missing Chinese translation while keeping the English source visible.
        zh = cue.zh or ("…" if cue.en else "")
        text = make_bilingual_text(zh, cue.en, cue.style)
        lines.append(make_dialogue(cue.start, cue.end, cue.style, text))
    return "\n".join(lines) + "\n"


def convert_file(path: Path, *, force: bool = False) -> tuple[Path, int]:
    output = path.with_suffix(".ass")
    if output.exists() and not force:
        raise FileExistsError(f"output already exists: {output}")
    cues = parse_bilingual_srt(path.read_text(encoding="utf-8-sig"))
    if not cues:
        raise ValueError(f"no usable subtitle cues: {path}")
    output.write_text(render_ass(cues), encoding="utf-8-sig")
    return output, len(cues)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert bilingual SRT files to the current ASS layout.")
    parser.add_argument("base_dir", nargs="?", default=".")
    parser.add_argument("--glob", default="*.srt")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    files = sorted(base_dir.glob(args.glob))
    total = 0
    for path in files:
        output, count = convert_file(path, force=args.force)
        total += count
        print(f"{path.name} -> {output.name}: cues={count}")
    print(f"files={len(files)} cues={total}")


if __name__ == "__main__":
    main()
