import json
import re
from pathlib import Path


ASS_TAG_RE = re.compile(r"\{[^}]*\}")
ZH_BREAK_CHARS = "，。！？；：、,.!?;: "


def load_json(path: Path, default):
    if not path.exists() or path.stat().st_size == 0:
        return default
    try:
        with path.open(encoding="utf-8-sig") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return default


def seconds_to_ass(value: float) -> str:
    value = max(0.0, float(value))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = int(value % 60)
    centiseconds = int((value - int(value)) * 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def ass_escape(text: str) -> str:
    return str(text or "").replace("{", "(").replace("}", ")").replace("\r\n", "\n").replace("\r", "\n").replace("\n", r"\N")


def break_zh(text: str, max_chars: int = 18) -> str:
    text = " ".join(str(text or "").replace(r"\N", " ").split()).strip()
    if len(text) <= max_chars:
        return text
    lines: list[str] = []
    rest = text
    while len(rest) > max_chars:
        cut = max_chars
        lower = max(1, max_chars - 6)
        for index in range(max_chars, lower - 1, -1):
            if index < len(rest) and rest[index] in ZH_BREAK_CHARS:
                cut = index + 1
                break
        lines.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        lines.append(rest)
    return r"\N".join(lines)


def split_ass_sections(ass_text: str) -> tuple[list[str], list[str]]:
    header: list[str] = []
    events: list[str] = []
    in_events = False
    for line in str(ass_text or "").splitlines():
        if line.strip() == "[Events]":
            in_events = True
            header.append(line)
            continue
        if in_events and line.startswith("Dialogue:"):
            events.append(line)
        else:
            header.append(line)
    return header, events


def _bbox_center(bbox: list | tuple, width: int, height: int) -> tuple[int, int]:
    if len(bbox) >= 4:
        try:
            values = [float(value) for value in bbox[:4]]
            x1, y1, x2, y2 = values
            if x2 < x1 or y2 < y1:
                x, y, w, h = values
                x1, y1, x2, y2 = x, y, x + w, y + h
            return int((x1 + x2) / 2), int((y1 + y2) / 2)
        except (TypeError, ValueError):
            pass
    return width // 2, int(height * 0.18)


def ocr_position(item: dict, width: int, height: int) -> tuple[int, int]:
    bbox = item.get("bbox") or item.get("box") or []
    if isinstance(bbox, (list, tuple)):
        x, y = _bbox_center(bbox, width, height)
    else:
        x, y = width // 2, int(height * 0.18)
    margin = 36
    x = min(max(margin, x), max(margin, width - margin))
    y = min(max(margin, y), max(margin, height - margin))
    return x, y


def make_dialogue(start: float, end: float, style: str, text: str, layer: int = 7) -> str:
    return f"Dialogue: {layer},{seconds_to_ass(start)},{seconds_to_ass(end)},{style},,0,0,0,,{text}"


def make_note_events(data: dict) -> list[str]:
    events: list[str] = []
    for note in data.get("notes", []) if isinstance(data, dict) else []:
        zh = str(note.get("zh") or note.get("note") or note.get("text") or "").strip()
        if not zh:
            continue
        try:
            start = float(note.get("start", 0.0))
            default_end = start + max(3.5, len(re.findall(r"[\u4e00-\u9fff]", zh)) / 5.0)
            end = max(float(note.get("end", default_end)), default_end)
        except (TypeError, ValueError):
            start, end = 0.0, 4.0
        text = r"{\an8}" + ass_escape(break_zh(zh, max_chars=28))
        events.append(make_dialogue(start, end, "EXPLANATION_NOTE", text, layer=7))
    return events
