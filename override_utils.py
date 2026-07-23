from __future__ import annotations

from pathlib import Path
from typing import Any

from common import load_json


TIME_TOLERANCE = 0.35


def load_overrides(base_dir: Path, stem: str) -> dict:
    path = base_dir / "overrides" / f"{stem}.json"
    data = load_json(path, {})
    return data if isinstance(data, dict) else {}


def override_list(overrides: dict, *keys: str) -> list[dict]:
    items: list[dict] = []
    for key in keys:
        value = overrides.get(key, [])
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
    return items


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\\N", " ").split()).strip()


def segment_text(seg: dict) -> str:
    return normalize_text(seg.get("text") or seg.get("en_wrap") or "")


def time_matches(item: dict, seg: dict, tolerance: float = TIME_TOLERANCE) -> bool:
    item_start = as_float(item.get("start"))
    item_end = as_float(item.get("end"))
    if item_start is None and item_end is None:
        return True

    seg_start = as_float(seg.get("start"), 0.0) or 0.0
    seg_end = as_float(seg.get("end"), seg_start) or seg_start
    if item_start is not None and item_end is not None:
        return seg_end >= item_start - tolerance and seg_start <= item_end + tolerance
    if item_start is not None:
        return abs(seg_start - item_start) <= tolerance or seg_start <= item_start <= seg_end
    if item_end is not None:
        return abs(seg_end - item_end) <= tolerance or seg_start <= item_end <= seg_end
    return True


def text_matches(item: dict, seg: dict) -> bool:
    contains = normalize_text(item.get("text_contains"))
    if not contains:
        return True
    return contains.lower() in segment_text(seg).lower()


def matches_segment(item: dict, seg: dict, index: int | None = None) -> bool:
    item_index = item.get("index")
    if item_index is not None and index is not None:
        try:
            if int(item_index) != index:
                return False
        except (TypeError, ValueError):
            return False
    return time_matches(item, seg) and text_matches(item, seg)


def clean_units(units: Any) -> list[dict]:
    if not isinstance(units, list):
        return []
    cleaned: list[dict] = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        en = normalize_text(unit.get("en"))
        zh = normalize_text(unit.get("zh"))
        if en and zh:
            cleaned.append({"en": en, "zh": zh})
    return cleaned


def apply_music_overrides(segments: list[dict], overrides: dict) -> int:
    items = override_list(overrides, "music", "segments")
    changed = 0
    for index, seg in enumerate(segments, start=1):
        for item in items:
            if "is_music" not in item or not matches_segment(item, seg, index):
                continue
            seg["is_music"] = bool(item["is_music"])
            seg["music_reason"] = item.get("reason") or item.get("music_reason") or "manual_override"
            changed += 1
            break
    return changed


def apply_segment_overrides(segments: list[dict], overrides: dict) -> dict:
    items = override_list(overrides, "segments", "translations", "display_units")
    counts = {"segments": 0, "translations": 0, "display_units": 0}
    for index, seg in enumerate(segments, start=1):
        for item in items:
            if not matches_segment(item, seg, index):
                continue
            applied = False
            if "text" in item:
                seg["text"] = normalize_text(item["text"])
                seg["en_wrap"] = seg["text"]
                applied = True
            if "zh" in item:
                seg["zh"] = normalize_text(item["zh"])
                seg["translation_origin"] = str(item.get("translation_origin") or "manual")
                counts["translations"] += 1
                applied = True
            if "is_music" in item:
                seg["is_music"] = bool(item["is_music"])
                applied = True
            units = clean_units(item.get("display_units") or item.get("units"))
            if units:
                seg["display_units"] = units
                seg["display_units_source"] = "override"
                counts["display_units"] += 1
                applied = True
            if item.get("clear_display_units"):
                seg.pop("display_units", None)
                seg.pop("display_units_source", None)
                applied = True
            if applied:
                counts["segments"] += 1
                break
    return counts


def apply_ocr_overrides(data: dict, overrides: dict) -> int:
    drops = override_list(overrides, "drop_ocr", "ocr_drop")
    replacements = override_list(overrides, "ocr")
    if not drops and not replacements:
        return 0

    changed = 0
    kept = []
    for item in data.get("items", []):
        if any(time_matches(rule, item) and text_matches(rule, item) for rule in drops):
            changed += 1
            continue
        for rule in replacements:
            if time_matches(rule, item) and text_matches(rule, item):
                if "zh" in rule:
                    item["zh"] = normalize_text(rule["zh"])
                    changed += 1
                if "text" in rule:
                    item["text"] = normalize_text(rule["text"])
                    changed += 1
                break
        kept.append(item)
    data["items"] = kept
    return changed
