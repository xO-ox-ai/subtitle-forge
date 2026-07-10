import json
import math
import os
import re
from difflib import SequenceMatcher
from pathlib import Path


LONG_OCR_SECONDS = float(os.environ.get("REPAIR_OCR_MIN_SECONDS", "8.0"))
MAX_REBUILT_SEGMENT_SECONDS = 6.0
MAX_CLUSTER_GAP_SECONDS = 2.25
MAX_CENTER_JUMP_PIXELS = 420.0
MAX_SEGMENTS_PER_EVENT = int(os.environ.get("REPAIR_OCR_MAX_SEGMENTS", "8"))
FALLBACK_SEGMENT_SECONDS = 6.0

ASS_TAG_RE = re.compile(r"\{[^}]*\}")
POS_RE = re.compile(r"\\pos\((-?\d+),(-?\d+)\)")


def ass_to_seconds(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, centiseconds = rest.split(".")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(centiseconds) / 100.0
    )


def seconds_to_ass(value: float) -> str:
    value = max(0.0, float(value))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = int(value % 60)
    centiseconds = int((value - int(value)) * 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open(encoding="utf-8-sig") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return default


def visible_text(text: str) -> str:
    return ASS_TAG_RE.sub("", text).replace(r"\N", " ").strip()


def normalize_zh(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "", str(text or "").lower())


def text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def ocr_similarity_score(left: str, right: str) -> float:
    lkey = text_key(left)
    rkey = text_key(right)
    if not lkey or not rkey:
        return 0.0
    score = ratio(lkey, rkey)
    if lkey == rkey:
        score = max(score, 1.0)
    small, big = (lkey, rkey) if len(lkey) <= len(rkey) else (rkey, lkey)
    if len(small) >= 4 and small in big:
        score = max(score, 0.82 + 0.18 * len(small) / len(big))
    return score


def similar_ocr_text(left: str, right: str) -> bool:
    lkey = text_key(left)
    rkey = text_key(right)
    if len(lkey) < 3 or len(rkey) < 3:
        return bool(lkey and lkey == rkey)
    return ocr_similarity_score(left, right) >= 0.74


def bbox_center(bbox) -> tuple[float, float] | None:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def scaled_ocr_position(item: dict, width: int, height: int, play_x: int, play_y: int) -> tuple[int, int]:
    bbox = item.get("bbox") or []
    center = bbox_center(bbox)
    if center is None:
        x, y = width / 2.0, height * 0.18
    else:
        x, y = center
    margin = 36
    x = min(max(margin, x), max(margin, width - margin))
    y = min(max(margin, y), max(margin, height - margin))

    scale_x = play_x / width if width else 1.0
    scale_y = play_y / height if height else 1.0
    x = int(round(x * scale_x))
    y = int(round(y * scale_y))

    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            by1, by2 = float(bbox[1]), float(bbox[3])
            if by2 < by1:
                by1, by2 = by2, by1
            half_h = int(round((by2 - by1) / 2.0 * scale_y))
            shift = half_h + 8
            if y - shift >= 8:
                y -= shift
            else:
                y += shift
        except (TypeError, ValueError):
            pass
    return x, y


def replace_pos(text: str, x: int, y: int) -> str:
    replacement = rf"\pos({x},{y})"
    if POS_RE.search(text):
        return POS_RE.sub(lambda _match: replacement, text, count=1)
    if text.startswith("{"):
        return text.replace("{", rf"{{\an5{replacement}", 1)
    return rf"{{\an5{replacement}}}" + text


def ass_play_res(lines: list[str]) -> tuple[int, int]:
    play_x, play_y = 1920, 1080
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("playresx:"):
            try:
                play_x = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif stripped.startswith("playresy:"):
            try:
                play_y = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return play_x, play_y


def split_dialogue(line: str) -> list[str] | None:
    if not line.startswith("Dialogue:"):
        return None
    parts = line.split(",", 9)
    return parts if len(parts) == 10 else None


def make_dialogue(parts: list[str], start: float, end: float, text: str) -> str:
    updated = parts.copy()
    updated[1] = seconds_to_ass(start)
    updated[2] = seconds_to_ass(end)
    updated[9] = text
    return ",".join(updated)


def match_translated_item(items: list[dict], start: float, end: float, zh_text: str) -> dict | None:
    candidates: list[tuple[float, dict]] = []
    visible_key = normalize_zh(zh_text)
    for item in items:
        try:
            item_start = float(item.get("start", 0.0) or 0.0)
            item_end = float(item.get("end", item_start) or item_start)
        except (TypeError, ValueError):
            continue
        if abs(item_start - start) > 0.06 or abs(item_end - end) > 0.06:
            continue
        item_key = normalize_zh(item.get("zh") or "")
        score = ratio(item_key, visible_key)
        if item_key and visible_key and (item_key in visible_key or visible_key in item_key):
            score = max(score, 0.95)
        candidates.append((score, item))
    if candidates:
        return max(candidates, key=lambda row: row[0])[1]

    for item in items:
        try:
            item_start = float(item.get("start", 0.0) or 0.0)
            item_end = float(item.get("end", item_start) or item_start)
        except (TypeError, ValueError):
            continue
        overlap = min(end, item_end) - max(start, item_start)
        if overlap <= 0:
            continue
        item_key = normalize_zh(item.get("zh") or "")
        score = ratio(item_key, visible_key)
        if item_key and visible_key and (item_key in visible_key or visible_key in item_key):
            score = max(score, 0.95)
        if score < 0.68:
            continue
        overlap_ratio = overlap / max(0.5, min(end - start, item_end - item_start))
        candidates.append((score * 10.0 + overlap_ratio, item))
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[0])[1]


def best_detection_per_frame(item: dict, raw: dict, start: float, end: float) -> list[dict]:
    buckets: dict[float, list[dict]] = {}
    item_text = item.get("text") or ""
    for det in raw.get("detections", []):
        try:
            det_start = float(det.get("start", 0.0) or 0.0)
            det_end = float(det.get("end", det_start) or det_start)
        except (TypeError, ValueError):
            continue
        if det_end < start - 0.2 or det_start > end + 0.2:
            continue
        if similar_ocr_text(item_text, det.get("text") or ""):
            buckets.setdefault(round(det_start, 2), []).append(det)

    chosen: list[dict] = []
    previous_center = None
    for frame_start in sorted(buckets):
        best = None
        best_score = -999.0
        for det in buckets[frame_start]:
            current_center = bbox_center(det.get("bbox") or [])
            score = (
                ocr_similarity_score(item_text, det.get("text") or "") * 3.0
                + float(det.get("confidence", 0.0) or 0.0)
            )
            if previous_center is not None and current_center is not None:
                score -= min(center_distance(previous_center, current_center) / 900.0, 1.5)
            if score > best_score:
                best = det
                best_score = score
        if best is not None:
            chosen.append(best)
            previous_center = bbox_center(best.get("bbox") or []) or previous_center
    return chosen


def cluster_detections(detections: list[dict]) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    current: list[dict] = []
    previous = None
    for det in detections:
        if previous is not None:
            gap = float(det.get("start", 0.0) or 0.0) - float(previous.get("end", 0.0) or 0.0)
            previous_center = bbox_center(previous.get("bbox") or [])
            current_center = bbox_center(det.get("bbox") or [])
            jumped = (
                previous_center is not None
                and current_center is not None
                and center_distance(previous_center, current_center) > MAX_CENTER_JUMP_PIXELS
            )
            if gap > MAX_CLUSTER_GAP_SECONDS or jumped:
                if current:
                    clusters.append(current)
                current = []
        current.append(det)
        previous = det
    if current:
        clusters.append(current)
    return clusters


def representative_detection(cluster: list[dict]) -> dict:
    return max(
        cluster,
        key=lambda det: (float(det.get("confidence", 0.0) or 0.0), len(text_key(det.get("text") or ""))),
    )


def rebuild_segments(item: dict, raw: dict, start: float, end: float) -> list[tuple[float, float, dict]]:
    detections = best_detection_per_frame(item, raw, start, end)
    clusters = cluster_detections(detections)
    segments: list[tuple[float, float, dict]] = []
    for cluster in clusters:
        chunk: list[dict] = []
        chunk_start = 0.0
        for det in cluster:
            det_start = float(det.get("start", 0.0) or 0.0)
            det_end = float(det.get("end", det_start) or det_start)
            if chunk and det_end - chunk_start > MAX_REBUILT_SEGMENT_SECONDS:
                seg_start = min(float(item.get("start", 0.0) or 0.0) for item in chunk)
                seg_end = max(float(item.get("end", seg_start) or seg_start) for item in chunk)
                if seg_end - seg_start >= 1.0:
                    segments.append((seg_start, seg_end, representative_detection(chunk)))
                chunk = []
            if not chunk:
                chunk_start = det_start
            chunk.append(det)
        if chunk:
            seg_start = min(float(det.get("start", 0.0) or 0.0) for det in chunk)
            seg_end = max(float(det.get("end", seg_start) or seg_start) for det in chunk)
            if seg_end - seg_start >= 1.0:
                segments.append((seg_start, seg_end, representative_detection(chunk)))

    compact: list[tuple[float, float, dict]] = []
    for seg_start, seg_end, det in segments:
        if compact and seg_start <= compact[-1][1] + 0.2:
            prev_start, prev_end, prev_det = compact[-1]
            merged_end = min(max(prev_end, seg_end), prev_start + MAX_REBUILT_SEGMENT_SECONDS)
            compact[-1] = (prev_start, merged_end, prev_det)
        else:
            compact.append((seg_start, seg_end, det))
    return compact[:MAX_SEGMENTS_PER_EVENT]


def repair_ass_file(ass_file: Path, raw_file: Path, translated_file: Path) -> dict:
    raw = load_json(raw_file, {})
    translated = load_json(translated_file, {})
    items = translated.get("items", []) if isinstance(translated, dict) else []
    lines = ass_file.read_text(encoding="utf-8-sig").splitlines()
    play_x, play_y = ass_play_res(lines)
    width = int(raw.get("width") or 1920) if isinstance(raw, dict) else 1920
    height = int(raw.get("height") or 1080) if isinstance(raw, dict) else 1080

    output: list[str] = []
    stats = {
        "ocr": 0,
        "changed": 0,
        "split": 0,
        "fallback": 0,
        "added": 0,
        "old_long": 0,
    }

    for line in lines:
        parts = split_dialogue(line)
        if parts is None or parts[3] != "OCR_TRANSLATION":
            output.append(line)
            continue

        stats["ocr"] += 1
        start = ass_to_seconds(parts[1])
        end = ass_to_seconds(parts[2])
        duration = end - start
        if duration <= LONG_OCR_SECONDS:
            output.append(line)
            continue

        stats["old_long"] += 1
        item = match_translated_item(items, start, end, visible_text(parts[9]))
        if item is None:
            new_end = min(end, start + FALLBACK_SEGMENT_SECONDS)
            output.append(make_dialogue(parts, start, new_end, parts[9]))
            stats["changed"] += 1
            stats["fallback"] += 1
            continue

        segments = rebuild_segments(item, raw, start, end)
        if not segments:
            new_end = min(end, start + FALLBACK_SEGMENT_SECONDS)
            output.append(make_dialogue(parts, start, new_end, parts[9]))
            stats["changed"] += 1
            stats["fallback"] += 1
            continue

        stats["changed"] += 1
        if len(segments) > 1:
            stats["split"] += 1
        stats["added"] += len(segments) - 1
        for seg_start, seg_end, det in segments:
            x, y = scaled_ocr_position(det, width, height, play_x, play_y)
            output.append(make_dialogue(parts, seg_start, seg_end, replace_pos(parts[9], x, y)))

    ass_file.write_text("\n".join(output) + "\n", encoding="utf-8-sig")
    return stats


def main() -> None:
    base_dir = Path(".")
    raw_dir = base_dir / "temp" / "ocr_raw"
    translated_dir = base_dir / "temp" / "ocr_translated"
    totals = {"files": 0, "ocr": 0, "changed": 0, "split": 0, "fallback": 0, "added": 0, "old_long": 0}
    for ass_file in sorted(base_dir.glob("*.ass")):
        raw_file = raw_dir / f"{ass_file.stem}.json"
        translated_file = translated_dir / f"{ass_file.stem}.json"
        if not raw_file.exists() or not translated_file.exists():
            continue
        stats = repair_ass_file(ass_file, raw_file, translated_file)
        totals["files"] += 1
        for key in ("ocr", "changed", "split", "fallback", "added", "old_long"):
            totals[key] += stats[key]
        print(
            f"{ass_file.name}: OCR={stats['ocr']} old_long={stats['old_long']} "
            f"changed={stats['changed']} split={stats['split']} fallback={stats['fallback']} "
            f"added={stats['added']}"
        )
    print(
        "TOTAL: "
        + " ".join(f"{key}={value}" for key, value in totals.items())
    )


if __name__ == "__main__":
    main()
