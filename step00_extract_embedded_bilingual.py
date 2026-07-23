import argparse
import re
import time
from collections import Counter
from pathlib import Path

from common import add_common_args, save_json, selected_videos, work_dir_for, write_status
from embedded_bitmap_ocr import BITMAP_SUBTITLE_CODECS, extract_bitmap_subtitle
from step00_extract_embedded_subs import (
    TEXT_SUBTITLE_CODECS,
    extract_stream,
    ffprobe_json,
    is_english_stream,
    is_sdh_stream,
    make_segment,
    normalize_compare,
    normalize_inline_text,
    parse_srt,
    split_sdh_labels,
    strip_tags,
    strip_leading_speaker_tag,
    stream_summary,
)


CHINESE_LANGUAGES = {"chi", "zho", "zh", "chs", "cmn", "chinese"}
SIMPLIFIED_MARKERS = ("simplified", "chs", "简体", "简中")
ALIGN_TOLERANCE = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embedded English SDH and Chinese subtitles and render-ready bilingual JSON."
    )
    add_common_args(parser)
    parser.add_argument("--force", action="store_true", help="Regenerate extracted subtitles and bilingual JSON.")
    return parser.parse_args()


def is_chinese_stream(stream: dict) -> bool:
    tags = stream.get("tags") or {}
    language = str(tags.get("language", "")).strip().lower()
    title = str(tags.get("title", "")).strip().lower()
    return language in CHINESE_LANGUAGES or "chinese" in title or "中文" in title


def subtitle_streams(video: Path) -> list[dict]:
    return [
        stream
        for stream in ffprobe_json(video).get("streams", [])
        if str(stream.get("codec_name", "")).lower() in TEXT_SUBTITLE_CODECS | BITMAP_SUBTITLE_CODECS
    ]


def choose_english_sdh(streams: list[dict]) -> dict | None:
    english = [stream for stream in streams if is_english_stream(stream)]
    return next((stream for stream in english if is_sdh_stream(stream)), None) or (english[0] if english else None)


def choose_chinese(streams: list[dict]) -> dict | None:
    chinese = [
        stream
        for stream in streams
        if is_chinese_stream(stream)
        and (
            not str(stream.get("codec_name", "")).lower()
            or str(stream.get("codec_name", "")).lower() in TEXT_SUBTITLE_CODECS
        )
    ]
    if not chinese:
        return None

    def priority(stream: dict) -> tuple[int, int]:
        tags = stream.get("tags") or {}
        title = str(tags.get("title", "")).lower()
        simplified = any(marker in title for marker in SIMPLIFIED_MARKERS)
        return (0 if simplified else 1, 0 if is_sdh_stream(stream) else 1)

    return min(chinese, key=priority)


def english_segments(cues: list[dict]) -> list[dict]:
    usable: list[dict] = []
    for cue in cues:
        if cue.get("pure_label") or not normalize_inline_text(cue.get("text", "")):
            continue
        raw = str(cue.get("raw", ""))
        visible = strip_tags(raw)
        # When SDH interleaves background lyrics with spoken dialogue in one
        # cue, remove complete music-marked spans and align only the dialogue.
        # This also handles both multiline and same-line forms such as
        # ``- Eight... - ♪ lyric ♪``.
        without_music = re.sub(r"[♪♫].*?[♪♫]", " ", visible, flags=re.DOTALL)
        if without_music != visible:
            labels, body = split_sdh_labels(without_music)
            body = strip_leading_speaker_tag(body)
            body = re.sub(r"^[A-Za-z][A-Za-z .'-]{1,30}:\s*", "", body).strip()
            body = re.sub(r"\s*[-\u2013\u2014]\s*$", "", body).strip()
            if any(character.isalnum() for character in body):
                dialogue_cue = dict(cue)
                dialogue_cue.update({"raw": without_music, "text": body, "labels": labels, "has_music": False})
                usable.append(dialogue_cue)
                continue
        usable.append(cue)
    counts = Counter(normalize_compare(cue.get("text", "")) for cue in usable)
    segments: list[dict] = []
    for index, cue in enumerate(usable, start=1):
        segment = make_segment(cue, list(cue.get("labels", [])), counts, "embedded_bilingual_en_sdh", index)
        if segment:
            if segment.get("is_music"):
                cleaned = re.sub(r"^[A-Za-z][A-Za-z .'-]{1,40}:\s*", "", str(segment.get("text", ""))).strip()
                segment["text"] = cleaned
                segment["en_wrap"] = cleaned
                if not cleaned:
                    continue
            segments.append(segment)
    # SDH marks the first cue of a sung run with ``(SINGING)`` while following
    # lyric cues are often only italicized. Carry the music role across a
    # tightly adjacent italic run so OCR engines do not need to recognize a
    # music-note glyph on every line.
    music_run_active = False
    previous_end = 0.0
    for segment in segments:
        gap = float(segment.get("start", 0.0)) - previous_end
        if segment.get("is_music"):
            music_run_active = True
        elif music_run_active and segment.get("italic") and gap <= 1.25:
            segment["is_music"] = True
            segment["kind"] = "lyric"
            segment["is_chant"] = False
        else:
            music_run_active = False
        previous_end = float(segment.get("end", segment.get("start", 0.0)))
    return segments


def chinese_cues(cues: list[dict]) -> list[dict]:
    return [
        cue
        for cue in cues
        if not cue.get("pure_label") and normalize_inline_text(cue.get("text", ""))
    ]


def interval_gap(left: dict, right: dict) -> float:
    if left["end"] < right["start"]:
        return float(right["start"] - left["end"])
    if right["end"] < left["start"]:
        return float(left["start"] - right["end"])
    return 0.0


def alignment_score(left: dict, right: dict) -> float | None:
    gap = interval_gap(left, right)
    if gap > ALIGN_TOLERANCE:
        return None
    overlap = max(0.0, min(float(left["end"]), float(right["end"])) - max(float(left["start"]), float(right["start"])))
    left_duration = max(float(left["end"]) - float(left["start"]), 0.05)
    right_duration = max(float(right["end"]) - float(right["start"]), 0.05)
    overlap_ratio = overlap / min(left_duration, right_duration)
    music_item = left if left.get("is_music") else (right if right.get("is_music") else None)
    # Translation subtitle tracks usually omit background songs. Do not bind
    # a nearby spoken Chinese cue to a music cue merely because their edges
    # are a few milliseconds apart; translated lyrics still match through a
    # substantial real overlap.
    if music_item is not None:
        start_delta = abs(float(left["start"]) - float(right["start"]))
        end_delta = abs(float(left["end"]) - float(right["end"]))
        if overlap <= 0 or overlap_ratio < 0.55 or start_delta > 0.50 or end_delta > 0.50:
            return None
    boundary_delta = abs(float(left["start"]) - float(right["start"])) + abs(float(left["end"]) - float(right["end"]))
    center_delta = abs(
        (float(left["start"]) + float(left["end"])) / 2
        - (float(right["start"]) + float(right["end"])) / 2
    )
    return overlap_ratio * 12.0 - boundary_delta / max(left_duration, right_duration) - center_delta - gap * 4.0


def best_match(item: dict, candidates: list[dict]) -> int | None:
    scored = [(alignment_score(item, candidate), index) for index, candidate in enumerate(candidates)]
    valid = [(score, index) for score, index in scored if score is not None]
    return max(valid, default=(None, None))[1]


def join_text(items: list[dict], field: str) -> str:
    parts = [normalize_inline_text(item.get(field, "")) for item in items]
    return " ".join(part for part in parts if part).strip()


def make_aligned_segment(en_items: list[dict], zh_items: list[dict]) -> dict:
    labels = sorted({label for item in en_items for label in item.get("sdh_labels", []) if label})
    is_music = any(bool(item.get("is_music")) for item in en_items)
    is_chant = any(bool(item.get("is_chant")) for item in en_items)
    return {
        "start": round(min(float(item["start"]) for item in en_items), 3),
        "end": round(max(float(item["end"]) for item in en_items), 3),
        "text": join_text(en_items, "text"),
        "en_wrap": join_text(en_items, "text"),
        "zh": join_text(zh_items, "text"),
        "source": "embedded_bilingual_partial",
        "translation_origin": "embedded_chinese",
        "kind": "chant" if is_chant else ("lyric" if is_music else "dialogue"),
        "is_music": is_music,
        "is_chant": is_chant,
        "sdh_labels": labels,
        "english_cue_count": len(en_items),
        "chinese_cue_count": len(zh_items),
    }


def align_bilingual(english: list[dict], chinese: list[dict]) -> tuple[list[dict], dict]:
    matched_en: set[int] = set()
    matched_zh: set[int] = set()
    segments: list[dict] = []

    dialogue = [(index, item) for index, item in enumerate(english) if not item.get("is_music")]
    music = [(index, item) for index, item in enumerate(english) if item.get("is_music")]

    # Match translated lyrics only when their Chinese timing is a strong
    # overlap and there is no spoken English cue occupying the same interval.
    for en_index, en_item in music:
        candidates: list[tuple[float, int]] = []
        for zh_index, zh_item in enumerate(chinese):
            if zh_index in matched_zh:
                continue
            score = alignment_score(en_item, zh_item)
            if score is None:
                continue
            spoken_overlap = any(
                max(0.0, min(float(en_dialogue["end"]), float(zh_item["end"])) - max(float(en_dialogue["start"]), float(zh_item["start"])))
                >= 0.20
                for _dialogue_index, en_dialogue in dialogue
            )
            if not spoken_overlap:
                candidates.append((score, zh_index))
        if candidates:
            zh_index = max(candidates)[1]
            matched_en.add(en_index)
            matched_zh.add(zh_index)
            segments.append(make_aligned_segment([en_item], [chinese[zh_index]]))

    # Align spoken captions as monotonic blocks. Most cues are 1:1; when one
    # track reflows a sentence over adjacent cues, extend the side whose end
    # time is earlier until both block boundaries agree within 400 ms.
    available_zh = [(index, item) for index, item in enumerate(chinese) if index not in matched_zh]
    en_pos = 0
    zh_pos = 0
    while en_pos < len(dialogue) and zh_pos < len(available_zh):
        en_index, en_item = dialogue[en_pos]
        zh_index, zh_item = available_zh[zh_pos]
        if float(en_item["end"]) < float(zh_item["start"]) - ALIGN_TOLERANCE:
            en_pos += 1
            continue
        if float(zh_item["end"]) < float(en_item["start"]) - ALIGN_TOLERANCE:
            zh_pos += 1
            continue
        current_score = alignment_score(en_item, zh_item)
        next_zh_score = (
            alignment_score(en_item, available_zh[zh_pos + 1][1])
            if zh_pos + 1 < len(available_zh)
            else None
        )
        next_en_score = (
            alignment_score(dialogue[en_pos + 1][1], zh_item)
            if en_pos + 1 < len(dialogue)
            else None
        )
        baseline = current_score if current_score is not None else -999.0
        if next_zh_score is not None and next_zh_score > baseline + 1.0:
            zh_pos += 1
            continue
        if next_en_score is not None and next_en_score > baseline + 1.0:
            en_pos += 1
            continue

        en_block = [(en_index, en_item)]
        zh_block = [(zh_index, zh_item)]
        next_en = en_pos + 1
        next_zh = zh_pos + 1
        while abs(float(en_block[-1][1]["end"]) - float(zh_block[-1][1]["end"])) > 0.40:
            if float(en_block[-1][1]["end"]) < float(zh_block[-1][1]["end"]):
                if next_en >= len(dialogue) or len(en_block) >= 4:
                    break
                candidate = dialogue[next_en]
                if float(candidate[1]["start"]) > float(zh_block[-1][1]["end"]) + ALIGN_TOLERANCE:
                    break
                en_block.append(candidate)
                next_en += 1
            else:
                if next_zh >= len(available_zh) or len(zh_block) >= 4:
                    break
                candidate = available_zh[next_zh]
                if float(candidate[1]["start"]) > float(en_block[-1][1]["end"]) + ALIGN_TOLERANCE:
                    break
                zh_block.append(candidate)
                next_zh += 1

        en_items = [item for _index, item in en_block]
        zh_items = [item for _index, item in zh_block]
        segments.append(make_aligned_segment(en_items, zh_items))
        matched_en.update(index for index, _item in en_block)
        matched_zh.update(index for index, _item in zh_block)
        en_pos = next_en
        zh_pos = next_zh

    # Preserve genuinely sung or chanted SDH content when the Chinese track
    # omits it. Ordinary unmatched SDH cues are discarded.
    for en_index, item in music:
        if en_index in matched_en:
            continue
        preserved = dict(item)
        preserved.update(
            {
                "zh": "",
                "source": "embedded_bilingual_partial_unmatched_music",
                "translation_origin": "missing",
            }
        )
        segments.append(preserved)

    segments.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    for index, segment in enumerate(segments, start=1):
        segment["index"] = index
    stats = {
        "english_segments": len(english),
        "chinese_cues": len(chinese),
        "matched_english": len(matched_en),
        "matched_chinese": len(matched_zh),
        "unmatched_english_removed": len(english) - len(matched_en) - sum(
            1 for index, item in enumerate(english) if index not in matched_en and (item.get("is_music") or item.get("is_chant"))
        ),
        "unmatched_chinese_removed": len(chinese) - len(matched_zh),
        "segments": len(segments),
        "music_segments": sum(1 for segment in segments if segment.get("is_music")),
        "chant_segments": sum(1 for segment in segments if segment.get("is_chant")),
    }
    return segments, stats


def process_video(base_dir: Path, work_dir: Path, video: Path, force: bool) -> bool:
    subtitle_dir = work_dir / "embedded_bilingual_subs"
    embedded_json_dir = work_dir / "embedded_json"
    out_json = embedded_json_dir / f"{video.stem}.json"
    if out_json.exists() and not force:
        print(f"[skip] {out_json.name} exists")
        return True

    streams = subtitle_streams(video)
    english_stream = choose_english_sdh(streams)
    chinese_stream = choose_chinese(streams)
    if not english_stream or not chinese_stream:
        print(f"[skip] missing usable English/Chinese subtitle stream: {video.name}")
        return False

    english_srt = subtitle_dir / f"{video.stem}.eng.sdh.srt"
    chinese_srt = subtitle_dir / f"{video.stem}.chi.srt"
    if str(english_stream.get("codec_name", "")).lower() in TEXT_SUBTITLE_CODECS:
        extract_stream(video, english_stream, english_srt, force=force)
    else:
        extract_bitmap_subtitle(video, english_stream, english_srt, work_dir, force=force)
    extract_stream(video, chinese_stream, chinese_srt, force=force)
    english = english_segments(parse_srt(english_srt))
    chinese = chinese_cues(parse_srt(chinese_srt))
    segments, counts = align_bilingual(english, chinese)
    if not segments:
        print(f"[skip] no aligned bilingual cues: {video.name}")
        return False

    data = {
        "video": video.name,
        "video_path": str(video.relative_to(base_dir)),
        "stem": video.stem,
        "source": "embedded_bilingual_partial",
        "streams": {"english": stream_summary(english_stream), "chinese": stream_summary(chinese_stream)},
        "counts": counts,
        "segments": segments,
    }
    save_json(out_json, data)
    print(
        f"[done] {video.name}: segments={counts['segments']} matched_en={counts['matched_english']}/"
        f"{counts['english_segments']} matched_zh={counts['matched_chinese']}/{counts['chinese_cues']} "
        f"music={counts['music_segments']} chant={counts['chant_segments']}"
    )
    return True


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    videos = selected_videos(base_dir, args.chunk, args.target_stem)
    if not videos:
        raise SystemExit(f"no target videos found in {base_dir}")

    write_status(base_dir, "V2_STEP0", f"{len(videos)} files", "START embedded bilingual subtitle extraction")
    start = time.time()
    available = sum(process_video(base_dir, work_dir, video, args.force) for video in videos)
    elapsed = time.time() - start
    write_status(
        base_dir,
        "V2_STEP0",
        f"{available}/{len(videos)} files",
        f"DONE embedded bilingual extraction ({elapsed:.1f}s)",
    )
    if available != len(videos):
        raise SystemExit(f"embedded bilingual subtitles available for only {available}/{len(videos)} files")
    print(f"embedded bilingual extraction done; files={available} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
