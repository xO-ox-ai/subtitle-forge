import argparse
import re
from collections import Counter
from pathlib import Path

from common import save_json, work_dir_for
from step00_extract_embedded_subs import (
    build_segments,
    has_music_text,
    label_is_chant,
    make_segment,
    normalize_compare,
    parse_srt,
)


VOCAL_MUSIC_RE = re.compile(r"\b(singing|sings?|song|lyrics?|vocal(?:s|izing)?)\b", re.IGNORECASE)
EDGE_SPEAKER_DASH_RE = re.compile(r"^(?:[-\u2013\u2014]\s*)+|(?:\s*[-\u2013\u2014])+$")
DESCRIPTIVE_VOICE_TAG = r"(?:FEMALE|MALE)\s+VOICE(?:\s+in\s+[A-Z][A-Za-z -]{1,30})?"
UPPERCASE_SPEAKER_TAG = r"[A-Z][A-Z0-9 .'-]{1,40}"
LEADING_SDH_SPEAKER_RE = re.compile(
    rf"^(?:{DESCRIPTIVE_VOICE_TAG}|{UPPERCASE_SPEAKER_TAG})\s*:\s*"
)
INLINE_SDH_SPEAKER_RE = re.compile(
    rf"\s*(?:-{{1,3}}|[\u2013\u2014])\s*(?:{DESCRIPTIVE_VOICE_TAG}|{UPPERCASE_SPEAKER_TAG})\s*:\s*"
)


def clean_sdh_dialogue_text(text: str) -> str:
    """Remove SDH speaker attribution while preserving speaker turns."""
    text = str(text or "").strip()
    text = re.sub(r"^\s*:\s*", "", text)
    text = LEADING_SDH_SPEAKER_RE.sub("", text, count=1)
    text = INLINE_SDH_SPEAKER_RE.sub(" \u2014 ", text)
    text = re.sub(r"\s+-\s+:\s*", " \u2014 ", text)
    return re.sub(r"\s+", " ", text).strip()


def import_english_sdh_srt(source: Path) -> dict:
    """Convert an English SDH SRT into Step 7 dialogue JSON.

    Pure sound descriptions are omitted.  A sound/music label attached to
    spoken text remains metadata and does not turn the spoken line into a
    lyric.  Chant styling is reserved for cues explicitly labelled as a
    chant/incantation; italic foreign or off-screen speech stays dialogue.
    """
    cues = parse_srt(source)
    text_counts = Counter(normalize_compare(cue.get("text", "")) for cue in cues)
    segments: list[dict] = []
    skipped_pure_labels = 0
    skipped_non_dialogue = 0

    for source_index, cue in enumerate(cues, start=1):
        if cue.get("pure_label"):
            skipped_pure_labels += 1
            continue
        labels = list(cue.get("labels", []))
        segment = make_segment(cue, labels, text_counts, "external_english_sdh", source_index)
        if not segment:
            skipped_non_dialogue += 1
            continue

        text = EDGE_SPEAKER_DASH_RE.sub("", str(segment.get("text", ""))).strip()
        text = clean_sdh_dialogue_text(text)
        if not text or not any(character.isalnum() for character in text):
            skipped_non_dialogue += 1
            continue
        segment["text"] = text
        segment["en_wrap"] = text

        explicit_chant = any(label_is_chant(label) for label in labels)
        vocal_music = has_music_text(text) or any(VOCAL_MUSIC_RE.search(label or "") for label in labels)
        segment["is_chant"] = bool(explicit_chant)
        segment["is_music"] = bool(vocal_music)
        segment["kind"] = "chant" if explicit_chant else ("lyric" if vocal_music else "dialogue")
        segments.append(segment)

    segments.sort(key=lambda item: (float(item["start"]), float(item["end"]), item["embedded_index"]))
    for index, segment in enumerate(segments, start=1):
        segment["index"] = index

    return {
        "video": "",
        "stem": source.stem,
        "source": "external_english_sdh_srt",
        "source_file": source.name,
        "streams": {"normal": None, "sdh": {"language": "eng", "is_sdh": True}},
        "counts": {
            "source_cues": len(cues),
            "segments": len(segments),
            "skipped_pure_labels": skipped_pure_labels,
            "skipped_non_dialogue": skipped_non_dialogue,
            "music_segments": sum(1 for segment in segments if segment.get("is_music")),
            "chant_segments": sum(1 for segment in segments if segment.get("is_chant")),
        },
        "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import an English SDH SRT for the subtitle-only pipeline.")
    parser.add_argument("base_dir", nargs="?", default=".")
    parser.add_argument("--work-dir", default="temp")
    parser.add_argument("--source", required=True, help="English SRT path, relative to base_dir unless absolute.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    source = Path(args.source)
    if not source.is_absolute():
        source = base_dir / source
    source = source.resolve()
    if not source.exists() or source.suffix.lower() != ".srt":
        raise FileNotFoundError(f"English SRT not found: {source}")

    out_file = work_dir / "embedded_json" / f"{source.stem}.json"
    if out_file.exists() and not args.force:
        print(f"[skip] {out_file.name} exists")
        return
    data = import_english_sdh_srt(source)
    if not data["segments"]:
        raise RuntimeError(f"No usable dialogue cues found: {source}")
    save_json(out_file, data)
    counts = data["counts"]
    print(
        "english SRT import done; source_cues={source_cues}, segments={segments}, "
        "skipped_pure_labels={skipped_pure_labels}, skipped_non_dialogue={skipped_non_dialogue}, "
        "music={music_segments}, chant={chant_segments}, output={output}".format(
            **counts, output=out_file
        )
    )


if __name__ == "__main__":
    main()
