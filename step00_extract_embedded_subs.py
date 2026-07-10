import argparse
import html
import json
import re
import subprocess
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from common import add_common_args, save_json, selected_videos, work_dir_for, write_status


TEXT_SUBTITLE_CODECS = {"subrip", "ass", "ssa", "mov_text", "webvtt"}
MUSIC_SYMBOL = "\u266a"
MUSIC_SYMBOLS = {MUSIC_SYMBOL, "\u266b"}
TIME_RE = re.compile(
    r"(?P<start>\d+:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(?P<end>\d+:\d{2}:\d{2}[,.]\d{1,3})"
)
TAG_RE = re.compile(r"<[^>]+>")
ITALIC_RE = re.compile(r"</?\s*i\b[^>]*>", re.IGNORECASE)
INLINE_LABEL_RE = re.compile(r"[\[(]([^()\[\]\n]{1,90})[\])]")
SDH_TITLE_RE = re.compile(r"\b(sdh|cc|closed captions?|hearing impaired|hi)\b", re.IGNORECASE)
CHANT_LABEL_RE = re.compile(r"\b(chant(?:ing)?|incantation|spell(?:casting)?|whispering continues)\b", re.IGNORECASE)
MUSIC_LABEL_RE = re.compile(r"\b(song|music|singing|sings|lyrics?)\b", re.IGNORECASE)
NON_MUSIC_SOUND_RE = re.compile(r"\b(birds?|phone|ringing|doorbell|alarm|siren)\b", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?")
MOJIBAKE_MUSIC_MARK_RE = re.compile(
    r"(?<![0-9A-Za-z])(?:[jJ][“”\"']|[jJ]n(?=\s))|[jJ][“”\"'](?=\s|$)"
)
COMMON_ENGLISH_WORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "back",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "come",
    "could",
    "did",
    "do",
    "does",
    "dont",
    "down",
    "for",
    "from",
    "get",
    "go",
    "got",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "just",
    "know",
    "let",
    "like",
    "me",
    "my",
    "no",
    "not",
    "now",
    "of",
    "oh",
    "on",
    "one",
    "or",
    "our",
    "out",
    "right",
    "see",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "to",
    "up",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}
KNOWN_CHANT_WORDS = {
    "accessum",
    "ad",
    "anima",
    "animo",
    "animam",
    "apne",
    "colligo",
    "conepiya",
    "det",
    "er",
    "gluttuli",
    "isk",
    "iskoristi",
    "laekna",
    "lames",
    "les",
    "louvri",
    "marcam",
    "magia",
    "miquiliztli",
    "pemet",
    "pémét",
    "pou",
    "repo",
    "revele",
    "soi",
    "som",
    "spasiti",
    "suam",
    "terras",
    "tillate",
    "toxicatus",
    "tollox",
    "ulaz",
    "vuca",
    "vuka",
    "votre",
    "vrai",
    "vrata",
    "yoli",
}
STRONG_CHANT_WORDS = KNOWN_CHANT_WORDS - {"det", "er", "som", "soi"}
CHANT_PHRASES = {
    ("yoli", "conepiya", "miquiliztli"),
    ("spasiti", "animam", "suam"),
    ("animam", "suam"),
    ("tillate", "ulaz"),
    ("louvri", "animo"),
    ("pémét", "pou", "accessum"),
    ("pemet", "pou", "accessum"),
    ("les", "lames", "colligo"),
    ("ad", "me", "gluttuli"),
    ("anima", "marcam"),
    ("iskoristi", "vuka"),
    ("iskoristi", "vuca"),
    ("magia", "tollox", "de", "terras"),
    ("repo", "oma", "dal", "most"),
}
SPEAKER_TAG_ONLY_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9 .'\-]{1,40})(?:\s*\((?:on phone|o\.s\.|v\.o\.|over phone)\))?\s*:?\s*$"
)
LEADING_SPEAKER_TAG_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9 .'\-]{1,40})(?:\s*\((?:on phone|o\.s\.|v\.o\.|over phone)\))?\s*:\s*"
)
NON_DIALOGUE_SOUND_WORDS = {
    "alarm",
    "alarms",
    "beep",
    "beeps",
    "bell",
    "bells",
    "bird",
    "birds",
    "breathing",
    "buzz",
    "buzzes",
    "chime",
    "chimes",
    "chuckling",
    "chuckles",
    "clatter",
    "clatters",
    "close",
    "closes",
    "closing",
    "coughing",
    "coughs",
    "crack",
    "cracking",
    "cracks",
    "creak",
    "creaking",
    "creaks",
    "cries",
    "crying",
    "ding",
    "dings",
    "door",
    "doors",
    "exhale",
    "exhales",
    "footstep",
    "footsteps",
    "gasp",
    "gasping",
    "gasps",
    "giggle",
    "giggling",
    "giggles",
    "groan",
    "groaning",
    "groans",
    "grunt",
    "grunting",
    "grunts",
    "knock",
    "knocking",
    "knocks",
    "laugh",
    "laughing",
    "laughs",
    "moan",
    "moaning",
    "moans",
    "open",
    "opens",
    "opening",
    "panting",
    "pants",
    "phone",
    "phones",
    "rain",
    "ring",
    "ringing",
    "rings",
    "rattle",
    "rattling",
    "rattles",
    "rustle",
    "rustles",
    "rustling",
    "scream",
    "screaming",
    "screams",
    "shriek",
    "shrieking",
    "shrieks",
    "sigh",
    "sighing",
    "sighs",
    "sirens",
    "siren",
    "slam",
    "slamming",
    "slams",
    "sob",
    "sobbing",
    "sobs",
    "thunder",
    "voice",
    "voices",
    "whimper",
    "whimpering",
    "whimpers",
    "whoosh",
    "whooshes",
    "wind",
}
NON_DIALOGUE_CONTEXT_WORDS = {
    "background",
    "car",
    "cell",
    "continued",
    "continues",
    "continue",
    "distant",
    "faint",
    "indistinct",
    "nearby",
    "off",
    "phone",
    "quietly",
    "screen",
    "softly",
    "stopped",
    "stops",
    "stop",
    "trolley",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 STEP0: extract and normalize embedded English subtitles.")
    add_common_args(parser)
    parser.add_argument("--force", action="store_true", help="Regenerate embedded subtitle files and JSON.")
    parser.add_argument(
        "--keep-sdh-cues",
        action="store_true",
        help="Keep SDH-only non-dialogue cues in the normalized subtitle JSON.",
    )
    parser.add_argument(
        "--prefer-sdh",
        action="store_true",
        help="Prefer an English SDH/CC subtitle stream as the main embedded subtitle source when one exists.",
    )
    return parser.parse_args()


def ffprobe_json(video: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "s",
        "-show_streams",
        "-of",
        "json",
        str(video),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video.name}: {result.stderr.strip()}")
    return json.loads(result.stdout or "{}")


def stream_text(stream: dict) -> str:
    tags = stream.get("tags") or {}
    parts = [
        str(stream.get("codec_name", "")),
        str(tags.get("language", "")),
        str(tags.get("title", "")),
        str(tags.get("handler_name", "")),
    ]
    disposition = stream.get("disposition") or {}
    parts.extend(key for key, value in disposition.items() if value)
    return " ".join(parts)


def is_english_stream(stream: dict) -> bool:
    tags = stream.get("tags") or {}
    lang = str(tags.get("language", "")).strip().lower()
    title = str(tags.get("title", "")).strip().lower()
    return lang in {"eng", "en", "english"} or "english" in title or not lang


def is_sdh_stream(stream: dict) -> bool:
    return bool(SDH_TITLE_RE.search(stream_text(stream)))


def subtitle_streams(video: Path) -> list[dict]:
    streams = []
    for stream in ffprobe_json(video).get("streams", []):
        codec = str(stream.get("codec_name", "")).lower()
        if codec in TEXT_SUBTITLE_CODECS and is_english_stream(stream):
            streams.append(stream)
    return streams


def choose_streams(streams: list[dict], prefer_sdh: bool = False) -> tuple[dict | None, dict | None]:
    if not streams:
        return None, None
    normal = next((stream for stream in streams if not is_sdh_stream(stream)), None)
    sdh = next((stream for stream in streams if is_sdh_stream(stream)), None)
    if prefer_sdh and sdh is not None:
        normal = sdh
    if normal is None:
        normal = streams[0]
    if sdh is None and is_sdh_stream(normal):
        sdh = normal
    return normal, sdh


def extract_stream(video: Path, stream: dict, out_file: Path, force: bool) -> None:
    if out_file.exists() and not force:
        return
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-map",
        f"0:{stream['index']}",
        "-c:s",
        "srt",
        str(out_file),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg subtitle extraction failed for {video.name}: {result.stderr.strip()}")


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def parse_srt_time(value: str) -> float:
    head, fraction = value.replace(",", ".").split(".", 1)
    hours, minutes, seconds = [int(part) for part in head.split(":")]
    millis = int((fraction + "000")[:3])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def normalize_inline_text(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = normalize_music_symbols(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [part.strip() for part in text.split("\n") if part.strip()]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def normalize_music_symbols(text: str) -> str:
    return MOJIBAKE_MUSIC_MARK_RE.sub(MUSIC_SYMBOL, str(text or ""))


def strip_tags(text: str) -> str:
    return normalize_music_symbols(TAG_RE.sub("", html.unescape(str(text or ""))))


def strip_leading_speaker_tag(text: str) -> str:
    text = normalize_inline_text(text)
    while text:
        updated = LEADING_SPEAKER_TAG_RE.sub("", text, count=1).strip()
        if updated == text:
            break
        text = updated
    return normalize_inline_text(text)


def speaker_tag_only(text: str) -> bool:
    return bool(SPEAKER_TAG_ONLY_RE.fullmatch(normalize_inline_text(text)))


def split_sdh_labels(raw_text: str) -> tuple[list[str], str]:
    plain = normalize_inline_text(strip_tags(raw_text))
    labels = [match.group(1).strip() for match in INLINE_LABEL_RE.finditer(plain)]
    body = INLINE_LABEL_RE.sub(" ", plain)
    body = normalize_inline_text(body)
    return labels, body


def looks_like_effect_only_text(text: str, labels: list[str]) -> bool:
    text = strip_leading_speaker_tag(text)
    if not text:
        return False
    if has_music_text(text) or any(label_is_music(label) or label_is_chant(label) for label in labels):
        return False
    tokens = [word.lower().replace("'", "") for word in WORD_RE.findall(text)]
    if not tokens:
        return False
    if any(word in STRONG_CHANT_WORDS for word in tokens):
        return False
    sound_hits = sum(1 for word in tokens if word in NON_DIALOGUE_SOUND_WORDS)
    context_hits = sum(1 for word in tokens if word in NON_DIALOGUE_CONTEXT_WORDS)
    if sound_hits == 0:
        return False
    if len(tokens) == 1:
        return True
    covered_ratio = (sound_hits + context_hits) / max(len(tokens), 1)
    if len(tokens) <= 6 and covered_ratio >= 0.75:
        return True
    if len(tokens) <= 3 and covered_ratio >= 0.5 and sum(word in COMMON_ENGLISH_WORDS for word in tokens) == 0:
        return True
    return False


def parse_srt(path: Path) -> list[dict]:
    if not path.exists():
        return []
    blocks = re.split(r"\n\s*\n", read_text(path).replace("\r\n", "\n").replace("\r", "\n").strip())
    cues = []
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        time_index = next((index for index, line in enumerate(lines) if "-->" in line), -1)
        if time_index < 0:
            continue
        match = TIME_RE.search(lines[time_index])
        if not match:
            continue
        raw_text = "\n".join(lines[time_index + 1 :]).strip()
        labels, body = split_sdh_labels(raw_text)
        raw_plain = normalize_inline_text(strip_tags(raw_text))
        text = strip_leading_speaker_tag(body or raw_plain)
        cues.append(
            {
                "start": parse_srt_time(match.group("start")),
                "end": parse_srt_time(match.group("end")),
                "raw": raw_text,
                "text": text,
                "raw_plain": raw_plain,
                "labels": labels,
                "pure_label": bool(labels and not body) or speaker_tag_only(raw_plain),
                "italic": bool(ITALIC_RE.search(raw_text)),
                "has_music": any(symbol in raw_plain for symbol in MUSIC_SYMBOLS),
            }
        )
    return cues


def overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def normalize_compare(text: str) -> str:
    text = strip_tags(text).lower()
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())


def text_similarity(left: str, right: str) -> float:
    left_norm = normalize_compare(left)
    right_norm = normalize_compare(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def label_is_chant(label: str) -> bool:
    return bool(CHANT_LABEL_RE.search(label or ""))


def label_is_music(label: str) -> bool:
    label = str(label or "")
    return bool(MUSIC_LABEL_RE.search(label)) and not NON_MUSIC_SOUND_RE.search(label)


def has_music_text(text: str) -> bool:
    return any(symbol in str(text or "") for symbol in MUSIC_SYMBOLS)


def repeated_phrase(text: str) -> bool:
    parts = [normalize_compare(part) for part in re.split(r"[.!?;]+", text) if normalize_compare(part)]
    counts = Counter(parts)
    return any(count >= 2 and len(part.split()) >= 2 for part, count in counts.items())


def looks_like_chant_text(text: str, italic: bool, labels: list[str], repeated_count: int) -> bool:
    words = [word.lower().replace("'", "") for word in WORD_RE.findall(text or "")]
    if not words:
        return False
    chant_phrase_hits = 0
    chant_phrase_words = 0
    index = 0
    phrase_items = sorted(CHANT_PHRASES, key=len, reverse=True)
    while index < len(words):
        matched_size = 0
        for phrase in phrase_items:
            size = len(phrase)
            if tuple(words[index : index + size]) == phrase:
                matched_size = size
                break
        if matched_size:
            chant_phrase_hits += 1
            chant_phrase_words += matched_size
            index += matched_size
        else:
            index += 1
    phrase_ratio = chant_phrase_words / max(len(words), 1)
    if chant_phrase_hits and phrase_ratio >= 0.6:
        return True
    common_ratio = sum(1 for word in words if word in COMMON_ENGLISH_WORDS) / max(len(words), 1)
    if any(label_is_chant(label) for label in labels):
        return common_ratio < 0.55 or any(word in STRONG_CHANT_WORDS for word in words)
    chant_hits = sum(1 for word in words if word in KNOWN_CHANT_WORDS)
    strong_chant_hits = sum(1 for word in words if word in STRONG_CHANT_WORDS)
    chant_ratio = chant_hits / max(len(words), 1)
    if chant_ratio >= 0.6 or (len(words) <= 4 and chant_hits >= 2):
        return True
    if repeated_phrase(text) and common_ratio <= 0.45:
        return True
    if italic and repeated_count >= 2:
        return common_ratio <= 0.45
    if italic and len(words) <= 7 and any(word.endswith(("atus", "atum", "icus", "ae")) for word in words):
        return True
    return False


def collect_sdh_labels(cue: dict, sdh_cues: list[dict]) -> list[str]:
    labels: list[str] = []
    for sdh in sdh_cues:
        cue_overlap = overlap(cue["start"], cue["end"], sdh["start"], sdh["end"])
        near_before = not bool(sdh.get("pure_label")) and bool(sdh.get("text")) and 0.0 <= cue["start"] - sdh["end"] <= 1.25
        near_after = not bool(sdh.get("pure_label")) and 0.0 <= sdh["start"] - cue["end"] <= 0.75
        similar = text_similarity(cue.get("text", ""), sdh.get("text", "")) >= 0.82
        if cue_overlap >= 0.08 or near_before or near_after or similar:
            labels.extend(sdh.get("labels", []))
    return sorted(set(label for label in labels if label))


def make_segment(cue: dict, labels: list[str], text_counts: Counter, source: str, index: int) -> dict | None:
    text = strip_leading_speaker_tag(cue.get("text", ""))
    if not text:
        return None
    if looks_like_effect_only_text(text, labels):
        return None
    if any(label_is_chant(label) for label in labels) and not bool(cue.get("italic")) and text.isascii():
        words = [word.lower().replace("'", "") for word in WORD_RE.findall(text)]
        english_ratio = sum(1 for word in words if word in COMMON_ENGLISH_WORDS) / max(len(words), 1) if words else 0.0
        if english_ratio >= 0.55 and not repeated_phrase(text):
            labels = [label for label in labels if not label_is_chant(label)]
    is_music = cue.get("has_music", False) or has_music_text(text) or any(label_is_music(label) for label in labels)
    if is_music and not WORD_RE.search(text):
        return None
    is_chant = looks_like_chant_text(text, bool(cue.get("italic")), labels, text_counts[normalize_compare(text)])
    kind = "chant" if is_chant else ("lyric" if is_music else "dialogue")
    return {
        "start": round(float(cue["start"]), 3),
        "end": round(float(cue["end"]), 3),
        "text": text,
        "en_wrap": text,
        "raw_text": cue.get("raw", ""),
        "source": source,
        "kind": kind,
        "is_music": bool(is_music),
        "is_chant": bool(is_chant),
        "italic": bool(cue.get("italic")),
        "sdh_labels": labels,
        "embedded_index": index,
    }


def build_segments(normal_cues: list[dict], sdh_cues: list[dict], keep_sdh_cues: bool) -> list[dict]:
    main_cues = [cue for cue in normal_cues if not cue.get("pure_label") and normalize_inline_text(cue.get("text", ""))]
    text_counts = Counter(normalize_compare(cue.get("text", "")) for cue in main_cues)
    segments: list[dict] = []
    for index, cue in enumerate(main_cues, start=1):
        labels = collect_sdh_labels(cue, sdh_cues)
        segment = make_segment(cue, labels, text_counts, "embedded", index)
        if segment:
            segments.append(segment)

    for index, cue in enumerate(sdh_cues, start=1):
        if cue.get("pure_label"):
            continue
        if any(overlap(cue["start"], cue["end"], seg["start"], seg["end"]) >= 0.12 for seg in segments):
            continue
        labels = list(cue.get("labels", []))
        temp_counts = Counter({normalize_compare(cue.get("text", "")): 2 if cue.get("italic") else 1})
        segment = make_segment(cue, labels, temp_counts, "embedded_sdh_only", index)
        if not segment:
            continue
        if keep_sdh_cues or segment["is_music"] or segment["is_chant"]:
            segments.append(segment)

    segments.sort(key=lambda item: (item["start"], item["end"]))
    for index, segment in enumerate(segments, start=1):
        segment["index"] = index
    return segments


def stream_summary(stream: dict | None) -> dict | None:
    if not stream:
        return None
    tags = stream.get("tags") or {}
    return {
        "index": stream.get("index"),
        "codec": stream.get("codec_name"),
        "language": tags.get("language", ""),
        "title": tags.get("title", ""),
        "is_sdh": is_sdh_stream(stream),
    }


def process_video(base_dir: Path, work_dir: Path, video: Path, force: bool, keep_sdh_cues: bool, prefer_sdh: bool) -> bool:
    embedded_subs_dir = work_dir / "embedded_subs"
    embedded_json_dir = work_dir / "embedded_json"
    out_json = embedded_json_dir / f"{video.stem}.json"
    if out_json.exists() and not force:
        print(f"[skip] {out_json.name} exists")
        return True

    streams = subtitle_streams(video)
    normal_stream, sdh_stream = choose_streams(streams, prefer_sdh=prefer_sdh)
    if not normal_stream:
        print(f"[skip] no text English subtitle stream: {video.name}")
        return False

    normal_srt = embedded_subs_dir / f"{video.stem}.eng.srt"
    sdh_srt = embedded_subs_dir / f"{video.stem}.eng.sdh.srt"
    extract_stream(video, normal_stream, normal_srt, force=force)
    if sdh_stream:
        extract_stream(video, sdh_stream, sdh_srt, force=force)
    else:
        sdh_srt = normal_srt

    normal_cues = parse_srt(normal_srt)
    sdh_cues = parse_srt(sdh_srt) if sdh_srt.exists() else []
    segments = build_segments(normal_cues, sdh_cues, keep_sdh_cues)
    if not segments:
        print(f"[skip] no usable embedded subtitle cues: {video.name}")
        return False

    data = {
        "video": video.name,
        "stem": video.stem,
        "source": "embedded_subtitle",
        "streams": {
            "normal": stream_summary(normal_stream),
            "sdh": stream_summary(sdh_stream),
        },
        "counts": {
            "normal_cues": len(normal_cues),
            "sdh_cues": len(sdh_cues),
            "segments": len(segments),
            "music_segments": sum(1 for seg in segments if seg.get("is_music")),
            "chant_segments": sum(1 for seg in segments if seg.get("is_chant")),
        },
        "segments": segments,
    }
    save_json(out_json, data)
    print(
        f"[done] {video.name}: segments={data['counts']['segments']} "
        f"music={data['counts']['music_segments']} chant={data['counts']['chant_segments']}"
    )
    return True


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    videos = selected_videos(base_dir, args.chunk, args.target_stem)
    if not videos:
        print(f"[error] no target videos found in {base_dir}")
        sys.exit(1)

    write_status(base_dir, "V2_STEP0", f"{len(videos)} files", "START embedded subtitle extraction")
    start = time.time()
    available = 0
    for video in videos:
        if process_video(base_dir, work_dir, video, args.force, args.keep_sdh_cues, args.prefer_sdh):
            available += 1
    elapsed = time.time() - start
    write_status(
        base_dir,
        "V2_STEP0",
        f"{available}/{len(videos)} files",
        f"DONE embedded subtitle extraction ({elapsed:.1f}s)",
    )
    print(f"v2 step0 done; embedded subtitle JSON available for {available}/{len(videos)} file(s)")


if __name__ == "__main__":
    main()
