import json
import re
import sys
from pathlib import Path

from common import filter_paths_by_stems, parse_common_args, selected_videos, work_dir_for, write_status
from override_utils import apply_segment_overrides, load_overrides


ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: BILINGUAL,Microsoft YaHei,56,&H00FFFFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,2.5,1,2,42,42,38,1
Style: BILINGUAL_MUSIC,Microsoft YaHei,56,&H0000FFFF,&H000000FF,&H00000000,&H90000000,-1,1,0,0,100,100,0,0,1,2.5,1,2,42,42,38,1
Style: BILINGUAL_CHANT,Microsoft YaHei,56,&H00D7C7FF,&H000000FF,&H00000000,&H90000000,-1,1,0,0,100,100,0,0,1,2.5,1,2,42,42,38,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
MIN_DIALOGUE_DURATION = 0.5
MIN_TRIMMED_DURATION = 0.12
ASS_EVENT_GAP = 0.03
DISPLAY_MAX_DURATION = 5.8
EN_DISPLAY_MAX_CHARS = 78
ZH_DISPLAY_MAX_CHARS = 34
DISPLAY_MIN_DURATION = 1.0
DISPLAY_MIN_WORDS = 3
DISPLAY_LOOKAHEAD = 8
DISPLAY_PAUSE_GAP = 0.55
ZH_FONT_SIZE = 56
EN_FONT_SIZE = 34
MUSIC_SYMBOL = "\u266a"
CHANT_SYMBOL = "\u2726"
SONG_TRANSLATIONS_FILE = Path(__file__).resolve().with_name("embedded_song_translations.json")
ZH_PUNCTUATION = "，。！？；：、,.!?:; "
EN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
MOJIBAKE_MUSIC_MARK_RE = re.compile(
    r"(?<![0-9A-Za-z])(?:[jJ][“”\"']|[jJ]n(?=\s))|[jJ][“”\"'](?=\s|$)"
)


def seconds_to_ass(value: float) -> str:
    value = max(0.0, float(value))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = int(value % 60)
    centiseconds = int((value - int(value)) * 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def make_dialogue(
    start: float,
    end: float,
    style: str,
    text: str,
    layer: int = 0,
    name: str = "",
) -> str:
    return f"Dialogue: {layer},{seconds_to_ass(start)},{seconds_to_ass(end)},{style},{name},0,0,0,,{text}"


def normalize_ass_text(text: str) -> str:
    text = str(text or "")
    text = normalize_music_symbols(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\\N", "\n")
    parts = [part.strip() for part in text.split("\n") if part.strip()]
    return r"\N".join(parts)


def normalize_plain_text(text: str) -> str:
    text = str(text or "")
    text = normalize_music_symbols(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\\N", "\n")
    parts = [part.strip() for part in text.split("\n") if part.strip()]
    return " ".join(parts).strip()


def normalize_music_symbols(text: str) -> str:
    return MOJIBAKE_MUSIC_MARK_RE.sub(MUSIC_SYMBOL, str(text or ""))


def ass_escape(text: str) -> str:
    normalized = normalize_ass_text(text)
    return normalized.replace("{", "(").replace("}", ")")


def make_bilingual_text(zh_text: str, en_text: str, style: str = "BILINGUAL") -> str:
    parts: list[str] = []
    if zh_text:
        parts.append(ass_escape(zh_text))
    if en_text:
        parts.append(r"{\fs" + str(EN_FONT_SIZE) + "}" + ass_escape(en_text))
    return r"\N".join(parts)


def word_start(word: dict, fallback: float) -> float:
    try:
        return float(word.get("start", fallback))
    except (TypeError, ValueError):
        return fallback


def word_end(word: dict, fallback: float) -> float:
    try:
        return float(word.get("end", fallback))
    except (TypeError, ValueError):
        return fallback


def words_text(words: list[dict]) -> str:
    return normalize_plain_text(" ".join(str(word.get("word", "")).strip() for word in words if str(word.get("word", "")).strip()))


def timed_words(seg: dict) -> list[dict]:
    words = [word for word in (seg.get("words") or []) if isinstance(word, dict)]
    return [word for word in words if word.get("start") is not None and word.get("end") is not None]


def en_tokens(text: str) -> list[str]:
    return [match.group(0).replace("’", "'").lower() for match in EN_TOKEN_RE.finditer(str(text or ""))]


def flattened_word_tokens(words: list[dict]) -> list[tuple[str, int]]:
    flattened: list[tuple[str, int]] = []
    for index, word in enumerate(words):
        for token in en_tokens(str(word.get("word", ""))):
            flattened.append((token, index))
    return flattened


def find_token_slice(tokens: list[str], needle: list[str], start: int) -> int | None:
    if not needle or len(needle) > len(tokens) - start:
        return None
    max_start = len(tokens) - len(needle)
    for index in range(start, max_start + 1):
        if tokens[index : index + len(needle)] == needle:
            return index
    return None


def display_needs_split(duration: float, zh_core: str, en_text: str, words: list[dict]) -> bool:
    if len(words) < DISPLAY_MIN_WORDS * 2:
        return False
    return duration > DISPLAY_MAX_DURATION or len(en_text) > EN_DISPLAY_MAX_CHARS or len(zh_core) > ZH_DISPLAY_MAX_CHARS


def boundary_score(words: list[dict], start: int, split_at: int) -> float:
    prev = words[split_at - 1]
    nxt = words[split_at]
    chunk_start = word_start(words[start], 0.0)
    chunk_end = word_end(prev, chunk_start)
    text = words_text(words[start:split_at])
    duration = max(0.0, chunk_end - chunk_start)
    gap = max(0.0, word_start(nxt, chunk_end) - chunk_end)
    token = str(prev.get("word", "")).strip()
    prev_clean = "".join(ch.lower() for ch in token if ch.isalnum())
    next_clean = "".join(ch.lower() for ch in str(nxt.get("word", "")).strip() if ch.isalnum())

    score = -abs(len(text) - EN_DISPLAY_MAX_CHARS * 0.78) / EN_DISPLAY_MAX_CHARS
    score -= abs(duration - DISPLAY_MAX_DURATION * 0.72) / DISPLAY_MAX_DURATION
    if (prev_clean, next_clean) in {
        ("all", "right"),
        ("you", "know"),
        ("i", "mean"),
        ("kind", "of"),
        ("sort", "of"),
        ("going", "to"),
        ("want", "to"),
        ("got", "to"),
        ("have", "to"),
        ("far", "from"),
        ("i", "thought"),
        ("i", "think"),
        ("i", "can"),
        ("i", "could"),
        ("i", "will"),
        ("i", "would"),
        ("i", "want"),
        ("i", "need"),
    }:
        score -= 8.0
    if token.endswith((".", "?", "!", ";", ":")):
        score += 4.0
    elif token.endswith(","):
        score += 2.0
    if gap >= DISPLAY_PAUSE_GAP:
        score += 3.0 + min(gap, 1.5)
    if len(text) > EN_DISPLAY_MAX_CHARS:
        score -= (len(text) - EN_DISPLAY_MAX_CHARS) / EN_DISPLAY_MAX_CHARS * 5.0
    if duration > DISPLAY_MAX_DURATION:
        score -= (duration - DISPLAY_MAX_DURATION) / DISPLAY_MAX_DURATION * 4.0
    return score


def choose_display_split(words: list[dict], start: int) -> int | None:
    if len(words) - start <= DISPLAY_MIN_WORDS * 2:
        return None

    limit = None
    for index in range(start + DISPLAY_MIN_WORDS, len(words) - DISPLAY_MIN_WORDS + 1):
        chunk_start = word_start(words[start], 0.0)
        chunk_end = word_end(words[index - 1], chunk_start)
        text = words_text(words[start:index])
        if chunk_end - chunk_start >= DISPLAY_MAX_DURATION or len(text) >= EN_DISPLAY_MAX_CHARS:
            limit = index
            break
    if limit is None:
        return None

    search_end = min(len(words) - DISPLAY_MIN_WORDS, limit + DISPLAY_LOOKAHEAD)
    best_index = None
    best_score = -999.0
    for index in range(start + DISPLAY_MIN_WORDS, search_end + 1):
        chunk_start = word_start(words[start], 0.0)
        chunk_end = word_end(words[index - 1], chunk_start)
        remaining_start = word_start(words[index], chunk_end)
        if chunk_end - chunk_start < DISPLAY_MIN_DURATION:
            continue
        if word_end(words[-1], remaining_start) - remaining_start < DISPLAY_MIN_DURATION:
            continue
        score = boundary_score(words, start, index)
        if score > best_score:
            best_score = score
            best_index = index

    return best_index or limit


def split_words_for_display(words: list[dict]) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    start = 0
    while start < len(words):
        split_at = choose_display_split(words, start)
        if split_at is None:
            chunks.append(words[start:])
            break
        chunks.append(words[start:split_at])
        start = split_at
    return [chunk for chunk in chunks if chunk]


def best_zh_cut(text: str, previous_cut: int, target: int, max_cut: int) -> int:
    min_cut = previous_cut + 1
    target = max(min_cut, min(target, max_cut))
    candidates = [
        index + 1
        for index in range(min_cut - 1, max_cut)
        if text[index] in ZH_PUNCTUATION
    ]
    nearby = [cut for cut in candidates if abs(cut - target) <= 12]
    if nearby:
        return min(nearby, key=lambda cut: (abs(cut - target), cut < target))
    return target


def split_zh_by_ratios(zh_core: str, en_chunks: list[str]) -> list[str]:
    zh_core = normalize_plain_text(zh_core)
    if len(en_chunks) <= 1 or not zh_core:
        return [zh_core]

    weights = [max(len(chunk), 1) for chunk in en_chunks]
    total_weight = sum(weights)
    cuts: list[int] = []
    previous_cut = 0
    cumulative = 0
    for index, weight in enumerate(weights[:-1]):
        cumulative += weight
        raw_target = round(len(zh_core) * cumulative / total_weight)
        max_cut = len(zh_core) - (len(weights) - index - 1)
        cut = best_zh_cut(zh_core, previous_cut, raw_target, max_cut)
        cuts.append(cut)
        previous_cut = cut

    pieces = []
    start = 0
    for cut in cuts + [len(zh_core)]:
        pieces.append(zh_core[start:cut].strip())
        start = cut
    return pieces


def coalesce_text_chunks(chunks: list[str], max_chunks: int) -> list[str]:
    """Merge adjacent untimed clauses into a small number of readable cues."""
    chunks = [normalize_plain_text(chunk) for chunk in chunks if normalize_plain_text(chunk)]
    if len(chunks) <= max_chunks or max_chunks <= 0:
        return chunks
    total = sum(max(len(chunk), 1) for chunk in chunks)
    result: list[str] = []
    current: list[str] = []
    current_weight = 0
    consumed = 0
    for index, chunk in enumerate(chunks):
        current.append(chunk)
        weight = max(len(chunk), 1)
        current_weight += weight
        consumed += weight
        groups_left = max_chunks - len(result) - 1
        items_left = len(chunks) - index - 1
        target = max((total - (consumed - current_weight)) / max(groups_left + 1, 1), 1)
        if groups_left > 0 and items_left >= groups_left and current_weight >= target:
            result.append(" ".join(current))
            current = []
            current_weight = 0
    if current:
        result.append(" ".join(current))
    return result


def music_core(text: str) -> str:
    return normalize_plain_text(text).strip().strip(MUSIC_SYMBOL).strip()


def with_music_marks(text: str, is_music: bool) -> str:
    text = normalize_plain_text(text)
    if not is_music:
        return text
    core = text.strip().strip(MUSIC_SYMBOL).strip()
    return f"{MUSIC_SYMBOL} {core} {MUSIC_SYMBOL}" if core else ""


def with_chant_marks(text: str, is_chant: bool) -> str:
    text = normalize_plain_text(text)
    if not is_chant:
        return text
    core = text.strip().strip(CHANT_SYMBOL).strip()
    return f"{CHANT_SYMBOL} {core} {CHANT_SYMBOL}" if core else ""


def is_chant_segment(seg: dict) -> bool:
    return str(seg.get("kind", "") or "").lower() == "chant" or bool(seg.get("is_chant"))


def has_music_marks(text: str) -> bool:
    value = normalize_plain_text(text)
    return any(symbol in value for symbol in (MUSIC_SYMBOL, "\u266b", "\u266c")) or bool(
        MOJIBAKE_MUSIC_MARK_RE.search(value)
    )


def song_translation_key(text: str) -> str:
    value = normalize_plain_text(text)
    value = value.replace(MUSIC_SYMBOL, " ").replace("\u266b", " ").replace("\u266c", " ")
    value = re.sub(r"^\s*[-\u2013\u2014]+\s*", "", value)
    value = re.sub(r"\s*[-\u2013\u2014]+\s*$", "", value)
    return re.sub(r"\s+", " ", value).strip()


def load_song_translations() -> dict[str, str]:
    if not SONG_TRANSLATIONS_FILE.exists():
        return {}
    try:
        with SONG_TRANSLATIONS_FILE.open(encoding="utf-8-sig") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(key).strip(): normalize_plain_text(value)
        for key, value in data.items()
        if str(key).strip() and normalize_plain_text(value)
    }


def decorate_zh(text: str, is_music: bool, is_chant: bool) -> str:
    text = with_music_marks(text, is_music)
    if is_chant and not is_music:
        text = with_chant_marks(text, True)
    return text


def semantic_display_parts_by_ratio(units: list[dict], start: float, end: float) -> list[dict]:
    cleaned = []
    for unit in units:
        if not isinstance(unit, dict):
            return []
        en = normalize_plain_text(unit.get("en", ""))
        zh = normalize_plain_text(unit.get("zh", ""))
        if not en or not zh:
            return []
        cleaned.append({"en": en, "zh": zh, "weight": max(len(en_tokens(en)), 1)})
    if len(cleaned) < 2:
        return []

    total_weight = sum(item["weight"] for item in cleaned)
    duration = max(0.0, end - start)
    cursor = start
    parts: list[dict] = []
    for index, item in enumerate(cleaned):
        if index == len(cleaned) - 1:
            part_end = end
        else:
            cursor_weight = sum(prev["weight"] for prev in cleaned[: index + 1])
            part_end = start + duration * cursor_weight / total_weight
        parts.append(
            {
                "start": cursor,
                "end": part_end,
                "zh": item["zh"],
                "en": item["en"],
                "split_source": "qwen_ratio",
            }
        )
        cursor = part_end
    return parts


def semantic_display_parts(seg: dict, start: float, end: float, is_music: bool) -> list[dict]:
    units = seg.get("display_units")
    if not isinstance(units, list) or len(units) < 2:
        return []
    words = timed_words(seg)
    if len(words) < DISPLAY_MIN_WORDS:
        return semantic_display_parts_by_ratio(units, start, end)

    flattened = flattened_word_tokens(words)
    token_values = [token for token, _word_index in flattened]
    if not token_values:
        return []

    chunks: list[dict] = []
    cursor = 0
    previous_end_word = -1
    for unit in units:
        if not isinstance(unit, dict):
            return []
        unit_en = normalize_plain_text(unit.get("en", ""))
        unit_zh = music_core(unit.get("zh", "")) if is_music else normalize_plain_text(unit.get("zh", ""))
        needle = en_tokens(unit_en)
        match_start = find_token_slice(token_values, needle, cursor)
        if match_start is None or match_start != cursor:
            return []
        match_end = match_start + len(needle) - 1
        start_word = flattened[match_start][1]
        end_word = flattened[match_end][1]
        if start_word <= previous_end_word:
            return []
        chunks.append(
            {
                "words": words[start_word : end_word + 1],
                "zh": unit_zh,
                "en": words_text(words[start_word : end_word + 1]),
            }
        )
        previous_end_word = end_word
        cursor = match_end + 1

    if cursor != len(token_values):
        return []

    parts: list[dict] = []
    duration = max(0.0, end - start)
    for index, chunk in enumerate(chunks):
        chunk_words = chunk["words"]
        part_start = max(start, word_start(chunk_words[0], start))
        part_end = min(end, word_end(chunk_words[-1], part_start))
        if part_end <= part_start:
            part_start = start + duration * index / len(chunks)
            part_end = start + duration * (index + 1) / len(chunks)
        parts.append(
            {
                "start": part_start,
                "end": part_end,
                "zh": chunk["zh"],
                "en": chunk["en"],
                "split_source": "qwen",
            }
        )
    return parts


def display_parts(seg: dict, start: float, end: float, zh_text: str, en_text: str) -> list[dict]:
    is_music = bool(seg.get("is_music", False)) or has_music_marks(en_text) or has_music_marks(zh_text)
    is_chant = is_chant_segment(seg)
    zh_core = music_core(zh_text) if is_music else normalize_plain_text(zh_text)
    en_plain = normalize_plain_text(en_text)
    duration = max(0.0, end - start)
    words = timed_words(seg)
    semantic_parts = semantic_display_parts(seg, start, end, is_music)
    if semantic_parts:
        for part in semantic_parts:
            part["zh"] = decorate_zh(part["zh"], is_music, is_chant)
        return semantic_parts

    # Embedded Chinese and English tracks can have different cue boundaries.
    # Step 0 aligns them into one semantically complete segment, but without
    # word timestamps there is no reliable way to split that segment back into
    # matching bilingual sub-cues. Character-ratio splitting can pair a lone
    # floor number with an unrelated English sentence. Preserve the aligned
    # block intact and let ASS/player wrapping handle its visual width.
    if not words and str(seg.get("translation_origin", "")).strip().lower() == "embedded_chinese":
        return [
            {
                "start": start,
                "end": end,
                "zh": decorate_zh(zh_core, is_music, is_chant),
                "en": en_plain,
            }
        ]

    needs_split = display_needs_split(duration, zh_core, en_plain, words)
    if not words:
        # Embedded subtitle cues have no word-level timestamps. Splitting a
        # short but punctuation-heavy cue by character ratio can create a
        # burst of sub-second fragments (notably rhythmic counting) and leave
        # some fragments without Chinese. Keep short cues intact and let ASS
        # wrapping handle their visual width; only time-split genuinely long
        # cues where the duration provides enough display time.
        needs_split = duration > DISPLAY_MAX_DURATION
    if not needs_split:
        return [{"start": start, "end": end, "zh": decorate_zh(zh_core, is_music, is_chant), "en": en_plain}]

    if not words:
        en_chunks = [chunk for chunk in re.split(r"(?<=[.!?;:])\s+", en_plain) if chunk] if len(en_plain) > EN_DISPLAY_MAX_CHARS else [en_plain]
        max_chunks = max(1, int(duration // max(DISPLAY_MIN_DURATION * 1.8, 0.1)))
        en_chunks = coalesce_text_chunks(en_chunks, max_chunks)
        if len(en_chunks) <= 1:
            return [{"start": start, "end": end, "zh": decorate_zh(zh_core, is_music, is_chant), "en": en_plain}]
        zh_chunks = split_zh_by_ratios(zh_core, en_chunks)
        parts: list[dict] = []
        for index, chunk in enumerate(en_chunks):
            part_start = start + duration * index / len(en_chunks)
            part_end = start + duration * (index + 1) / len(en_chunks)
            parts.append(
                {
                    "start": part_start,
                    "end": part_end,
                    "zh": decorate_zh(zh_chunks[index] if index < len(zh_chunks) else "", is_music, is_chant),
                    "en": chunk,
                }
            )
        return parts

    word_chunks = split_words_for_display(words)
    if len(word_chunks) <= 1:
        return [{"start": start, "end": end, "zh": decorate_zh(zh_core, is_music, is_chant), "en": en_plain}]

    en_chunks = [words_text(chunk) for chunk in word_chunks]
    zh_chunks = split_zh_by_ratios(zh_core, en_chunks)
    parts: list[dict] = []
    for index, chunk in enumerate(word_chunks):
        part_start = max(start, word_start(chunk[0], start))
        part_end = min(end, word_end(chunk[-1], part_start))
        if part_end <= part_start:
            part_start = start + duration * index / len(word_chunks)
            part_end = start + duration * (index + 1) / len(word_chunks)
        parts.append(
            {
                "start": part_start,
                "end": part_end,
                "zh": decorate_zh(zh_chunks[index] if index < len(zh_chunks) else "", is_music, is_chant),
                "en": en_chunks[index],
            }
        )
    return parts


def render_ass(json_file: Path, out_ass: Path) -> int:
    with json_file.open(encoding="utf-8-sig") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    song_translations = load_song_translations()
    override_counts = apply_segment_overrides(segments, load_overrides(out_ass.parent, json_file.stem))
    if override_counts["segments"]:
        print(f"[overrides] {json_file.name}: {override_counts}")
    header = ASS_HEADER
    if str(data.get("source", "")).startswith("embedded_bilingual"):
        header = header.replace(
            "ScriptType: v4.00+",
            f"; Source: {data.get('source')}\nScriptType: v4.00+",
        )
    lines = [header]
    events: list[dict] = []
    for seg in segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        zh_text = str(seg.get("zh", "")).strip()
        en_text = str(seg.get("en_wrap", seg.get("text", ""))).strip()
        is_music = bool(seg.get("is_music", False)) or has_music_marks(en_text) or has_music_marks(zh_text)
        is_chant = is_chant_segment(seg)
        origin = str(seg.get("translation_origin", "")).strip().lower()
        ass_name = (
            "QWEN_ZH"
            if origin == "qwen"
            else (
                "EMBEDDED_CHINESE"
                if origin == "embedded_chinese"
                else ("MANUAL_ZH" if origin == "manual" else "")
            )
        )
        if is_music and not zh_text:
            zh_text = song_translations.get(song_translation_key(en_text), "")
        if not zh_text and not en_text:
            continue
        style = "BILINGUAL_CHANT" if is_chant and not is_music else ("BILINGUAL_MUSIC" if is_music else "BILINGUAL")
        for part in display_parts(seg, start, end, zh_text, en_text):
            text = make_bilingual_text(part["zh"], part["en"], style)
            if text:
                events.append(
                    {
                        "start": part["start"],
                        "end": part["end"],
                        "style": style,
                        "text": text,
                        "name": ass_name,
                    }
                )

    # A long aligned dialogue segment can contain a shorter music segment.
    # Expansion into display parts must be re-sorted before collision trimming
    # so a later dialogue part never precedes the nested music cue.
    events.sort(key=lambda event: (float(event["start"]), float(event["end"]), str(event["style"])))

    written = 0
    for index, event in enumerate(events):
        start = float(event["start"])
        end = float(event["end"])
        if end - start < MIN_DIALOGUE_DURATION:
            end = start + MIN_DIALOGUE_DURATION
        next_start = float(events[index + 1]["start"]) if index + 1 < len(events) else None
        if next_start is not None and end > next_start - ASS_EVENT_GAP:
            end = max(start + MIN_TRIMMED_DURATION, next_start - ASS_EVENT_GAP)
        text = event["text"]
        if text:
            lines.append(make_dialogue(start, end, event["style"], text, layer=0, name=event.get("name", "")))
            written += 1

    with out_ass.open("w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines) + "\n")
    return written


def main() -> None:
    args = parse_common_args("V2 STEP8: render bilingual ASS subtitles.")
    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    translated_dir = work_dir / "translated"
    videos_by_stem: dict[str, Path] = {}
    try:
        videos = selected_videos(base_dir, args.chunk, args.target_stem)
        videos_by_stem = {video.stem: video for video in videos}
        stems = set(videos_by_stem)
    except FileNotFoundError:
        # Permit exact translated JSON selection when a subtitle-only project
        # has no matching video file in the project root.
        # Preserve dotted stems such as "Movie.2026.SDH" verbatim.
        requested = {Path(value).name for value in args.target_stem if value}
        available = {path.stem for path in translated_dir.glob("*.json")}
        stems = requested & available
        if not stems:
            raise
        print(f"[subtitle-only] selected translated JSON: {', '.join(sorted(stems))}")
    json_files = filter_paths_by_stems(sorted(translated_dir.glob("*.json")), stems)
    if not json_files:
        print(f"[error] no translated files found: {translated_dir}")
        sys.exit(1)

    count = 0
    for json_file in json_files:
        video = videos_by_stem.get(json_file.stem)
        out_ass = (video.parent if video else base_dir) / f"{json_file.stem}.ass"
        print(f"[write] {out_ass.name}")
        write_status(base_dir, "V2_STEP8", out_ass.name, "render ASS")
        written = render_ass(json_file, out_ass)
        print(f"[done] {out_ass}; events={written}")
        count += 1

    write_status(base_dir, "V2_STEP8", f"{count} files", "DONE render ASS")
    print(f"v2 step8 done; generated {count} ASS file(s)")


if __name__ == "__main__":
    main()

