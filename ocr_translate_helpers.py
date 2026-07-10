import json
import math
import re
from pathlib import Path

from qwen_common import QWEN_MODEL, chat_json


OCR_BATCH_SIZE = 12
MAX_NOTES_PER_VIDEO = 12
NOTE_BATCH_SIZE = 24  # candidate items per Qwen request (keeps prompt well under ctx)
MERGE_GAP_SECONDS = 3.0
MAX_MERGED_OCR_SECONDS = 6.0
MAX_MERGE_CENTER_JUMP_PIXELS = 420.0
TEXT_SIMPLIFY_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+", re.IGNORECASE)
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
OCR_LOW_VALUE_SHORT_TEXTS_FILE = "ocr_low_value_short_texts.json"


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\\N", " ").split()).strip()


def text_key(text: str) -> str:
    return TEXT_SIMPLIFY_RE.sub("", normalize_text(text).lower())


def load_low_value_ocr_texts() -> tuple[set[str], list[str]]:
    path = Path(__file__).resolve().parent / OCR_LOW_VALUE_SHORT_TEXTS_FILE
    if not path.exists():
        return set(), []
    try:
        with path.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return set(), []
    if not isinstance(data, list):
        return set(), []
    values = [normalize_text(item) for item in data if normalize_text(item)]
    return {text_key(item) for item in values if text_key(item)}, values


OCR_LOW_VALUE_SHORT_TEXT_KEYS, OCR_LOW_VALUE_SHORT_TEXTS = load_low_value_ocr_texts()


def is_low_value_ocr_text(text: str) -> bool:
    key = text_key(text)
    return bool(key and key in OCR_LOW_VALUE_SHORT_TEXT_KEYS)


def glossary_term_matches(term: str, text: str) -> bool:
    term = normalize_text(term)
    haystack = normalize_text(text)
    if not term or not haystack:
        return False
    # Glossary keys must match as words/phrases, not raw substrings: short keys
    # like "LA", "G", or "MIT" otherwise fire inside ordinary English words.
    pattern = re.escape(term)
    pattern = re.sub(r"\\\s+", r"\\s+", pattern)
    return bool(re.search(rf"(?<![0-9A-Za-z]){pattern}(?![0-9A-Za-z])", haystack, re.IGNORECASE))


def similar_text(left: str, right: str) -> bool:
    lkey = text_key(left)
    rkey = text_key(right)
    if not lkey or not rkey:
        return False
    return lkey == rkey or lkey in rkey or rkey in lkey


def normalize_bbox(value) -> list[float]:
    if not isinstance(value, list) or len(value) < 4:
        return []
    try:
        return [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return []


def bbox_center(bbox) -> tuple[float, float] | None:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_jumped(left_bbox, right_bbox) -> bool:
    left = bbox_center(left_bbox)
    right = bbox_center(right_bbox)
    if left is None or right is None:
        return False
    return math.hypot(left[0] - right[0], left[1] - right[1]) > MAX_MERGE_CENTER_JUMP_PIXELS


def can_merge_ocr_item(group: dict, text: str, start: float, end: float, bbox: list[float]) -> bool:
    if start - float(group.get("end", 0.0)) > MERGE_GAP_SECONDS:
        return False
    if end - float(group.get("start", start)) > MAX_MERGED_OCR_SECONDS:
        return False
    if not similar_text(text, group.get("text", "")):
        return False
    if center_jumped(group.get("bbox", []), bbox):
        return False
    return True


def source_items(raw: dict) -> list[dict]:
    items = raw.get("items")
    if isinstance(items, list) and items:
        return [item for item in items if isinstance(item, dict)]
    detections = raw.get("detections")
    if isinstance(detections, list):
        return [item for item in detections if isinstance(item, dict)]
    return []


def merge_ocr(raw: dict) -> list[dict]:
    groups: list[dict] = []
    for item in sorted(source_items(raw), key=lambda row: float(row.get("start", 0.0) or 0.0)):
        text = normalize_text(item.get("text", ""))
        if not text:
            continue
        try:
            start = float(item.get("start", 0.0) or 0.0)
            end = max(float(item.get("end", start + 1.5) or start + 1.5), start + 0.5)
        except (TypeError, ValueError):
            continue
        bbox = normalize_bbox(item.get("bbox"))
        matched = None
        for group in reversed(groups[-30:]):
            if can_merge_ocr_item(group, text, start, end, bbox):
                matched = group
                break
        if matched:
            matched["end"] = max(float(matched.get("end", end)), end)
            matched["confidence"] = max(float(matched.get("confidence", 0.0)), float(item.get("confidence", 0.0) or 0.0))
            if len(text) > len(matched.get("text", "")):
                matched["text"] = text
                matched["bbox"] = bbox
        else:
            groups.append(
                {
                    "id": len(groups) + 1,
                    "start": start,
                    "end": end,
                    "text": text,
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                    "bbox": bbox,
                }
            )
    return groups


# On-screen text that marks the end-credit legal/disclaimer/vanity-card block.
# Once seen, a wide window covers the whole block (these run as one dense crawl).
LEGAL_DISCLAIMER_RE = re.compile(
    r"copyright|unauthorized|all rights reserved|motion picture|warner bros|"
    r"protected under|throughout the world|similarity to|actual persons|"
    r"production #|soundtrack|criminal prosecution|warner bros\.?discovery|"
    r"chuck lorre productions|country of first",
    re.IGNORECASE,
)


def build_credit_windows(items: list[dict]) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for item in items:
        text = normalize_text(item.get("text", ""))
        start = float(item.get("start", 0.0) or 0.0)
        end = float(item.get("end", start + 4.0) or start + 4.0)
        low = text.lower()
        # Subtitle-group / ripper watermarks: small window.
        if any(word in low for word in ("subtitle", "subtitles", "caption", "sync", "rip", "encoded", "www.", ".com")):
            windows.append((start - 1.0, end + 1.0))
        # End-credit legal disclaimer / vanity card: a single dense block, so use a
        # wide window to also catch the vanity-card essay text and the company tag.
        if LEGAL_DISCLAIMER_RE.search(text):
            windows.append((start - 20.0, end + 20.0))
    return windows


def in_windows(item: dict, windows: list[tuple[float, float]]) -> bool:
    start = float(item.get("start", 0.0) or 0.0)
    end = float(item.get("end", start) or start)
    return any(end >= left and start <= right for left, right in windows)


def _text_similarity(a: str, b: str) -> float:
    """Character-level similarity for short OCR strings.

    OCR re-reads the same on-screen word several ways ("Auto&Tire" vs
    "Awto&Tired"); character ratio catches these token-concatenated variants
    that word-split overlap would miss.
    """
    from difflib import SequenceMatcher
    a, b = a.lower(), b.lower()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def filter_ocr_groups(items: list[dict], credit_windows: list[tuple[float, float]]) -> list[dict]:
    kept: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for item in items:
        text = normalize_text(item.get("text", ""))
        if not text or len(text_key(text)) < 2:
            continue
        if is_low_value_ocr_text(text):
            continue
        if in_windows(item, credit_windows):
            continue
        minute = int(float(item.get("start", 0.0) or 0.0) // 60)
        key = (minute, text_key(text))
        if key in seen:
            continue
        # Fuzzy de-dup: OCR often re-reads the same sign several ways in one
        # window ("Auto&Tire" / "Awto&Tired" / "MeAllisterd"). Drop a new item if
        # it closely resembles an already-kept item in the same minute, keeping
        # the longer (cleaner) spelling. This removes mangled duplicates before
        # they reach Qwen, so only one translation is overlaid.
        duplicate_of = None
        for kept_item in kept:
            if int(float(kept_item.get("start", 0.0) or 0.0) // 60) != minute:
                continue
            if _text_similarity(text, normalize_text(kept_item.get("text", ""))) >= 0.6:
                duplicate_of = kept_item
                break
        if duplicate_of is not None:
            # Replace the kept item if the new spelling is cleaner (longer text).
            if len(text) > len(normalize_text(duplicate_of.get("text", ""))):
                kept[kept.index(duplicate_of)] = item
            continue
        seen.add(key)
        kept.append(item)
    for index, item in enumerate(kept, start=1):
        item["id"] = index
    return kept


OCR_SYSTEM_PROMPT = """You classify and translate on-screen OCR text for TV subtitles.
Return strict JSON only:
{"items": [{"id": 1, "keep": true, "zh": "简体中文"}]}
Classification (set "keep"):
- keep=false for TV/film credits: actor/staff names, job titles (Starring, Created by,
  Directed by, Executive/Co/Associate Producer, Casting, Director of Photography, Editor,
  Production Designer, Gaffer, Grip, Sound, Costume, etc.), and OCR-mangled variants of them.
- keep=false for legal disclaimers/copyright/vanity cards (Copyright, Warner Bros,
  Unauthorized, This motion picture, "all rights reserved", Chuck Lorre's essay text).
- keep=false for OCR garbage: single letters/digits, random consonant clusters with no
  vowel structure (e.g. "MNNNC", "TIRESTATFII", "SUTOSTE"), fragments of a word that are
  themselves unreadable, or stray punctuation. When two items are clearly the same source
  word read two ways (e.g. "Auto&Tire" vs "Awto&Tired"), keep only the cleaner one.
- keep=false for proper-name credits even with suffixes: "NAME, CSA" / "NAME. CSA" are
  casting directors; any all-caps person name appearing among job titles is staff.
- keep=false if the useful translation would be only a known low-value standalone
  fragment such as a direction, partial location word, or isolated sign shard.
- keep=true for in-story screen text worth translating: shop/business signs, brand names,
  location/place cards, on-screen titles, advertisements, price/phone/address lines.
For keep=true items, "zh" is the Simplified Chinese translation (short, for overlay).
For keep=false items, set "zh" to "" (empty).
Rules:
1. Use Simplified Chinese only for kept translations.
2. Keep names and brands natural.
3. If kept text is already Chinese, return it unchanged."""


def translate_ocr_batch(client, batch: list[dict]) -> dict[int, tuple[bool, str]]:
    """Classify + translate one OCR batch.

    Returns ``{id: (keep, zh)}``. ``keep`` is False for credits/disclaimers/garbage
    (which must be dropped before overlay); ``zh`` is the translation when kept.
    """
    if not batch:
        return {}
    payload = [{"id": int(item["id"]), "text": item.get("text", "")} for item in batch]
    fallback = {"items": [{"id": row["id"], "keep": True, "zh": row["text"]} for row in payload]}
    low_value_examples = OCR_LOW_VALUE_SHORT_TEXTS[:40]
    low_value_hint = ""
    if low_value_examples:
        low_value_hint = (
            "\nKnown low-value OCR fragments from prior polish. If an item or its translation is only one of these, set keep=false:\n"
            + json.dumps(low_value_examples, ensure_ascii=False)
        )
    result = chat_json(
        client,
        OCR_SYSTEM_PROMPT,
        "Classify and translate these OCR text items:\n" + json.dumps(payload, ensure_ascii=False) + low_value_hint,
        fallback,
        max_tokens=900,
    )
    translations: dict[int, tuple[bool, str]] = {}
    for item in result.get("items", []) if isinstance(result, dict) else []:
        try:
            item_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        keep = bool(item.get("keep", True))
        zh = normalize_text(item.get("zh", ""))
        if keep and is_low_value_ocr_text(zh):
            keep = False
            zh = ""
        translations[item_id] = (keep, zh)
    return translations


def load_glossary(base_dir: Path) -> dict:
    path = base_dir / "subtitle_glossary.json"
    if not path.exists():
        return {"terms": []}
    try:
        with path.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {"terms": []}
    except json.JSONDecodeError:
        return {"terms": []}


def save_glossary(base_dir: Path, glossary: dict) -> None:
    path = base_dir / "subtitle_glossary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(glossary if isinstance(glossary, dict) else {"terms": []}, handle, ensure_ascii=False, indent=2)


def _is_noise_term(term: str) -> bool:
    """Reject terms that should never be a glossary entry.

    - Chinese-only terms (glossary tracks EN references; zh echoes are noise)
    - Main-character names that get over-matched (exact OR as part of a phrase,
      e.g. "Mandy McAllister" — she is the show's title character, no note needed)
    - Credit/disclaimer tokens (also blocked as notes, kept out of glossary too)
    """
    norm = normalize_text(term)
    if not norm:
        return True
    if HAN_RE.search(norm):  # e.g. "塔可钟" learned by mistake
        return True
    lower = norm.lower()
    if lower in NOISE_TERMS:
        return True
    # Reject multi-word terms built around a main-character name.
    if any(lower == name or name in lower.split() for name in NOISE_TERMS):
        return True
    if SDH_CREDIT_RE.search(norm):  # credit tokens, but NOT sponsor fragments
        return True
    return False


# Single-word main-character / show-title names that are not worth annotating
# (over-matched in every episode).
NOISE_TERMS = {
    "georgie", "mandy", "mcallister", "cooper", "sheldon",
    "meemaw", "connie", "dale", "audra", "cecee", "cece", "ruben",
}


def update_glossary(glossary: dict, notes: list[dict]) -> None:
    """Merge new terms confirmed by Qwen into the glossary for cross-episode reuse.

    Glossary keys are kept by normalized term text so different zh phrasings of
    the same reference do not create duplicates. Existing entries are kept
    (human-curated glossary wins). Noise/credit terms are never learned.
    """
    if not isinstance(glossary, dict):
        return
    terms = glossary.setdefault("terms", [])
    existing = {text_key(t.get("term") or t.get("text")) for t in terms if isinstance(t, dict)}
    for note in notes:
        term = normalize_text(note.get("term") or "")
        zh = normalize_text(note.get("zh") or note.get("note"))
        if not term or not zh:
            continue
        if _is_noise_term(term):
            continue
        key = text_key(term)
        if key and key not in existing:
            terms.append({"term": term, "zh": zh, "source": "qwen", "reviewed": False})
            existing.add(key)


def collect_dialogue_items(translated_data: dict) -> list[dict]:
    items: list[dict] = []
    for index, seg in enumerate(translated_data.get("segments", []), start=1):
        text = normalize_text(seg.get("text") or seg.get("en_wrap") or "")
        zh = normalize_text(seg.get("zh", ""))
        if not text:
            continue
        try:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start + 1.0) or start + 1.0)
        except (TypeError, ValueError):
            start, end = 0.0, 1.0
        items.append({"id": index, "source": "dialogue", "start": start, "end": end, "text": text, "translation": zh})
    return items


NOTE_SYSTEM_PROMPT = """You find helpful cultural/encyclopedic notes for Chinese TV subtitles.
Return strict JSON only:
{"notes": [{"start": 12.3, "end": 16.0, "term": "the referenced name/phrase", "zh": "36到60个汉字左右的一句中文背景注解"}]}
Rules:
1. ONLY add a note when it genuinely helps a Chinese viewer — e.g. a real person (celebrity/historical), a place/brand/product, a work (film/song/book/show), a cultural custom, slang, or a Latin/French/etc. phrase.
2. EXCLUDE ordinary dialogue, sentence fragments, and capitalized words that are NOT proper references. "To Mandy", "The Georgie", "And Mandy", "Cause Daddy" are NOT references — skip them.
3. The note must explain both what/who it is and a tiny bit of useful cultural background, in one compact sentence or two short clauses. Do not just echo the term or say "专有名词/人名/品牌".
4. Keep it readable as an on-screen note: informative but not encyclopedic, no brackets, no markdown, no extra line breaks.
5. Simplified Chinese only. If nothing in the batch deserves a note, return {"notes": []}."""


def glossary_notes(glossary: dict, candidates: list[dict]) -> list[dict]:
    notes: list[dict] = []
    terms = glossary.get("terms", []) if isinstance(glossary, dict) else []
    for term in terms:
        if not isinstance(term, dict):
            continue
        if term.get("reviewed") is False:
            continue
        key = normalize_text(term.get("term") or term.get("text"))
        zh = normalize_text(term.get("zh") or term.get("note"))
        if not key or not zh:
            continue
        for item in candidates:
            if is_sdh_credit_text(normalize_text(item.get("text", ""))):
                continue
            if glossary_term_matches(key, item.get("text", "")):
                start = float(item.get("start", 0.0) or 0.0)
                end = float(item.get("end", start + 2.0) or start + 2.0)
                notes.append({"start": start, "end": adaptive_note_end(start, end, zh), "term": key, "zh": zh, "_src": "glossary"})
                break
    return notes


# Pre-filter signals for a possible cultural reference.
PROPER_NAME_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3}\b")
# All-caps acronyms (WGBH, NHL, FBI, MTV ...), at least 2 letters.
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
# Common Latin/French/etc. phrases worth a note.
FOREIGN_PHRASE_RE = re.compile(r"\b(?:et al|c'est la vie|cara mia|in vitro|quid pro quo|status quo|magnum opus)\b", re.IGNORECASE)
# SDH/broadcast captioner & sponsor disclaimer text that must NEVER be annotated
# (it is end-credit boilerplate, not story dialogue). Case-insensitive.
SDH_CREDIT_RE = re.compile(
    r"captioned by|captioning sponsored|sponsored by|media access group|"
    r"access\.wgbh\.org|wgbh\b|closed caption|subtitled by|encoded by|sync by|"
    r"ripper|\bwww\.|\.com\b|viewers like you|contributions to your",
    re.IGNORECASE,
)
# Sponsor voiceover fragment lines from the credits. These are short, brand-only
# continuations of "Captioning sponsored by ...": e.g. "and TOYOTA." We match
# the literal "and <Brand>." shape to avoid touching real dialogue (characters
# are addressed by single names like "Georgie." which must NOT be filtered).
SPONSOR_FRAGMENT_RE = re.compile(r"^\s*and\s+[A-Z][A-Za-z0-9 &.'+-]{1,30}\.?\s*$")


def is_sdh_credit_text(text: str) -> bool:
    """True for SDH/broadcast disclaimer boilerplate (captioner, sponsor URL)."""
    norm = normalize_text(text)
    if not norm:
        return True
    if SDH_CREDIT_RE.search(norm):
        return True
    # Only the literal "and <Brand>." sponsor-credit continuation is filtered
    # here; bare brand names in real dialogue are left for Qwen to judge.
    if SPONSOR_FRAGMENT_RE.match(norm):
        return True
    return False


def _has_reference(text: str) -> bool:
    """True if the line carries any plausible reference signal.

    Uses finditer so a line qualifies if ANY match clears the threshold — a
    single subtitle can legitimately contain several references, and we want
    Qwen to see the whole line and decide.
    """
    if any(len(m.group(0)) >= 8 for m in PROPER_NAME_RE.finditer(text)):
        return True
    if ACRONYM_RE.search(text):
        return True
    if FOREIGN_PHRASE_RE.search(text):
        return True
    return False


def note_candidate_items(candidates: list[dict]) -> list[dict]:
    """Pre-filter candidate items that may contain a cultural reference.

    This is a cheap (no-Qwen) filter that returns the *items* worth sending to
    Qwen — it never fabricates notes. Single-word proper nouns that are already
    known are also caught by the glossary (glossary_notes), so this filter
    focuses on multi-word names, acronyms, and foreign phrases.
    """
    kept: list[dict] = []
    for item in candidates:
        text = normalize_text(item.get("text", ""))
        if not text or HAN_RE.search(text):
            continue
        if is_sdh_credit_text(text):
            continue
        if _has_reference(text):
            kept.append(item)
    return kept


def find_notes(client, glossary: dict, candidates: list[dict]) -> list[dict]:
    notes = glossary_notes(glossary, candidates)
    # Pre-filter to items that plausibly contain a reference, so the whole
    # episode never overflows the context window (a full episode is ~10k tokens).
    candidate_items = note_candidate_items(candidates)
    for offset in range(0, len(candidate_items), NOTE_BATCH_SIZE):
        batch = candidate_items[offset : offset + NOTE_BATCH_SIZE]
        compact = [
            {
                "start": round(float(item.get("start", 0.0) or 0.0), 2),
                "end": round(float(item.get("end", 0.0) or 0.0), 2),
                "text": normalize_text(item.get("text", "")),
                "translation": normalize_text(item.get("translation", "")),
                "source": item.get("source", "dialogue"),
            }
            for item in batch
        ]
        if not compact:
            continue
        result = chat_json(
            client,
            NOTE_SYSTEM_PROMPT,
            "Find useful notes in these subtitle/OCR items:\n" + json.dumps(compact, ensure_ascii=False),
            {"notes": []},
            max_tokens=1200,
        )
        for note in result.get("notes", []) if isinstance(result, dict) else []:
            zh = normalize_text(note.get("zh") or note.get("note"))
            term = normalize_text(note.get("term") or note.get("text"))
            if not zh:
                continue
            try:
                start = float(note.get("start", 0.0) or 0.0)
                end = float(note.get("end", start + 4.0) or start + 4.0)
            except (TypeError, ValueError):
                start, end = 0.0, 4.0
            notes.append({"start": start, "end": adaptive_note_end(start, end, zh), "term": term, "zh": zh, "_src": "qwen"})
    # Learn confirmed references for cross-episode reuse (glossary wins on
    # pre-existing keys, so human-curated entries are never overwritten).
    update_glossary(glossary, [n for n in notes if n.get("_src") == "qwen"])
    return notes


def note_zh_char_count(zh: str) -> int:
    """Count Chinese characters + latin words to estimate reading load."""
    zh = normalize_text(zh)
    hans = len(HAN_RE.findall(zh))
    latin_words = len(re.findall(r"[A-Za-z0-9]+", zh))
    return hans + latin_words


def adaptive_note_end(start: float, end: float, zh: str) -> float:
    """Ensure a note stays on screen long enough to read (~5 chars/sec).

    Floor 2.5s, then widen by reading load; never shrink Qwen's own end time.
    """
    duration = max(2.5, note_zh_char_count(zh) / 5.0)
    return max(float(end), float(start) + duration)


def dedupe_notes(notes: list[dict]) -> list[dict]:
    """Keep one note per normalized *term* — its earliest appearance.

    Deduping by term (not by the zh wording) means the same reference is
    annotated once per video at first mention, dropping repeats that differ
    only in start bucket (e.g. "San Antonio" at 01:16, 03:37, 12:18). Within a
    short 20s window we also collapse near-duplicate zh notes for different
    terms, to avoid stacking two notes on the same cue.
    """
    seen_terms: set[str] = set()
    seen_zh_window: list[tuple[float, str]] = []
    deduped: list[dict] = []
    for note in sorted(notes, key=lambda item: float(item.get("start", 0.0) or 0.0)):
        zh = normalize_text(note.get("zh") or note.get("note"))
        if not zh:
            continue
        term = normalize_text(note.get("term", ""))
        # Drop main-character / noise terms before they ever reach the output.
        if _is_noise_term(term or zh):
            continue
        # Normalize plural vs singular so "Dallas Cowboy"/"Dallas Cowboys" or
        # "The Canadien"/"The Canadiens" collapse to one note.
        term_key = text_key(term or zh)
        if term_key:
            term_key = re.sub(r"(?:s|es)$", "", term_key)
        start = float(note.get("start", 0.0) or 0.0)
        # Collapse same term repeats across the whole video.
        if term_key and term_key in seen_terms:
            continue
        # Collapse near-identical zh notes inside a 20s sliding window.
        zh_k = text_key(zh)
        if zh_k and any(abs(start - s) <= 20.0 and text_key(w) == zh_k for s, w in seen_zh_window):
            continue
        if term_key:
            seen_terms.add(term_key)
        seen_zh_window.append((start, zh))
        clean = dict(note)
        clean["zh"] = zh
        clean["end"] = adaptive_note_end(float(clean.get("start", 0.0) or 0.0), float(clean.get("end", 0.0) or 0.0), zh)
        clean.pop("_src", None)  # internal only — do not leak into notes.json / ASS
        deduped.append(clean)
    return deduped
