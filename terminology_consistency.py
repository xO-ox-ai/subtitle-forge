import hashlib
import json
import re
from pathlib import Path


SUBTITLE_TERMINOLOGY_FILE = "subtitle_terminology.json"
FILE_TERMINOLOGY_DIR = "terminology"
NAME_TERMINOLOGY_PROMPT_VERSION = "name-terminology-20260727-v4"

_NAME_TOKEN_RE = re.compile(r"(?<![A-Za-zÀ-ÖØ-öø-ÿ])(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ]*|[A-Z]{2,})(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)?")
_SPACE_RE = re.compile(r"\s+")
_FILENAME_SEPARATOR_RE = re.compile(r"[._-]+")
_CONTRACTION_SUFFIX_RE = re.compile(r"(?:n['’]t|['’](?:m|re|ll|ve|d))$", re.IGNORECASE)
_POSSESSIVE_SUFFIX_RE = re.compile(r"['’]s$", re.IGNORECASE)
_COMMON_CAPITALIZED_WORDS = {
    "A",
    "About",
    "After",
    "All",
    "Also",
    "Am",
    "An",
    "And",
    "Another",
    "Any",
    "Are",
    "As",
    "At",
    "Because",
    "Before",
    "But",
    "By",
    "Can",
    "Come",
    "Could",
    "Dad",
    "Did",
    "Do",
    "Does",
    "Don",
    "Even",
    "For",
    "Floor",
    "From",
    "Get",
    "Give",
    "Go",
    "Good",
    "Got",
    "Had",
    "Has",
    "Have",
    "He",
    "Hello",
    "Her",
    "Here",
    "Hey",
    "His",
    "How",
    "I",
    "If",
    "In",
    "Is",
    "It",
    "Just",
    "Let",
    "Li'l",
    "Listen",
    "Look",
    "Me",
    "Mom",
    "My",
    "Never",
    "No",
    "Not",
    "Now",
    "Of",
    "Oh",
    "Okay",
    "On",
    "Or",
    "Our",
    "Please",
    "Right",
    "See",
    "She",
    "So",
    "Some",
    "Take",
    "Tell",
    "Thank",
    "That",
    "The",
    "Their",
    "Them",
    "Then",
    "There",
    "These",
    "They",
    "This",
    "Those",
    "To",
    "Today",
    "Tomorrow",
    "Tonight",
    "Too",
    "Up",
    "Us",
    "Very",
    "Was",
    "We",
    "Well",
    "Were",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
    "Will",
    "With",
    "Would",
    "Yes",
    "You",
    "Your",
}
_AUTO_NAME_REVIEW_BLOCKLIST = {
    "Twenty Questions",
}


def normalize_inline_text(value) -> str:
    return _SPACE_RE.sub(" ", str(value or "").replace(r"\N", " ")).strip()


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def _word_pattern(value: str) -> str:
    escaped = re.escape(normalize_inline_text(value))
    return re.sub(r"\\\s+", r"\\s+", escaped)


def _term_candidates(item: dict) -> list[str]:
    source = normalize_inline_text(item.get("source_en") or item.get("source") or item.get("term"))
    raw_aliases = item.get("aliases") or []
    aliases = raw_aliases if isinstance(raw_aliases, list) else [raw_aliases]
    return [value for value in [source, *(normalize_inline_text(alias) for alias in aliases)] if value]


def terminology_entry_matches_source(item: dict, source_text: str) -> bool:
    flags = 0 if item.get("case_sensitive") is True else re.IGNORECASE
    text = normalize_inline_text(source_text)
    for candidate in _term_candidates(item):
        if re.search(rf"(?<![0-9A-Za-z]){_word_pattern(candidate)}(?![0-9A-Za-z])", text, flags):
            return True
    return False


def terminology_scope_matches(item: dict, segment_kind: str) -> bool:
    raw = item.get("scope") or item.get("style_scope") or ""
    values = raw if isinstance(raw, list) else re.split(r"[,/| ]+", str(raw))
    scopes = {str(value).strip().lower() for value in values if str(value).strip()}
    if not scopes or "all" in scopes:
        return True
    aliases = {
        "dialogue": {"dialogue", "spoken", "line"},
        "lyric": {"lyric", "lyrics", "music", "song"},
        "chant": {"chant", "incantation", "spell"},
        "ocr": {"ocr", "screen"},
        "note": {"note", "cultural_note"},
    }
    return bool(scopes & aliases.get(segment_kind, {segment_kind}))


def series_matches_source(series: str, source_name: str) -> bool:
    normalized_series = normalize_inline_text(_FILENAME_SEPARATOR_RE.sub(" ", series))
    normalized_name = normalize_inline_text(_FILENAME_SEPARATOR_RE.sub(" ", source_name))
    if not normalized_series:
        return True
    return bool(
        re.search(
            rf"(?<![0-9A-Za-z]){_word_pattern(normalized_series)}(?![0-9A-Za-z])",
            normalized_name,
            re.IGNORECASE,
        )
    )


def valid_terminology_entries(entries) -> list[dict]:
    result: list[dict] = []
    for item in entries if isinstance(entries, list) else []:
        if not isinstance(item, dict) or item.get("reviewed") is False:
            continue
        source = normalize_inline_text(item.get("source_en") or item.get("source") or item.get("term"))
        preferred = normalize_inline_text(item.get("preferred_zh"))
        is_auto_review = str(item.get("source") or "").strip() in {
            "qwen-file-name-review",
            "polish-file-name-review",
        }
        if is_auto_review and (
            source in _COMMON_CAPITALIZED_WORDS
            or source in _AUTO_NAME_REVIEW_BLOCKLIST
        ):
            continue
        if source and preferred:
            normalized = dict(item)
            if is_auto_review:
                raw_aliases = normalized.get("zh_aliases") or []
                aliases = raw_aliases if isinstance(raw_aliases, list) else [raw_aliases]
                safe_aliases = [
                    normalize_inline_text(alias)
                    for alias in aliases
                    if normalize_inline_text(alias)
                    and preferred not in normalize_inline_text(alias)
                    and normalize_inline_text(alias) not in preferred
                ]
                if safe_aliases:
                    normalized["zh_aliases"] = safe_aliases
                else:
                    normalized.pop("zh_aliases", None)
            result.append(normalized)
    return result


def merge_terminology_entries(*entry_groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for group in entry_groups:
        for item in valid_terminology_entries(group):
            source = normalize_inline_text(item.get("source_en") or item.get("source") or item.get("term"))
            key = source.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def load_project_terminology(base_dir: Path, source_name: str) -> list[dict]:
    data = _load_json(Path(base_dir) / SUBTITLE_TERMINOLOGY_FILE, {})
    entries = data.get("entries") if isinstance(data, dict) else []
    return [
        item
        for item in valid_terminology_entries(entries)
        if series_matches_source(str(item.get("series") or ""), source_name)
    ]


def file_terminology_path(work_dir: Path, source_stem: str) -> Path:
    return Path(work_dir) / FILE_TERMINOLOGY_DIR / f"{source_stem}.json"


def load_file_terminology(work_dir: Path, source_stem: str) -> list[dict]:
    data = _load_json(file_terminology_path(work_dir, source_stem), {})
    entries = data.get("entries") if isinstance(data, dict) else []
    return valid_terminology_entries(entries)


def _target_contains_known_zh_alias(item: dict, target_text: str) -> bool:
    raw_aliases = item.get("zh_aliases") or []
    aliases = raw_aliases if isinstance(raw_aliases, list) else [raw_aliases]
    text = str(target_text or "")
    return any(
        normalize_inline_text(alias) in text
        for alias in aliases
        if len(normalize_inline_text(alias)) >= 2
    )


def matching_terminology(
    entries: list[dict],
    source_text: str,
    segment_kind: str,
    target_text: str = "",
) -> list[dict]:
    normalized_source = normalize_inline_text(source_text)
    return [
        item
        for item in valid_terminology_entries(entries)
        if terminology_scope_matches(item, segment_kind)
        and (
            terminology_entry_matches_source(item, source_text)
            # Target-only matching supports OCR and other overlays that have no
            # English source.  When English exists it must identify the entry;
            # otherwise overlapping Chinese names such as Greg/格雷格 and
            # Gregers/格雷格斯 can rewrite the wrong character.
            or (not normalized_source and _target_contains_known_zh_alias(item, target_text))
        )
    ]


def _replace_english_variant(text: str, variant: str, preferred: str, case_sensitive: bool) -> str:
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.sub(
        rf"(?<![0-9A-Za-z]){_word_pattern(variant)}(?![0-9A-Za-z])",
        lambda _match: preferred,
        text,
        flags=flags,
    )


def _canonicalize_known_variants(text: str, entries: list[dict]) -> str:
    result = str(text or "")
    for item in entries:
        preferred = normalize_inline_text(item.get("preferred_zh"))
        if not preferred:
            continue
        raw_zh_aliases = item.get("zh_aliases") or []
        zh_aliases = raw_zh_aliases if isinstance(raw_zh_aliases, list) else [raw_zh_aliases]
        for alias in sorted(
            {normalize_inline_text(alias) for alias in zh_aliases if normalize_inline_text(alias)} - {preferred},
            key=len,
            reverse=True,
        ):
            if alias in preferred and preferred in result:
                continue
            result = result.replace(alias, preferred)
        for source_variant in sorted(set(_term_candidates(item)), key=len, reverse=True):
            result = _replace_english_variant(
                result,
                source_variant,
                preferred,
                item.get("case_sensitive") is True,
            )
    return result


def canonicalize_known_variants(text: str, entries: list[dict]) -> str:
    """Replace reviewed terminology variants without changing surrounding formatting."""
    return _canonicalize_known_variants(text, valid_terminology_entries(entries))


def enforce_terminology(
    text: str,
    source_text: str,
    entries: list[dict],
    *,
    segment_kind: str = "dialogue",
    fallback_text: str = "",
) -> tuple[str, list[str]]:
    matched = matching_terminology(entries, source_text, segment_kind, text)
    if not matched:
        return str(text or ""), []

    result = _canonicalize_known_variants(text, matched)
    # A translated subtitle may naturally omit an English vocative, and a longer
    # nickname can legitimately contain a surname whose standalone rendering is
    # different (for example Baby Clade -> 索奇宝宝, Clade -> 克莱德).  Only require
    # a name to survive polishing when it was actually present in the pre-polish
    # Chinese.  Known Chinese aliases and leaked English forms are still replaced
    # deterministically in every target.
    required: list[str] = []
    fallback = ""
    if fallback_text:
        fallback = _canonicalize_known_variants(fallback_text, matched)
        required = [
            normalize_inline_text(item.get("preferred_zh"))
            for item in matched
            if normalize_inline_text(item.get("preferred_zh")) in fallback
        ]
        if any(preferred not in result for preferred in required):
            result = fallback
    missing = [preferred for preferred in required if preferred not in result]
    return result, missing


def extract_recurring_name_candidates(
    items: list[dict],
    *,
    min_occurrences: int = 2,
    max_candidates: int = 80,
    max_examples: int = 6,
) -> list[dict]:
    occurrences: dict[str, list[dict]] = {}
    display_forms: dict[str, str] = {}
    for item in items:
        source = normalize_inline_text(item.get("en") or item.get("text"))
        if not source:
            continue
        chinese = normalize_inline_text(item.get("zh"))
        seen_in_item: set[str] = set()
        tokens = [match.group(0) for match in _NAME_TOKEN_RE.finditer(source)]
        candidates = list(tokens)
        for length in (2, 3, 4):
            for offset in range(0, len(tokens) - length + 1):
                phrase = " ".join(tokens[offset : offset + length])
                if source.find(phrase) >= 0:
                    candidates.append(phrase)
        for candidate in candidates:
            normalized_words: list[str] = []
            skip_candidate = False
            for word in candidate.split():
                if _CONTRACTION_SUFFIX_RE.search(word):
                    skip_candidate = True
                    break
                normalized_words.append(_POSSESSIVE_SUFFIX_RE.sub("", word))
            if skip_candidate:
                continue
            candidate = " ".join(normalized_words)
            words = candidate.split()
            if len(words) == 1 and candidate in _COMMON_CAPITALIZED_WORDS:
                continue
            if all(word in _COMMON_CAPITALIZED_WORDS for word in words):
                continue
            key = candidate.casefold()
            if key in seen_in_item:
                continue
            seen_in_item.add(key)
            display_forms.setdefault(key, candidate)
            occurrences.setdefault(key, []).append({"en": source, "zh": chinese})

    ranked: list[dict] = []
    for key, examples in occurrences.items():
        if len(examples) < max(min_occurrences, 1):
            continue
        example_limit = max(max_examples, 1)
        if len(examples) <= example_limit:
            selected_examples = examples
        elif example_limit == 1:
            selected_examples = [examples[0]]
        else:
            # Sample the complete subtitle timeline instead of only the opening
            # occurrences, where one late inconsistent rendering would be hidden.
            indexes = {
                round(position * (len(examples) - 1) / (example_limit - 1))
                for position in range(example_limit)
            }
            selected_examples = [examples[index] for index in sorted(indexes)]
        ranked.append(
            {
                "source_en": display_forms[key],
                "occurrences": len(examples),
                "examples": selected_examples,
            }
        )
    ranked.sort(key=lambda item: (-int(item["occurrences"]), -len(item["source_en"].split()), item["source_en"].casefold()))
    return ranked[: max(max_candidates, 1)]


def name_terminology_system_prompt() -> str:
    return (
        "You build a binding Simplified Chinese name glossary for one movie or TV episode. "
        "Return strict JSON only: "
        "{\"entries\":[{\"source_en\":\"...\",\"aliases\":[\"...\"],"
        "\"preferred_zh\":\"...\",\"zh_aliases\":[\"...\"]}]}. "
        "Select only recurring person or character names, nicknames, and clearly person-like aliases. "
        "source_en must exactly match one supplied candidate. "
        "Keep full names, first names, surnames, and nicknames as separate entries when their literal Chinese "
        "renderings differ: for example Max Manus -> 马克斯·曼努斯, Max -> 马克斯, Manus -> 曼努斯. "
        "Use aliases only for spelling/OCR variants that must produce exactly the same preferred_zh; never use "
        "aliases to expand a short form into a full name. "
        "Choose one concise, natural Chinese rendering and use existing Chinese evidence when it is reliable. "
        "zh_aliases may contain only other Chinese renderings actually visible in the supplied examples. "
        "Do not include ordinary sentence-initial words, pronouns, forms of address, places, organizations, "
        "brands, titles, or uncertain candidates. Every returned preferred_zh will be enforced literally, "
        "so omit anything that is not a high-confidence person name. Simplified Chinese only."
    )


def name_terminology_payload(source_name: str, candidates: list[dict]) -> dict:
    return {
        "prompt_version": NAME_TERMINOLOGY_PROMPT_VERSION,
        "source_name": source_name,
        "candidates": candidates,
    }


def normalize_reviewed_name_entries(
    raw_entries,
    candidates: list[dict],
    *,
    review_source: str,
) -> list[dict]:
    candidate_names = {item["source_en"].casefold(): item["source_en"] for item in candidates}
    entries: list[dict] = []
    seen: set[str] = set()
    for item in raw_entries if isinstance(raw_entries, list) else []:
        if not isinstance(item, dict):
            continue
        requested_source = normalize_inline_text(item.get("source_en", ""))
        source = candidate_names.get(requested_source.casefold())
        preferred = normalize_inline_text(item.get("preferred_zh", ""))
        if not source or not preferred or not re.search(r"[\u4e00-\u9fff]", preferred):
            continue
        key = source.casefold()
        if key in seen:
            continue
        seen.add(key)
        raw_zh_aliases = item.get("zh_aliases") or []
        zh_aliases = raw_zh_aliases if isinstance(raw_zh_aliases, list) else [raw_zh_aliases]
        normalized_zh_aliases = sorted(
            {
                normalize_inline_text(alias)
                for alias in zh_aliases
                if normalize_inline_text(alias)
                and normalize_inline_text(alias) != preferred
                and preferred not in normalize_inline_text(alias)
                and normalize_inline_text(alias) not in preferred
                and re.search(r"[\u4e00-\u9fff]", normalize_inline_text(alias))
            }
        )
        entry = {
            "source_en": source,
            "preferred_zh": preferred,
            "case_sensitive": True,
            "scope": "all",
            "reviewed": True,
            "source": review_source,
        }
        raw_source_aliases = item.get("aliases") or []
        source_aliases = raw_source_aliases if isinstance(raw_source_aliases, list) else [raw_source_aliases]
        normalized_source_aliases = sorted(
            {
                candidate_names[normalize_inline_text(alias).casefold()]
                for alias in source_aliases
                if normalize_inline_text(alias).casefold() in candidate_names
                and normalize_inline_text(alias).casefold() != source.casefold()
            },
            key=lambda value: (-len(value.split()), value.casefold()),
        )
        if normalized_source_aliases:
            entry["aliases"] = normalized_source_aliases
        if normalized_zh_aliases:
            entry["zh_aliases"] = normalized_zh_aliases
        entries.append(entry)
    return entries


def terminology_source_digest(source_name: str, candidates: list[dict], _model: str = "") -> str:
    payload = {
        "prompt_version": NAME_TERMINOLOGY_PROMPT_VERSION,
        "source_name": source_name,
        "candidates": candidates,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
