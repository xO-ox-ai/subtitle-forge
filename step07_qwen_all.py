import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

from common import (
    StatusHeartbeat,
    add_common_args,
    filter_paths_by_stems,
    load_json,
    runtime_file_for,
    save_json,
    selected_videos,
    status_preview,
    work_dir_for,
    write_status,
)
import qwen_common
from override_utils import apply_segment_overrides, load_overrides

try:
    from ocr_translate_helpers import (
        MAX_NOTES_PER_VIDEO,
        OCR_BATCH_SIZE,
        build_credit_windows,
        collect_dialogue_items,
        dedupe_notes,
        filter_ocr_groups,
        find_notes,
        load_glossary,
        merge_ocr,
        save_glossary,
        translate_ocr_batch,
    )
    OCR_HELPERS_AVAILABLE = True
except ModuleNotFoundError:
    MAX_NOTES_PER_VIDEO = 12
    OCR_BATCH_SIZE = 12
    OCR_HELPERS_AVAILABLE = False


MUSIC_SYMBOL = "\u266a"
FIXED_RECAP_ZH = "《小谢尔曼》剧情回顾"
BREAK_CHARS = ",.!?;: " + "\uff0c\u3002\uff01\uff1f\uff1b\uff1a"
DISPLAY_MAX_DURATION = 5.8
EN_DISPLAY_MAX_CHARS = 78
DISPLAY_MIN_WORDS = 8
DEFAULT_DIALOGUE_BATCH_SIZE = 4
DEFAULT_DIALOGUE_CONTEXT_SIZE = 4
MOJIBAKE_MUSIC_MARK_RE = re.compile(
    r"(?<![0-9A-Za-z])(?:[jJ][“”\"']|[jJ]n(?=\s))|[jJ][“”\"'](?=\s|$)"
)
LEADING_SPEAKER_TAG_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9 .'\-]{1,40})(?:\s*\((?:on phone|o\.s\.|v\.o\.|over phone)\))?\s*:\s*"
)
EN_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)?")
HINT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
TRADITIONAL_TO_SIMPLIFIED = str.maketrans(
    {
        "過": "过",
        "這": "这",
        "個": "个",
        "們": "们",
        "會": "会",
        "為": "为",
        "與": "与",
        "說": "说",
        "對": "对",
        "還": "还",
        "後": "后",
        "裡": "里",
        "讓": "让",
        "從": "从",
        "沒": "没",
        "麼": "么",
        "開": "开",
        "關": "关",
        "聽": "听",
        "嗎": "吗",
        "妳": "你",
        "臺": "台",
        "萬": "万",
        "點": "点",
        "現": "现",
        "難": "难",
        "訴": "诉",
        "邊": "边",
        "實": "实",
        "發": "发",
        "裡": "里",
        "來": "来",
        "給": "给",
        "盧": "卢",
        "爾": "尔",
        "壞": "坏",
        "長": "长",
        "聖": "圣",
        "馬": "马",
        "樂": "乐",
        "氣": "气",
        "該": "该",
        "話": "话",
        "誰": "谁",
        "樣": "样",
        "見": "见",
        "親": "亲",
        "愛": "爱",
        "傷": "伤",
        "離": "离",
        "歲": "岁",
        "間": "间",
        "無": "无",
        "帶": "带",
        "嗎": "吗",
        "妳": "你",
        "別": "别",
        "寫": "写",
        "張": "张",
        "連": "连",
        "聲": "声",
        "應": "应",
        "記": "记",
        "處": "处",
        "轉": "转",
        "臉": "脸",
        "淚": "泪",
        "備": "备",
        "選": "选",
        "戰": "战",
        "歡": "欢",
        "覺": "觉",
        "據": "据",
        "隻": "只",
        "條": "条",
        "體": "体",
        "價": "价",
        "臨": "临",
        "壓": "压",
        "媽": "妈",
        "爸": "爸",
    }
)

COMMON_MISTRANSLATION_HINTS_FILE = "common_mistranslation_hints.json"
COMMON_PHRASE_CORRECTION_HINTS_FILE = "common_phrase_correction_hints.json"
SUBTITLE_TERMINOLOGY_FILE = "subtitle_terminology.json"
STEP09_GUIDANCE_VERSION = "step09-guidance-20260712-v2"
GUIDANCE_LIMITS = {
    "terminology": 20,
    "mistranslation": 12,
    "phrase": 20,
}
SHORT_DIALOGUE_PHRASE_HINTS = [
    {
        "phrase": "yeah",
        "preferred_zh": "对 / 嗯 / 是啊 / 行，按上下文语气选择",
        "style_scope": "dialogue",
        "confidence": "high",
        "support_count": 3,
    },
    {
        "phrase": "come on",
        "preferred_zh": "得了吧 / 拜托 / 快点 / 走吧 / 上啊，按语气选择",
        "bad_zh": "不要机械译成“来吧”",
        "style_scope": "dialogue",
        "confidence": "high",
        "support_count": 3,
    },
    {
        "phrase": "you know what",
        "preferred_zh": "这样吧 / 你知道吗，按后文功能选择",
        "style_scope": "dialogue",
        "confidence": "high",
        "support_count": 3,
    },
    {
        "phrase": "get out",
        "preferred_zh": "表示惊讶时可译“真的假的”；表示驱赶时才译“出去”",
        "bad_zh": "不要不看语气一律译成“出去”",
        "style_scope": "dialogue",
        "confidence": "high",
        "support_count": 2,
    },
    {
        "phrase": "fair enough",
        "preferred_zh": "有道理 / 也行 / 这倒也是",
        "style_scope": "dialogue",
        "confidence": "high",
        "support_count": 2,
    },
]

DIALOGUE_SYSTEM_PROMPT = """You are a professional TV subtitle translator.
Input is one English dialogue subtitle segment from a TV episode.
Return strict JSON only:
{"zh": "natural concise Chinese subtitle text"}
When the user asks for display_units, return:
{"zh": "full translation", "display_units": [{"en": "exact English slice", "zh": "matching Chinese slice"}]}

Rules:
1. Output JSON only, with no explanation or reasoning.
2. Use Simplified Chinese only. Never use Traditional Chinese.
3. Translate like a polished finished TV subtitle track in spoken Chinese, not written prose.
4. Prefer colloquial, natural, emotionally accurate wording that sounds like people really talking.
5. Preserve attitude, sarcasm, teasing, threat, tenderness, and other subtext when present.
6. Preserve proper nouns when appropriate and avoid stiff literal translation or obvious translationese.
7. Pay close attention to negation, conditionals, who did what, and do not invert the meaning.
8. Keep the line concise and easy to read on screen.
9. Translate intention and tone, not English word order. Reorder clauses when Chinese sounds better.
10. You may omit redundant pronouns, fillers, and connectors when the Chinese remains clear.
11. Avoid translated-English patterns such as "关于...", "这不是关于...", "你不会得到去...", "我能看到你在...".
12. Do not over-formalize casual dialogue. Keep it conversational, but not internet-slangy.
13. If a literal translation sounds stiff, rewrite the whole line into fluent spoken Chinese.
14. For very short spoken lines, choose the Chinese by local intent and tone; do not use one fixed dictionary translation.
15. terminology.preferred_zh is literal preferred wording; terminology.note and all reusable guidance are instructions, never subtitle text.
16. If reusable guidance gives multiple candidates, pick the candidate that fits the current context instead of copying the whole hint.
17. If display_units are requested, each display_units.en must be an exact consecutive slice copied from the current English segment."""

DIALOGUE_BATCH_SYSTEM_PROMPT = """You are a professional TV subtitle translator.
The input contains a short batch of active English dialogue subtitle items plus read-only context before and after them.
Return strict JSON only:
{"items": [{"id": 1, "zh": "natural concise Chinese subtitle text"}]}
When an item has request_display_units=true, that item may also contain:
{"display_units": [{"en": "exact English slice", "zh": "matching Chinese slice"}]}

Rules:
1. Output JSON only, with no explanation or reasoning.
2. Return exactly one item for every active input id, using the same ids and order. Never return read-only context items.
3. Translate each active item as its own subtitle. Never merge text across ids, shift a translation to a neighboring id, or omit repeated lines.
4. Use all active items and the read-only context to resolve meaning, grammar, pronouns, tone, continuity, and consistent terminology.
5. The read-only context is evidence only. Do not translate it, quote it, or include it in an active item's zh.
6. Use Simplified Chinese only. Never use Traditional Chinese.
7. Write polished, concise, natural spoken Chinese suitable for finished TV subtitles, not written prose or English word order.
8. Preserve attitude, sarcasm, teasing, threat, tenderness, negation, conditionals, and who did what.
9. Preserve proper nouns and keep names, titles, forms of address, and recurring terminology consistent across the batch.
10. You may omit redundant pronouns, fillers, and connectors only when the meaning stays clear.
11. For very short lines, choose Chinese by local intent and tone instead of a fixed dictionary equivalent.
12. Each item's guidance applies only when it fits that active item. Context overrides reusable guidance.
13. When guidance.terminology.source occurs in that item's en, use its preferred_zh literally. terminology.note and all other guidance are instructions and must never appear as subtitle text.
14. If request_display_units=false, omit display_units for that item.
15. If request_display_units=true, use 2 units, or 3 only when necessary; every display_units.en must be an exact consecutive slice of that item's en, and the slices joined with single spaces must reproduce that item's en.
16. Never place another item's source or translation inside display_units."""

LYRIC_SYSTEM_PROMPT = """You are a professional song lyric subtitle translator.
Input is one English lyric segment from a TV episode.
Return strict JSON only:
{"zh": "natural concise Chinese lyric subtitle text"}
When the user asks for display_units, return:
{"zh": "full lyric translation", "display_units": [{"en": "exact English lyric slice", "zh": "matching Chinese lyric slice"}]}

Rules:
1. Output JSON only, with no explanation or reasoning.
2. Use Simplified Chinese only. Never use Traditional Chinese.
3. Preserve the lyric feeling, imagery, rhythm, and emotional contour.
4. Keep the Chinese concise and readable as subtitles.
5. Do not over-explain metaphors or translate like ordinary dialogue.
6. Keep repeated lyric lines naturally repeated.
7. Use Chinese punctuation sparingly.
8. Preserve proper nouns when appropriate.
9. If display_units are requested, each display_units.en must be an exact consecutive slice copied from the current English segment."""

CHANT_SYSTEM_PROMPT = """You are a professional subtitle translator for magical chants and incantations.
Input is one chant, spell, or ritual subtitle segment from a TV episode.
Return strict JSON only:
{"zh": "natural concise Simplified Chinese subtitle text"}
When the user asks for display_units, return:
{"zh": "full translation", "display_units": [{"en": "exact source slice", "zh": "matching Chinese slice"}]}

Rules:
1. Output JSON only, with no explanation or reasoning.
2. Use Simplified Chinese only. Never use Traditional Chinese.
3. Preserve a ritual, spell-like tone without adding exposition.
4. If the source is an invented or foreign spell phrase and the meaning is unclear, keep it as a transliterated/ritual phrase instead of inventing a false literal meaning.
5. Preserve repetition and short spell cadence.
6. Do not translate the segment as ordinary casual dialogue.
7. If display_units are requested, each display_units.en must be an exact consecutive slice copied from the current segment."""


FIXED_CHANT_PHRASES = {
    ("yoli", "conepiya", "miquiliztli"): "\u7ea6\u5229\u00b7\u79d1\u6d85\u76ae\u4e9a\u00b7\u7c73\u57fa\u5229\u5179\u7279\u5229\u3002",
    ("spasiti", "animam", "suam"): "\u62ef\u6551\u4ed6\u7684\u7075\u9b42\u3002",
    ("animam", "suam"): "\u4ed6\u7684\u7075\u9b42\u3002",
    ("tillate", "ulaz"): "\u63d0\u62c9\u7279\u00b7\u4e4c\u62c9\u5179\u3002",
    ("tillate",): "\u63d0\u62c9\u7279\u2026\u2026",
    ("louvri", "animo"): "\u5f00\u542f\u5fc3\u95e8\u3002",
    ("louvri",): "\u5f00\u542f\u2026\u2026",
    ("p\u00e9m\u00e9t", "pou", "accessum"): "\u51c6\u6211\u8fdb\u5165\u3002",
    ("pemet", "pou", "accessum"): "\u51c6\u6211\u8fdb\u5165\u3002",
    ("les", "lames", "colligo"): "\u805a\u96c6\u5200\u5203\u3002",
    ("ad", "me", "gluttuli"): "\u5f52\u4e8e\u6211\u624b\u3002",
    ("anima", "marcam"): "\u5524\u9192\u5370\u8bb0\u3002",
    ("iskoristi", "vuka"): "\u501f\u72fc\u4e4b\u529b\u3002",
    ("iskoristi", "vuca"): "\u501f\u72fc\u4e4b\u529b\u3002",
    ("isk",): "\u501f\u2026\u2026",
    ("magia", "tollox", "de", "terras"): "\u5524\u8d77\u5927\u5730\u4e4b\u529b\u3002",
    ("repo", "oma", "dal", "most"): "破除结界。",
}


def positive_int_env(name: str, default: int) -> int:
    try:
        return max(int(os.environ.get(name, default)), 1)
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 STEP7: run one Qwen session for dialogue, lyrics, OCR, and notes.")
    add_common_args(parser)
    parser.add_argument(
        "--source",
        choices=["auto", "audio", "embedded"],
        default="auto",
        help="Dialogue subtitle source. auto prefers embedded_json for selected stems and falls back to audio outputs.",
    )
    parser.add_argument("--skip-ocr", action="store_true", help="Only translate dialogue; skip OCR translation and notes.")
    parser.add_argument(
        "--ocr-over-embedded",
        action="store_true",
        help="Allow OCR translation/notes for embedded-subtitle sources too "
        "(normally OCR is skipped there because the dialogue already comes from SDH).",
    )
    parser.add_argument("--force-dialogue", action="store_true", help="Retranslate dialogue even if translated JSON exists.")
    parser.add_argument("--force-ocr", action="store_true", help="Regenerate OCR translations and notes even if outputs exist.")
    parser.add_argument(
        "--dialogue-batch-size",
        type=int,
        default=positive_int_env("QWEN_DIALOGUE_BATCH_SIZE", DEFAULT_DIALOGUE_BATCH_SIZE),
        help="Active dialogue subtitles per Qwen request. Default: 4.",
    )
    parser.add_argument(
        "--dialogue-context-size",
        type=int,
        default=positive_int_env("QWEN_DIALOGUE_CONTEXT_SIZE", DEFAULT_DIALOGUE_CONTEXT_SIZE),
        help="Read-only subtitle items before and after each dialogue batch. Default: 4 per side.",
    )
    parser.add_argument(
        "--qwen-profile",
        choices=sorted(qwen_common.QWEN_PROFILES),
        default=os.environ.get("QWEN_PROFILE", qwen_common.DEFAULT_QWEN_PROFILE),
        help="Local Qwen model/runtime profile. Use 32b to fall back from the higher-quality 80b profile.",
    )
    parser.add_argument(
        "--qwen-gguf",
        default="",
        help="Optional GGUF path override for the selected Qwen profile.",
    )
    return parser.parse_args()


def parse_json_response(content: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group())
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {"zh": cleaned}


def normalize_inline_text(text: str) -> str:
    text = str(text or "")
    text = normalize_music_symbols(text)
    text = text.replace("\\N", "\n").replace("\r\n", "\n").replace("\r", "\n")
    parts = [part.strip() for part in text.split("\n") if part.strip()]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def normalize_music_symbols(text: str) -> str:
    return MOJIBAKE_MUSIC_MARK_RE.sub(MUSIC_SYMBOL, str(text or ""))


def strip_speaker_placeholders(text: str) -> str:
    text = re.sub(r"\[SPEAKER_\d+\]\s*:?\s*", "", str(text or ""), flags=re.IGNORECASE)
    text = re.sub(r"SPEAKER_\d+\s*:?\s*", "", text, flags=re.IGNORECASE)
    while text:
        updated = LEADING_SPEAKER_TAG_RE.sub("", text, count=1)
        if updated == text:
            break
        text = updated
    return normalize_inline_text(text)


def normalize_music_marks(text: str, is_music: bool) -> str:
    text = normalize_inline_text(text)
    if not is_music:
        return text.replace(MUSIC_SYMBOL, "").strip()
    stripped = text.strip()
    if stripped.startswith(MUSIC_SYMBOL) and stripped.endswith(MUSIC_SYMBOL):
        return stripped
    core = stripped.strip(MUSIC_SYMBOL).strip()
    return f"{MUSIC_SYMBOL} {core} {MUSIC_SYMBOL}" if core else stripped


def to_simplified(text: str) -> str:
    return str(text or "").translate(TRADITIONAL_TO_SIMPLIFIED)


def module_dir() -> Path:
    return Path(__file__).resolve().parent


def load_hint_dict_list(path: Path) -> list[dict]:
    data = load_json(path, [])
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def load_step09_guidance(base_dir: Path, source_name: str = "") -> dict:
    terminology_data = load_json(base_dir / SUBTITLE_TERMINOLOGY_FILE, {"entries": []})
    terminology_entries = terminology_data.get("entries") if isinstance(terminology_data, dict) else []
    if not isinstance(terminology_entries, list):
        terminology_entries = []
    source_haystack = re.sub(r"[._-]+", " ", source_name)
    scoped_terminology: list[dict] = []
    for item in terminology_entries:
        if not isinstance(item, dict) or item.get("reviewed") is False:
            continue
        series = normalize_inline_text(item.get("series", ""))
        if series and not hint_term_matches(series, source_haystack):
            continue
        scoped_terminology.append(item)
    return {
        "version": STEP09_GUIDANCE_VERSION,
        "terminology": scoped_terminology,
        "mistranslation": load_hint_dict_list(module_dir() / COMMON_MISTRANSLATION_HINTS_FILE),
        "phrase": load_hint_dict_list(module_dir() / COMMON_PHRASE_CORRECTION_HINTS_FILE) + SHORT_DIALOGUE_PHRASE_HINTS,
    }


def normalize_hint_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace(r"\N", " ")).strip().lower()


def hint_term_matches(term: str, haystack: str) -> bool:
    term = normalize_hint_match_text(term)
    haystack = normalize_hint_match_text(haystack)
    if not term or not haystack:
        return False
    pattern = re.escape(term)
    pattern = re.sub(r"\\\s+", r"\\s+", pattern)
    return bool(re.search(rf"(?<![0-9A-Za-z]){pattern}(?![0-9A-Za-z])", haystack, re.IGNORECASE))


def hint_scope_values(item: dict) -> set[str]:
    raw = item.get("style_scope") or item.get("scope") or item.get("kind") or ""
    if isinstance(raw, list):
        values = raw
    else:
        values = re.split(r"[,/| ]+", str(raw))
    return {str(value).strip().lower() for value in values if str(value).strip()}


def hint_scope_matches(item: dict, segment_kind: str, *, default_all: bool) -> bool:
    scopes = hint_scope_values(item)
    if not scopes:
        return default_all
    if "all" in scopes:
        return True
    if segment_kind == "lyric":
        return bool(scopes & {"lyric", "lyrics", "music", "song"})
    if segment_kind == "chant":
        return bool(scopes & {"chant", "incantation"})
    return bool(scopes & {"dialogue", "spoken", "line"})


def hint_confidence_score(item: dict) -> int:
    raw = item.get("confidence")
    if isinstance(raw, (int, float)):
        return int(raw)
    text = str(raw or "").strip().lower()
    return {"high": 3, "medium": 2, "med": 2, "low": 1}.get(text, 2)


def hint_support_count(item: dict) -> int:
    try:
        return max(int(item.get("support_count") or item.get("count") or 1), 1)
    except (TypeError, ValueError):
        return 1


def hint_guidance(item: dict) -> str:
    return normalize_inline_text(item.get("guidance") or item.get("preferred_zh") or item.get("zh") or item.get("note") or "")


def sort_guidance_items(items: list[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda item: (
            hint_confidence_score(item),
            hint_support_count(item),
            len(str(item.get("source") or item.get("term") or item.get("phrase") or "")),
        ),
        reverse=True,
    )


def collect_matching_guidance_items(
    items: list[dict],
    haystack: str,
    *,
    key_fields: tuple[str, ...],
    segment_kind: str,
    default_all: bool,
    limit: int,
) -> list[dict]:
    selected: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if not hint_scope_matches(item, segment_kind, default_all=default_all):
            continue
        source = ""
        for field in key_fields:
            source = normalize_inline_text(item.get(field, ""))
            if source:
                break
        guidance = hint_guidance(item)
        if not source or not guidance:
            continue
        if not hint_term_matches(source, haystack):
            continue
        key = normalize_hint_match_text(source)
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "source": source,
                "guidance": guidance,
                "bad_zh": normalize_inline_text(item.get("bad_zh", "")),
                "confidence": item.get("confidence", ""),
                "support_count": hint_support_count(item),
            }
        )
    return sort_guidance_items(selected)[:limit]


def collect_matching_terminology_items(
    items: list[dict],
    haystack: str,
    *,
    segment_kind: str,
    limit: int,
) -> list[dict]:
    selected: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if not hint_scope_matches(item, segment_kind, default_all=True):
            continue
        source = normalize_inline_text(item.get("source_en") or item.get("term") or "")
        preferred_zh = normalize_inline_text(item.get("preferred_zh") or "")
        raw_aliases = item.get("aliases") or []
        aliases = raw_aliases if isinstance(raw_aliases, list) else [raw_aliases]
        candidates = [source, *(normalize_inline_text(alias) for alias in aliases)]
        flags = 0 if item.get("case_sensitive") is True else re.IGNORECASE
        matched = False
        for candidate in candidates:
            if not candidate:
                continue
            pattern = re.escape(normalize_hint_match_text(candidate) if flags else normalize_inline_text(candidate))
            pattern = re.sub(r"\\\s+", r"\\s+", pattern)
            comparison_text = normalize_hint_match_text(haystack) if flags else normalize_inline_text(haystack)
            if re.search(rf"(?<![0-9A-Za-z]){pattern}(?![0-9A-Za-z])", comparison_text, flags):
                matched = True
                break
        if not source or not preferred_zh or not matched:
            continue
        key = normalize_hint_match_text(source)
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "source": source,
                "preferred_zh": preferred_zh,
                "note": normalize_inline_text(item.get("note", "")),
                "confidence": item.get("confidence", ""),
                "support_count": hint_support_count(item),
            }
        )
    return sort_guidance_items(selected)[:limit]


def select_step09_translation_guidance(guidance: dict, text: str, context: dict | None, segment_kind: str) -> dict:
    context_text = ""
    if context:
        context_text = f"{context.get('previous') or ''}\n{context.get('next') or ''}"
    haystack = f"{text}\n{context_text}"
    current_word_count = len(HINT_TOKEN_RE.findall(text))
    return {
        "terminology": collect_matching_terminology_items(
            guidance.get("terminology", []),
            haystack,
            segment_kind=segment_kind,
            limit=GUIDANCE_LIMITS["terminology"],
        ),
        "mistranslation": collect_matching_guidance_items(
            guidance.get("mistranslation", []),
            haystack,
            key_fields=("source_en", "term", "phrase"),
            segment_kind=segment_kind,
            default_all=True,
            limit=GUIDANCE_LIMITS["mistranslation"],
        ),
        "phrase": collect_matching_guidance_items(
            guidance.get("phrase", []),
            text,
            key_fields=("source_en", "phrase", "term"),
            segment_kind=segment_kind,
            default_all=segment_kind == "dialogue",
            limit=GUIDANCE_LIMITS["phrase"],
        ),
        "short_dialogue": segment_kind == "dialogue" and 0 < current_word_count <= 5,
    }


def format_guidance_line(item: dict) -> str:
    line = f"- {item['source']}：{item['guidance']}"
    if item.get("bad_zh"):
        line += f"；规避：{item['bad_zh']}"
    return line


def format_terminology_line(item: dict) -> str:
    line = f"- {item['source']} -> {item['preferred_zh']}"
    if item.get("note"):
        line += f"；使用说明（不要照抄）：{item['note']}"
    return line


def format_step09_guidance(hints: dict | None, segment_kind: str) -> str:
    if not hints:
        return ""
    sections: list[str] = []
    if hints.get("terminology"):
        sections.append("Exact terminology:\n" + "\n".join(format_terminology_line(item) for item in hints["terminology"]))
    if hints.get("mistranslation"):
        sections.append("Mistranslation traps:\n" + "\n".join(format_guidance_line(item) for item in hints["mistranslation"]))
    if hints.get("phrase"):
        sections.append("Phrase tendencies:\n" + "\n".join(format_guidance_line(item) for item in hints["phrase"]))
    if not sections and not hints.get("short_dialogue"):
        return ""
    lines = [
        "",
        "Reusable guidance matched from prior Step09 polish. Apply it only if it fits the CURRENT segment; context overrides hints.",
        "Only the right side of an Exact terminology arrow is literal subtitle wording; usage notes and all other guidance are instructions and must never be copied wholesale.",
    ]
    if segment_kind == "lyric":
        lines.append("This is a lyric segment: keep lyric feeling and ignore dialogue-only colloquial hints.")
    if hints.get("short_dialogue"):
        lines.append("For this short spoken line, pick a natural interjection or response by intent and tone; do not translate word-by-word.")
    lines.extend(sections)
    return "\n\n" + "\n".join(lines)


def format_zh_text(text: str, is_music: bool) -> str:
    text = to_simplified(normalize_inline_text(text))
    core = text.strip().strip(MUSIC_SYMBOL).strip()
    if not is_music:
        return core
    return f"{MUSIC_SYMBOL} {core} {MUSIC_SYMBOL}" if core else ""


def break_text(text: str, max_chars: int) -> str:
    text = normalize_inline_text(text)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    lines: list[str] = []
    rest = text
    while len(rest) > max_chars:
        cut = max_chars
        lower_bound = max(max_chars - 12, 1)
        for index in range(max_chars, lower_bound - 1, -1):
            if index < len(rest) and rest[index] in BREAK_CHARS:
                cut = index + 1
                break
        lines.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        lines.append(rest)
    return r"\N".join(lines)


def update_progress_status(base_dir: Path, step: str, video: str, index: int, total: int, en_text: str, zh_text: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    remaining = max(total - index, 0)
    progress = 100.0 if total <= 0 else index * 100.0 / total
    runtime_file_for(base_dir, "subtitle_status.txt").write_text(
        (
            f"time: {now}\n"
            f"step: {step}\n"
            f"video: {video}\n"
            f"message: Qwen translation/postprocess\n"
            f"progress: {index}/{total} ({progress:.1f}%)\n"
            f"remaining: {remaining}\n"
            f"current_en: {status_preview(en_text)}\n"
            f"current_zh: {status_preview(zh_text, 60)}\n"
        ),
        encoding="utf-8",
    )


def update_note_status(base_dir: Path, video: str, phase: str, index: int, total: int, detail: str = "") -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    progress = 100.0 if total <= 0 else index * 100.0 / total
    detail_line = f"detail: {status_preview(detail, 100)}\n" if detail else ""
    runtime_file_for(base_dir, "subtitle_status.txt").write_text(
        (
            f"time: {now}\n"
            f"step: V2_STEP7\n"
            f"video: {video}\n"
            f"message: cultural notes - {phase}\n"
            f"progress: {index}/{total} ({progress:.1f}%)\n"
            f"{detail_line}"
        ),
        encoding="utf-8",
    )


def semantic_split_requested(seg: dict, en_text: str) -> bool:
    try:
        duration = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    word_count = len(en_text.split())
    if word_count < DISPLAY_MIN_WORDS:
        return False
    return duration > DISPLAY_MAX_DURATION or len(en_text) > EN_DISPLAY_MAX_CHARS


def clean_display_unit_zh(text: str) -> str:
    return to_simplified(normalize_inline_text(text)).strip().strip(MUSIC_SYMBOL).strip()


def en_tokens(text: str) -> list[str]:
    return [match.group(0).replace("’", "'").lower() for match in EN_TOKEN_RE.finditer(str(text or ""))]


def display_units_cover_source(units: list[dict], source_text: str) -> bool:
    source_tokens = en_tokens(source_text)
    unit_tokens: list[str] = []
    for unit in units:
        unit_tokens.extend(en_tokens(unit.get("en", "")))
    return bool(source_tokens) and unit_tokens == source_tokens


def clean_display_units(value, source_text: str) -> list[dict]:
    if not isinstance(value, list):
        return []
    units: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        en = normalize_inline_text(item.get("en", ""))
        zh = clean_display_unit_zh(item.get("zh", ""))
        if en and zh and en_tokens(en):
            units.append({"en": en, "zh": zh})
    if len(units) < 2 or not display_units_cover_source(units, source_text):
        return []
    return units


def fixed_chant_translation(text: str) -> str:
    tokens = en_tokens(text)
    if not tokens:
        return ""
    output: list[str] = []
    index = 0
    phrase_items = sorted(FIXED_CHANT_PHRASES.items(), key=lambda item: len(item[0]), reverse=True)
    while index < len(tokens):
        matched = False
        for phrase, zh in phrase_items:
            size = len(phrase)
            if tuple(tokens[index : index + size]) == phrase:
                output.append(zh)
                index += size
                matched = True
                break
        if not matched:
            return ""
    return "".join(output)



def fixed_recap_translation(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    return FIXED_RECAP_ZH if "previously on young sheldon" in normalized else ""


def translate_segment(
    client,
    text: str,
    is_music: bool = False,
    context: dict | None = None,
    request_display_units: bool = False,
    kind: str | None = None,
    translation_hints: dict | None = None,
) -> dict:
    segment_kind = kind or ("lyric" if is_music else "dialogue")
    if segment_kind == "dialogue":
        fixed_recap = fixed_recap_translation(text)
        if fixed_recap:
            return {"zh": fixed_recap}
    if segment_kind == "chant":
        fixed_zh = fixed_chant_translation(text)
        if fixed_zh:
            return {"zh": fixed_zh}
    if segment_kind == "chant":
        system_prompt = CHANT_SYSTEM_PROMPT
    elif is_music or segment_kind == "lyric":
        system_prompt = LYRIC_SYSTEM_PROMPT
    else:
        system_prompt = DIALOGUE_SYSTEM_PROMPT
    display_instruction = ""
    if request_display_units:
        display_instruction = (
            "\n\nAlso propose display_units for on-screen subtitle display.\n"
            "Rules for display_units:\n"
            "- Split only at natural semantic, lyric-phrase, or chant-phrase boundaries.\n"
            "- Prefer 2 units; use 3 only when the segment is very long.\n"
            "- Each unit must be readable on its own.\n"
            "- The display_units.en values must be exact consecutive slices copied from the current English segment.\n"
            "- The display_units.en values, joined with single spaces, must reproduce the current English segment.\n"
            "- The display_units.zh values should be matching Simplified Chinese slices of the full translation.\n"
            "- Do not add music symbols to display_units.zh."
        )
    guidance_instruction = format_step09_guidance(translation_hints, segment_kind)
    if context:
        kind_label = {
            "chant": "magical chant/incantation subtitle",
            "lyric": "song lyric subtitle",
        }.get(segment_kind, "subtitle")
        user_content = (
            f"Translate ONLY the current {kind_label} segment into concise natural Chinese.\n"
            "The previous and next segments are context only. Use them to understand meaning, grammar, pronouns, and continuity.\n"
            "Return only the current segment translation.\n\n"
            f"Previous segment: {context.get('previous') or '(none)'}\n"
            f"Current segment: {text}\n"
            f"Next segment: {context.get('next') or '(none)'}"
            f"{guidance_instruction}"
            f"{display_instruction}"
        )
    else:
        if segment_kind == "chant":
            user_task = "Translate this magical chant/incantation subtitle into concise natural Chinese:\n"
        elif is_music or segment_kind == "lyric":
            user_task = "Translate this song lyric subtitle into concise natural Chinese:\n"
        else:
            user_task = "Translate this subtitle into concise natural Chinese:\n"
        user_content = f"{user_task}{text}{guidance_instruction}{display_instruction}"
    try:
        resp = client.chat.completions.create(
            model=qwen_common.QWEN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=520 if request_display_units else (260 if is_music else 220),
        )
        content = (resp.choices[0].message.content or "").strip()
        return parse_json_response(content)
    except Exception as exc:
        raise RuntimeError(
            f"Qwen translation request failed for subtitle: {status_preview(text, 80)}"
        ) from exc


def context_window_items(segments: list[dict], anchor: int, step: int, limit: int) -> list[dict]:
    items: list[dict] = []
    cursor = anchor + step
    while 0 <= cursor < len(segments) and len(items) < max(limit, 0):
        seg = segments[cursor]
        text = strip_speaker_placeholders(seg.get("text", ""))
        if text:
            is_music = bool(seg.get("is_music", False))
            items.append(
                {
                    "id": cursor + 1,
                    "kind": segment_translation_kind(seg, is_music),
                    "en": text,
                }
            )
        cursor += step
    if step < 0:
        items.reverse()
    return items


def dialogue_batch_payload(
    segments: list[dict],
    indices: list[int],
    step09_guidance: dict,
    context_size: int,
) -> dict:
    if not indices:
        return {"read_only_context_before": [], "items": [], "read_only_context_after": []}
    active_items = []
    for index in indices:
        seg = segments[index]
        en_text = strip_speaker_placeholders(seg.get("text", ""))
        hints = select_step09_translation_guidance(
            step09_guidance,
            en_text,
            translation_context(segments, index, context_size),
            "dialogue",
        )
        active_items.append(
            {
                "id": index + 1,
                "en": en_text,
                "request_display_units": semantic_split_requested(seg, en_text),
                "guidance": hints,
            }
        )
    return {
        "read_only_context_before": context_window_items(segments, indices[0], -1, context_size),
        "items": active_items,
        "read_only_context_after": context_window_items(segments, indices[-1], 1, context_size),
    }


def parse_dialogue_batch_response(content: str, expected_ids: list[int]) -> tuple[dict[int, dict], list[int], list[str]]:
    parsed = parse_json_response(content)
    raw_items = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(raw_items, list):
        return {}, list(expected_ids), ["response does not contain an items array"]

    expected = set(expected_ids)
    results: dict[int, dict] = {}
    duplicates: set[int] = set()
    received_order: list[int] = []
    issues: list[str] = []
    for row in raw_items:
        if not isinstance(row, dict):
            issues.append("non-object response item")
            continue
        raw_id = row.get("id")
        if isinstance(raw_id, bool):
            issues.append(f"invalid response id={raw_id!r}")
            continue
        try:
            item_id = int(raw_id)
        except (TypeError, ValueError):
            issues.append(f"invalid response id={raw_id!r}")
            continue
        if item_id not in expected:
            issues.append(f"unexpected response id={item_id}")
            continue
        received_order.append(item_id)
        if item_id in results or item_id in duplicates:
            duplicates.add(item_id)
            results.pop(item_id, None)
            continue
        zh = normalize_inline_text(row.get("zh", ""))
        if not zh:
            issues.append(f"empty zh for id={item_id}")
            continue
        result = {"zh": zh}
        if "display_units" in row:
            result["display_units"] = row.get("display_units")
        results[item_id] = result

    if duplicates:
        issues.append("duplicate response ids=" + ",".join(str(item_id) for item_id in sorted(duplicates)))
    if received_order != [item_id for item_id in expected_ids if item_id in received_order]:
        issues.append("response ids are out of order")
    fallback_ids = [item_id for item_id in expected_ids if item_id not in results]
    if fallback_ids:
        issues.append("missing/invalid response ids=" + ",".join(str(item_id) for item_id in fallback_ids))
    return results, fallback_ids, issues


def translate_dialogue_batch(
    client,
    segments: list[dict],
    indices: list[int],
    step09_guidance: dict,
    context_size: int = DEFAULT_DIALOGUE_CONTEXT_SIZE,
) -> tuple[dict[int, dict], list[int]]:
    payload = dialogue_batch_payload(segments, indices, step09_guidance, context_size)
    expected_ids = [item["id"] for item in payload["items"]]
    if not expected_ids:
        return {}, []
    prompt = (
        "Translate only payload.items. Treat both context arrays as read-only. "
        "Return one result for every active id in the same order.\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    try:
        resp = client.chat.completions.create(
            model=qwen_common.QWEN_MODEL,
            messages=[
                {"role": "system", "content": DIALOGUE_BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=min(2200, 500 + 420 * len(expected_ids)),
        )
        content = (resp.choices[0].message.content or "").strip()
        results, fallback_ids, issues = parse_dialogue_batch_response(content, expected_ids)
        if issues:
            print("  [warning] dialogue batch validation: " + "; ".join(issues))
        return results, fallback_ids
    except Exception as exc:
        if not qwen_common.server_ready(timeout=3, expected_model=qwen_common.QWEN_MODEL):
            raise RuntimeError(
                f"Qwen service became unavailable during a dialogue batch; profile={qwen_common.QWEN_PROFILE}. "
                "Restart with --qwen-profile 32b to continue with the lower-memory model."
            ) from exc
        print(f"  [warning] dialogue batch failed while the service is still alive; falling back to single items: {exc}")
        return {}, expected_ids


def retry_display_units(client, text: str, zh_text: str, is_music: bool, kind: str | None = None) -> list[dict]:
    display_kind = kind or ("lyric" if is_music else "dialogue")
    prompt = (
        f"Create display_units for this translated {display_kind} subtitle.\n"
        "Return strict JSON only:\n"
        "{\"display_units\": [{\"en\": \"exact English slice\", \"zh\": \"matching Simplified Chinese slice\"}]}\n\n"
        "Hard rules:\n"
        "1. Split into 2 units, or 3 only if absolutely necessary.\n"
        "2. Each en value must be an exact consecutive slice copied from English.\n"
        "3. The en values joined with single spaces must reproduce English exactly by words.\n"
        "4. Do not include previous or next subtitles.\n"
        "5. Do not create punctuation-only units.\n"
        "6. zh values should be matching slices of the Chinese translation.\n\n"
        f"English: {text}\n"
        f"Chinese: {zh_text}"
    )
    try:
        resp = client.chat.completions.create(
            model=qwen_common.QWEN_MODEL,
            messages=[
                {"role": "system", "content": "You split subtitles for display. Output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=420,
        )
        return clean_display_units(parse_json_response(resp.choices[0].message.content or "").get("display_units"), text)
    except Exception as exc:
        print(f"  [warning] display_units retry failed: {exc}")
        return []


def translation_context(segments: list[dict], index: int, context_size: int = 1) -> dict:
    previous = context_window_items(segments, index, -1, context_size)
    following = context_window_items(segments, index, 1, context_size)
    return {
        "previous": "\n".join(item["en"] for item in previous),
        "next": "\n".join(item["en"] for item in following),
    }


def segment_translation_kind(seg: dict, is_music: bool) -> str:
    raw_kind = str(seg.get("kind", "") or "").lower()
    if raw_kind == "chant" or bool(seg.get("is_chant")):
        return "chant"
    if is_music or raw_kind == "lyric":
        return "lyric"
    return "dialogue"


def clear_empty_segment(seg: dict) -> None:
    seg["text"] = ""
    seg["zh"] = ""
    seg["en_wrap"] = ""
    seg["is_music"] = False
    seg["kind"] = "dialogue"
    seg["is_chant"] = False
    seg.pop("display_units", None)
    seg.pop("display_units_source", None)


def apply_translation_result(client, seg: dict, en_text: str, segment_kind: str, result: dict) -> str:
    is_music = segment_kind == "lyric" or bool(seg.get("is_music", False))
    zh_text = normalize_music_marks(strip_speaker_placeholders(result.get("zh", en_text)), is_music)
    formatted_zh = format_zh_text(zh_text, is_music)
    seg["text"] = en_text
    seg["zh"] = formatted_zh
    seg["en_wrap"] = en_text
    seg["is_music"] = is_music
    seg["kind"] = segment_kind
    seg["is_chant"] = segment_kind == "chant"
    request_display_units = semantic_split_requested(seg, en_text)
    units = clean_display_units(result.get("display_units"), en_text) if request_display_units else []
    if request_display_units and not units:
        units = retry_display_units(client, en_text, formatted_zh, is_music, segment_kind)
    if units:
        seg["display_units"] = units
        seg["display_units_source"] = "qwen"
    else:
        seg.pop("display_units", None)
        seg.pop("display_units_source", None)
    return zh_text


def consecutive_dialogue_indices(segments: list[dict], start: int, batch_size: int) -> tuple[list[int], int]:
    indices: list[int] = []
    cursor = start
    while cursor < len(segments) and len(indices) < max(batch_size, 1):
        seg = segments[cursor]
        en_text = strip_speaker_placeholders(seg.get("text", ""))
        if not en_text:
            clear_empty_segment(seg)
            cursor += 1
            continue
        is_music = bool(seg.get("is_music", False))
        if segment_translation_kind(seg, is_music) != "dialogue":
            break
        indices.append(cursor)
        cursor += 1
    return indices, cursor


def translate_dialogue_file(
    client,
    base_dir: Path,
    json_file: Path,
    out_file: Path,
    dialogue_batch_size: int = DEFAULT_DIALOGUE_BATCH_SIZE,
    dialogue_context_size: int = DEFAULT_DIALOGUE_CONTEXT_SIZE,
) -> None:
    print(f"\n[dialogue] {json_file.name}")
    data = load_json(json_file, {"segments": []})
    segments = data.get("segments", [])
    overrides = load_overrides(base_dir, json_file.stem)
    total = len(segments)
    translation_cache: dict[tuple[str, str, bool], dict] = {}
    step09_guidance = load_step09_guidance(base_dir, json_file.stem)
    update_progress_status(base_dir, "V2_STEP7", json_file.stem, 0, total, "", "")

    offset = 0
    while offset < total:
        seg = segments[offset]
        index = offset + 1
        en_text = strip_speaker_placeholders(seg.get("text", ""))
        if not en_text:
            clear_empty_segment(seg)
            update_progress_status(base_dir, "V2_STEP7", json_file.stem, index, total, "", "")
            offset += 1
            continue

        is_music = bool(seg.get("is_music", False))
        segment_kind = segment_translation_kind(seg, is_music)
        if segment_kind == "dialogue":
            batch_indices, next_offset = consecutive_dialogue_indices(segments, offset, dialogue_batch_size)
            batch_ids = [item_index + 1 for item_index in batch_indices]
            print(
                f"  [{batch_ids[0]}-{batch_ids[-1]}/{total}] dialogue batch "
                f"items={len(batch_indices)} context={dialogue_context_size}+{dialogue_context_size}"
            )
            batch_results, _fallback_ids = translate_dialogue_batch(
                client,
                segments,
                batch_indices,
                step09_guidance,
                dialogue_context_size,
            )
            for item_index in batch_indices:
                item = segments[item_index]
                item_id = item_index + 1
                item_en = strip_speaker_placeholders(item.get("text", ""))
                fixed_recap = fixed_recap_translation(item_en)
                result = {"zh": fixed_recap} if fixed_recap else batch_results.get(item_id)
                if result is None:
                    context = translation_context(segments, item_index, dialogue_context_size)
                    translation_hints = select_step09_translation_guidance(
                        step09_guidance,
                        item_en,
                        context,
                        "dialogue",
                    )
                    result = translate_segment(
                        client,
                        item_en,
                        False,
                        context,
                        request_display_units=semantic_split_requested(item, item_en),
                        kind="dialogue",
                        translation_hints=translation_hints,
                    )
                zh_text = apply_translation_result(client, item, item_en, "dialogue", result)
                update_progress_status(
                    base_dir,
                    "V2_STEP7",
                    json_file.stem,
                    item_id,
                    total,
                    item_en,
                    zh_text,
                )
            offset = next_offset
            continue

        print(f"  [{index}/{total}] {segment_kind}: {status_preview(en_text, 40)}")
        request_display_units = semantic_split_requested(seg, en_text)
        cache_key = (segment_kind, en_text.strip().lower(), request_display_units)
        if segment_kind == "chant" and cache_key in translation_cache:
            result = copy.deepcopy(translation_cache[cache_key])
        else:
            context = translation_context(segments, offset)
            translation_hints = select_step09_translation_guidance(step09_guidance, en_text, context, segment_kind)
            result = translate_segment(
                client,
                en_text,
                is_music,
                context,
                request_display_units=request_display_units,
                kind=segment_kind,
                translation_hints=translation_hints,
            )
            if segment_kind == "chant":
                translation_cache[cache_key] = copy.deepcopy(result)
        zh_text = apply_translation_result(client, seg, en_text, segment_kind, result)
        update_progress_status(base_dir, "V2_STEP7", json_file.stem, index, total, en_text, zh_text)
        offset += 1

    override_counts = apply_segment_overrides(segments, overrides)
    if override_counts["segments"]:
        print(f"[overrides] {json_file.name}: {override_counts}")

    save_json(out_file, data)
    print(f"[done] translated dialogue: {out_file}")


def generate_notes(client, base_dir: Path, work_dir: Path, translated_file: Path) -> None:
    """Generate cultural notes for a translated dialogue file, independent of OCR.

    Works for any source (embedded or audio): it only needs the translated
    segments and lets Qwen decide which cues deserve a note.
    """
    if not OCR_HELPERS_AVAILABLE:
        print(f"[skip] notes for {translated_file.name}: ocr_translate_helpers.py missing")
        return
    notes_dir = work_dir / "notes"
    notes_file = notes_dir / translated_file.name

    translated_data = load_json(translated_file, {"segments": []})
    candidates = collect_dialogue_items(translated_data)
    if not candidates:
        save_json(notes_file, {"stem": translated_file.stem, "notes": []})
        print(f"[done] Notes: {notes_file.name} (0 notes, no candidates)")
        return

    glossary = load_glossary(base_dir)
    update_note_status(base_dir, translated_file.name, "collect dialogue candidates", 0, 3)
    update_note_status(
        base_dir,
        translated_file.name,
        f"scan candidates dialogue={len(candidates)}",
        1,
        3,
    )
    update_note_status(base_dir, translated_file.name, f"Qwen/glossary scan candidates={len(candidates)}", 2, 3)
    with StatusHeartbeat(
        base_dir,
        "V2_STEP7",
        translated_file.name,
        f"cultural notes Qwen/glossary scan candidates={len(candidates)}",
    ):
        notes = find_notes(client, glossary, candidates)
    update_note_status(base_dir, translated_file.name, f"dedupe notes raw={len(notes)}", 3, 3)
    notes = dedupe_notes(notes)[:MAX_NOTES_PER_VIDEO]
    save_json(notes_file, {"stem": translated_file.stem, "notes": notes})
    save_glossary(base_dir, glossary)
    print(f"[done] Notes: {notes_file.name} ({len(notes)} notes)")


def translate_ocr_and_notes(client, base_dir: Path, work_dir: Path, raw_file: Path) -> None:
    if not OCR_HELPERS_AVAILABLE:
        raise RuntimeError("OCR helpers unavailable: ocr_translate_helpers.py is missing")
    translated_dir = work_dir / "translated"
    out_dir = work_dir / "ocr_translated"
    notes_dir = work_dir / "notes"
    out_file = out_dir / raw_file.name
    notes_file = notes_dir / raw_file.name

    write_status(base_dir, "V2_STEP7", raw_file.name, "merge OCR text")
    raw = load_json(raw_file, {})
    merged_all = merge_ocr(raw)
    credit_windows = build_credit_windows(merged_all)
    merged = filter_ocr_groups(merged_all, credit_windows)
    print(
        f"[OCR merge] {raw_file.name}: {len(raw.get('detections', []))} detections "
        f"-> {len(merged_all)} groups -> {len(merged)} kept "
        f"({len(credit_windows)} credit windows)"
    )

    for offset in range(0, len(merged), OCR_BATCH_SIZE):
        batch = merged[offset : offset + OCR_BATCH_SIZE]
        write_status(base_dir, "V2_STEP7", raw_file.name, f"translate OCR {offset + len(batch)}/{len(merged)}")
        translations = translate_ocr_batch(client, batch)
        for item in batch:
            keep, zh = translations.get(item["id"], (True, ""))
            item["zh"] = (normalize_inline_text(zh) or item["text"]) if keep else ""
            item["keep"] = keep
    # Drop items Qwen classified as credits/disclaimers/garbage so they never reach the overlay.
    merged = [item for item in merged if item.get("keep", True)]

    glossary = load_glossary(base_dir)
    translated_data = load_json(translated_dir / raw_file.name, {"segments": []})
    update_note_status(base_dir, raw_file.name, "collect dialogue candidates", 0, 3)
    dialogue_items = collect_dialogue_items(translated_data)
    update_note_status(
        base_dir,
        raw_file.name,
        f"merge candidates dialogue={len(dialogue_items)} ocr={len(merged)}",
        1,
        3,
    )
    note_candidates = []
    note_candidates.extend(dialogue_items)
    for item in merged:
        note_candidates.append(
            {
                "id": 100000 + int(item["id"]),
                "source": "ocr",
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": item["text"],
                "translation": item.get("zh", ""),
                "bbox": item.get("bbox", []),
            }
        )

    update_note_status(base_dir, raw_file.name, f"Qwen/glossary scan candidates={len(note_candidates)}", 2, 3)
    with StatusHeartbeat(
        base_dir,
        "V2_STEP7",
        raw_file.name,
        f"cultural notes Qwen/glossary scan candidates={len(note_candidates)}",
    ):
        notes = find_notes(client, glossary, note_candidates)
    update_note_status(base_dir, raw_file.name, f"dedupe notes raw={len(notes)}", 3, 3)
    notes = dedupe_notes(notes)[:MAX_NOTES_PER_VIDEO]

    out_data = {
        "video": raw.get("video", raw_file.name),
        "stem": raw.get("stem", raw_file.stem),
        "width": raw.get("width", 1920),
        "height": raw.get("height", 1080),
        "items": merged,
    }
    save_json(out_file, out_data)
    save_json(notes_file, {"stem": raw_file.stem, "notes": notes})
    save_glossary(base_dir, glossary)
    print(f"[done] OCR translated: {out_file}")
    print(f"[done] Notes: {notes_file} ({len(notes)} notes)")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    diarized_dir = work_dir / "diarized"
    music_marked_dir = work_dir / "music_marked"
    embedded_json_dir = work_dir / "embedded_json"
    translated_dir = work_dir / "translated"
    ocr_raw_dir = work_dir / "ocr_raw"
    ocr_out_dir = work_dir / "ocr_translated"
    notes_dir = work_dir / "notes"
    translated_dir.mkdir(parents=True, exist_ok=True)
    ocr_out_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    stems = {video.stem for video in selected_videos(base_dir, args.chunk, args.target_stem)}
    embedded_files = filter_paths_by_stems(sorted(embedded_json_dir.glob("*.json")), stems)
    audio_source_dir = music_marked_dir if music_marked_dir.exists() else diarized_dir
    audio_files = filter_paths_by_stems(sorted(audio_source_dir.glob("*.json")), stems)
    if not audio_files and audio_source_dir != diarized_dir:
        audio_source_dir = diarized_dir
        audio_files = filter_paths_by_stems(sorted(diarized_dir.glob("*.json")), stems)

    if args.source == "embedded":
        dialogue_files = embedded_files
        dialogue_source_dir = embedded_json_dir
    elif args.source == "audio":
        dialogue_files = audio_files
        dialogue_source_dir = audio_source_dir
    else:
        embedded_stems = {path.stem for path in embedded_files}
        dialogue_files = embedded_files + [path for path in audio_files if path.stem not in embedded_stems]
        dialogue_source_dir = embedded_json_dir if embedded_files else audio_source_dir
    if not dialogue_files:
        print(f"[error] no dialogue files found for source={args.source}: {embedded_json_dir}, {music_marked_dir}, or {diarized_dir}")
        sys.exit(1)

    pending_dialogue = [
        path
        for path in dialogue_files
        if args.force_dialogue or not (translated_dir / path.name).exists()
    ]

    pending_ocr: list[Path] = []
    ocr_excluded_stems = set()
    if args.source in {"auto", "embedded"} and not args.ocr_over_embedded:
        ocr_excluded_stems.update(path.stem for path in embedded_files)
    if not args.skip_ocr:
        raw_files = filter_paths_by_stems(sorted(ocr_raw_dir.glob("*.json")), stems)
        pending_ocr = [
            path
            for path in raw_files
            if path.stem not in ocr_excluded_stems
            and (
                args.force_ocr
            or not (ocr_out_dir / path.name).exists()
            or not (notes_dir / path.name).exists()
            )
        ]

    # Notes are generated independently of OCR: any translated dialogue file can
    # receive cultural notes. Audio files that go through the OCR path already get
    # their notes from translate_ocr_and_notes, so they are excluded here to avoid
    # duplicate work; everything else (embedded subtitles included) is eligible.
    ocr_path_stems = {path.stem for path in pending_ocr}
    def collect_pending_notes() -> list[Path]:
        return [
            translated_dir / path.name
            for path in dialogue_files
            if path.stem not in ocr_path_stems
            and (args.force_dialogue or not (notes_dir / path.name).exists())
            and (translated_dir / path.name).exists()
        ]

    pending_notes = collect_pending_notes()

    if not pending_dialogue and not pending_ocr and not pending_notes:
        print("v2 step7 done; all selected Qwen outputs already exist")
        return

    qwen_common.configure_qwen_profile(args.qwen_profile, args.qwen_gguf or None)
    client, proc = qwen_common.get_client(base_dir)
    qwen_start = time.time()
    write_status(
        base_dir,
        "V2_STEP7",
        f"{len(pending_dialogue)} dialogue / {len(pending_ocr)} OCR / {len(pending_notes)} notes",
        "START Qwen session",
    )
    try:
        for json_file in pending_dialogue:
            out_file = translated_dir / json_file.name
            translate_dialogue_file(
                client,
                base_dir,
                json_file,
                out_file,
                dialogue_batch_size=max(args.dialogue_batch_size, 1),
                dialogue_context_size=max(args.dialogue_context_size, 1),
            )

        for raw_file in pending_ocr:
            translate_ocr_and_notes(client, base_dir, work_dir, raw_file)

        pending_notes = collect_pending_notes()
        for translated_file in pending_notes:
            generate_notes(client, base_dir, work_dir, translated_file)
    finally:
        qwen_common.stop_qwen_server(proc)

    elapsed = time.time() - qwen_start
    write_status(base_dir, "V2_STEP7", "", f"DONE Qwen session ({elapsed:.1f}s)")
    print(f"v2 step7 done; Qwen outputs saved in: {work_dir}; dialogue_source={dialogue_source_dir}")


if __name__ == "__main__":
    main()


