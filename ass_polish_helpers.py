import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ass_filter_helpers import has_chinese
from ass_overlay_helpers import ass_escape, break_zh, load_json


PROMPT_VERSION = "ass-polish-20260712-v7"
GLOSSARY_REVIEW_PROMPT_VERSION = "glossary-review-20260707-v1"
STATIC_HINT_REVIEW_PROMPT_VERSION = "static-hint-review-20260712-v3"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
DEFAULT_CODEX_REASONING_EFFORT = "high"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_STYLES = (
    "BILINGUAL,BILINGUAL_MUSIC,BILINGUAL_CHANT,OCR_TRANSLATION,EXPLANATION_NOTE,"
    "Chinese,Default,ZH,ZH_i"
)
DEFAULT_POLISH_PROVIDER = "codex-cli"
DEFAULT_GLOSSARY_REVIEW_BATCH_SIZE = 48
DEFAULT_POLISH_BATCH_SIZE = 48
DEFAULT_POLISH_FILE_BATCH_SIZE = 1
DEBUG_POLISH_DIR_ENV = "SUB_POLISH_DEBUG_DIR"
COMMON_MISTRANSLATION_HINTS_FILE = "common_mistranslation_hints.json"
COMMON_PHRASE_CORRECTION_HINTS_FILE = "common_phrase_correction_hints.json"
OCR_LOW_VALUE_SHORT_TEXTS_FILE = "ocr_low_value_short_texts.json"
SUBTITLE_TERMINOLOGY_FILE = "subtitle_terminology.json"
MUSIC_SYMBOL = "\u266a"
CHANT_SYMBOL = "\u2726"
EN_FONT_SIZE = 34
EN_SIZE_TAG = r"{\fs" + str(EN_FONT_SIZE) + "}"
MAX_BILINGUAL_ZH_CHARS = 42
BILINGUAL_ZH_WRAP_CHARS = 36
DELETE_SENTINEL = "__DELETE_OVERLAY_EVENT__"

ASS_TAG_RE = re.compile(r"\{[^}]*\}")
LEADING_ASS_TAGS_RE = re.compile(r"^((?:\{[^}]*\})*)")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
MOJIBAKE_MUSIC_MARK_RE = re.compile(
    r"(?<![0-9A-Za-z])(?:[jJ][“”\"']|[jJ]n(?=\s))|[jJ][“”\"'](?=\s|$)"
)
LATIN_RE = re.compile(r"[A-Za-z]")
HAN_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
CJK_INTERNAL_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff，。、；：？！《》“”（）])\s+(?=[\u4e00-\u9fff，。、；：？！《》“”（）])")
PRESENTATION_TAG_COMMANDS = [
    re.compile(r"\\fn[^\\}]*"),
    re.compile(r"\\fs[+-]?\d+(?:\.\d+)?"),
    re.compile(r"\\fsc[xy][+-]?\d+(?:\.\d+)?"),
    re.compile(r"\\b[01]"),
    re.compile(r"\\i[01]"),
    re.compile(r"\\(?:[1-4]?c|alpha|[1-4]a)&H[0-9A-Fa-f]+&"),
]

BILINGUAL_STYLES = {
    "BILINGUAL": "dialogue",
    "BILINGUAL_MUSIC": "lyric",
    "BILINGUAL_CHANT": "chant",
}
OVERLAY_STYLES = {
    "OCR_TRANSLATION": "ocr",
    "EXPLANATION_NOTE": "note",
}
LEGACY_PAIRED_SOURCE_STYLES = {"English", "Default", "EN", "EN_i"}
LEGACY_PAIRED_ZH_STYLES = {"Chinese", "Default", "ZH", "ZH_i"}
OCR_LOW_VALUE_COMPOUND_RE = re.compile(r"^(?:[东西南北上下左右前后内外里中]|[东西南北](?:塔|门))$")
OCR_LOW_VALUE_SUFFIXES = ("大道", "大街", "公路", "路口", "中心", "广场")


@dataclass
class PolishTarget:
    line_index: int
    style: str
    kind: str
    start: str
    end: str
    text_field: str
    zh: str
    en: str


@dataclass
class PolishFileState:
    ass_file: Path
    lines: list[str]
    targets: list[PolishTarget]
    cache_file: Path
    cache: dict
    entries: dict
    updates: dict[int, str]
    delete_lines: set[int]
    pending: list[PolishTarget]
    stats: dict


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_hint_dicts(filename: str) -> list[dict]:
    data = load_json(_module_dir() / filename, [])
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _load_hint_strings(filename: str) -> set[str]:
    data = load_json(_module_dir() / filename, [])
    if not isinstance(data, list):
        return set()
    return {str(item).strip() for item in data if str(item).strip()}


COMMON_MISTRANSLATION_HINTS = _load_hint_dicts(COMMON_MISTRANSLATION_HINTS_FILE)
COMMON_PHRASE_CORRECTION_HINTS = _load_hint_dicts(COMMON_PHRASE_CORRECTION_HINTS_FILE)
OCR_LOW_VALUE_SHORT_TEXTS = _load_hint_strings(OCR_LOW_VALUE_SHORT_TEXTS_FILE)


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def chat_json(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


class CodexCLIClient:
    def __init__(
        self,
        command: str,
        cwd: Path,
        cache_dir: Path,
        timeout: int,
        reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
    ) -> None:
        self.command = resolve_codex_command(command)
        self.cwd = Path(cwd)
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort or DEFAULT_CODEX_REASONING_EFFORT
        self._last_request_monotonic = 0.0

    def chat_json(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ):
        wait_seconds = 1.0 - (time.monotonic() - self._last_request_monotonic)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.cache_dir / f"codex_last_{os.getpid()}_{time.time_ns()}.txt"
        prompt = "\n\n".join(
            f"{str(message.get('role', 'user')).upper()}:\n{message.get('content', '')}"
            for message in messages
        )
        prompt += "\n\nFINAL OUTPUT RULE: return one JSON object only, with no markdown or commentary."
        cmd = [
            self.command,
            "--ask-for-approval",
            "never",
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "-C",
            str(self.cwd),
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "--output-last-message",
            str(output_file),
        ]
        if model:
            cmd.extend(["-m", model])
        try:
            completed = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.timeout,
                cwd=str(self.cwd),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"codex exec timed out after {self.timeout}s") from exc
        finally:
            self._last_request_monotonic = time.monotonic()

        content = ""
        if output_file.exists():
            content = output_file.read_text(encoding="utf-8", errors="replace").strip()
            try:
                output_file.unlink()
            except OSError:
                pass
        if completed.returncode != 0:
            tail = (completed.stdout or "")[-1200:]
            raise RuntimeError(f"codex exec failed with exit={completed.returncode}: {tail}")
        if not content:
            content = extract_codex_stdout_message(completed.stdout or "")
        return {"choices": [{"message": {"content": content}}]}


def resolve_codex_command(command: str) -> str:
    command = str(command or "").strip() or "codex"
    if os.name == "nt":
        path = Path(command)
        if path.parent == Path(".") and path.name.lower() in {"codex", "codex.exe"}:
            candidates = [
                Path.home() / ".codex" / "plugins" / ".plugin-appserver" / "codex.exe",
            ]
            local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
            if local_appdata:
                bin_dir = Path(local_appdata) / "OpenAI" / "Codex" / "bin"
                if bin_dir.exists():
                    candidates.extend(sorted(bin_dir.glob("*/codex.exe"), key=lambda p: p.stat().st_mtime, reverse=True))
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate)
            exe = shutil.which("codex.exe")
            if exe and "WindowsApps" not in exe:
                return exe
    return command


def add_polish_args(parser) -> None:
    parser.add_argument(
        "--polish-provider",
        choices=["openai", "codex-cli"],
        default=os.environ.get("SUB_POLISH_PROVIDER", DEFAULT_POLISH_PROVIDER),
        help="Backend provider for --polish-backend: openai API or local codex exec CLI.",
    )
    parser.add_argument(
        "--polish-backend",
        action="store_true",
        help="Polish finished ASS Chinese text with an OpenAI-compatible backend model.",
    )
    parser.add_argument(
        "--polish-model",
        default=os.environ.get("SUB_POLISH_MODEL") or os.environ.get("OPENAI_MODEL") or "",
        help="Model name for --polish-backend. openai defaults to gpt-4.1-mini; codex-cli defaults to gpt-5.6-sol.",
    )
    parser.add_argument(
        "--polish-base-url",
        default=os.environ.get("SUB_POLISH_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
        help="OpenAI-compatible /v1 base URL for --polish-backend.",
    )
    parser.add_argument(
        "--polish-api-key",
        default=os.environ.get("SUB_POLISH_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
        help="API key for --polish-backend. Prefer SUB_POLISH_API_KEY or OPENAI_API_KEY to avoid shell history leaks.",
    )
    parser.add_argument(
        "--polish-batch-size",
        type=int,
        default=_int_env("SUB_POLISH_BATCH_SIZE", DEFAULT_POLISH_BATCH_SIZE),
        help="Number of ASS events per backend polish request. Default 48.",
    )
    parser.add_argument(
        "--polish-file-batch-size",
        type=int,
        default=_int_env("SUB_POLISH_FILE_BATCH_SIZE", DEFAULT_POLISH_FILE_BATCH_SIZE),
        help="Number of ASS files to combine per backend polish request. Default 1.",
    )
    parser.add_argument(
        "--polish-timeout",
        type=int,
        default=_int_env("SUB_POLISH_TIMEOUT", 180),
        help="Timeout in seconds for each backend polish request.",
    )
    parser.add_argument(
        "--polish-temperature",
        type=float,
        default=_float_env("SUB_POLISH_TEMPERATURE", 0.25),
        help="Backend temperature for polish requests.",
    )
    parser.add_argument(
        "--polish-styles",
        default=os.environ.get("SUB_POLISH_STYLES", DEFAULT_STYLES),
        help="Comma-separated ASS styles to polish, or 'all'.",
    )
    parser.add_argument(
        "--polish-cache-dir",
        default=os.environ.get("SUB_POLISH_CACHE_DIR", ""),
        help="Cache directory for polish results. Defaults to WORK_DIR/polish_cache.",
    )
    parser.add_argument(
        "--polish-force",
        action="store_true",
        help="Ignore existing polish cache and re-polish current ASS text.",
    )
    parser.add_argument(
        "--polish-codex-command",
        default=os.environ.get("SUB_POLISH_CODEX_COMMAND", "codex"),
        help="Command/path used for --polish-provider codex-cli.",
    )
    parser.add_argument(
        "--polish-codex-reasoning-effort",
        default=os.environ.get("SUB_POLISH_CODEX_REASONING_EFFORT", DEFAULT_CODEX_REASONING_EFFORT),
        help="Codex CLI model_reasoning_effort override for --polish-provider codex-cli.",
    )


def build_polish_client(
    provider: str,
    base_url: str,
    api_key: str,
    timeout: int,
    *,
    cache_dir: Path,
    cwd: Path,
    codex_command: str = "codex",
    codex_reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
):
    if provider == "codex-cli":
        return CodexCLIClient(codex_command, cwd, cache_dir, timeout, codex_reasoning_effort)
    if not api_key and not _is_local_base_url(base_url):
        raise RuntimeError(
            "--polish-backend needs SUB_POLISH_API_KEY/OPENAI_API_KEY or --polish-api-key "
            "when --polish-provider=openai and the base URL is not local"
        )
    return OpenAICompatibleClient(base_url, api_key, timeout)


def parse_polish_styles(value: str) -> set[str]:
    text = str(value or "").strip()
    all_styles = set(BILINGUAL_STYLES) | set(OVERLAY_STYLES) | LEGACY_PAIRED_ZH_STYLES
    if not text or text.lower() == "all":
        return all_styles
    return {item.strip() for item in text.split(",") if item.strip()} & all_styles


def polish_ass_file(
    ass_file: Path,
    base_dir: Path,
    work_dir: Path,
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    batch_size: int,
    timeout: int,
    temperature: float,
    style_names: set[str],
    cache_dir: Path | None = None,
    force: bool = False,
    codex_command: str = "codex",
    codex_reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
) -> dict:
    results = polish_ass_files(
        [ass_file],
        base_dir,
        work_dir,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        batch_size=batch_size,
        timeout=timeout,
        temperature=temperature,
        style_names=style_names,
        cache_dir=cache_dir,
        force=force,
        codex_command=codex_command,
        codex_reasoning_effort=codex_reasoning_effort,
    )
    return results.get(ass_file, make_polish_stats())


def polish_ass_files(
    ass_files: list[Path],
    base_dir: Path,
    work_dir: Path,
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    batch_size: int,
    timeout: int,
    temperature: float,
    style_names: set[str],
    cache_dir: Path | None = None,
    force: bool = False,
    codex_command: str = "codex",
    codex_reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
) -> dict[Path, dict]:
    states: list[PolishFileState] = []
    if not style_names:
        return {ass_file: make_polish_stats() for ass_file in ass_files}

    cache_dir = cache_dir or (work_dir / "polish_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    effective_model = model or (DEFAULT_MODEL if provider == "openai" else DEFAULT_CODEX_MODEL)
    model_label = effective_model
    base_label = "codex-cli" if provider == "codex-cli" else base_url

    for ass_file in ass_files:
        states.append(
            prepare_polish_file_state(
                ass_file,
                style_names,
                cache_dir,
                provider,
                model_label,
                force,
            )
        )

    pending_pairs: list[tuple[PolishFileState, PolishTarget]] = [
        (state, target)
        for state in states
        for target in state.pending
    ]
    if pending_pairs:
        client = build_polish_client(
            provider,
            base_url,
            api_key,
            timeout,
            cache_dir=cache_dir,
            cwd=base_dir,
            codex_command=codex_command,
            codex_reasoning_effort=codex_reasoning_effort,
        )
        effective_batch_size = len(pending_pairs) if batch_size <= 0 else max(batch_size, 1)
        for batch_offset in range(0, len(pending_pairs), effective_batch_size):
            batch_pairs = pending_pairs[batch_offset : batch_offset + effective_batch_size]
            terminology = load_relevant_terminology(base_dir, batch_pairs)
            mistranslation_hints = select_mistranslation_hints([target for _, target in batch_pairs])
            phrase_hints = select_phrase_correction_hints([target for _, target in batch_pairs])
            result = request_polish_batch(
                client,
                effective_model,
                [
                    (state.ass_file.name, target)
                    for state, target in batch_pairs
                ],
                terminology,
                mistranslation_hints,
                phrase_hints,
                temperature,
            )
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            counted_batch_files: set[Path] = set()
            for state, _ in batch_pairs:
                if state.ass_file in counted_batch_files:
                    continue
                counted_batch_files.add(state.ass_file)
                state.stats["batches"] += 1
            if result is None:
                for state, _target in batch_pairs:
                    state.stats["requested"] += 1
                    state.stats["failed"] += 1
                continue
            for request_id, (state, target) in enumerate(batch_pairs, start=1):
                state.stats["requested"] += 1
                apply_polish_result(
                    state,
                    target,
                    result.get(request_id),
                    provider,
                    model_label,
                    now,
                )

    for state in states:
        finalize_polish_file_state(state, style_names, provider, model_label, base_label)
    return {state.ass_file: state.stats for state in states}


def make_polish_stats() -> dict:
    stats = {
        "targets": 0,
        "requested": 0,
        "cache_hits": 0,
        "already_polished": 0,
        "unchanged": 0,
        "deleted": 0,
        "changed": 0,
        "failed": 0,
        "batches": 0,
    }
    return stats


def prepare_polish_file_state(
    ass_file: Path,
    style_names: set[str],
    cache_dir: Path,
    provider: str,
    model_label: str,
    force: bool,
) -> PolishFileState:
    stats = make_polish_stats()
    lines = ass_file.read_text(encoding="utf-8-sig").splitlines()
    targets = extract_polish_targets(lines, style_names)
    stats["targets"] = len(targets)
    cache_file = cache_dir / f"{ass_file.stem}.json"
    cache = load_polish_cache(cache_file)
    entries = cache.setdefault("entries", {})

    updates: dict[int, str] = {}
    delete_lines: set[int] = set()
    pending: list[PolishTarget] = []
    for target in targets:
        key = cache_key(provider, model_label, target)
        cached = entries.get(key)
        if target.kind == "ocr" and ocr_is_noise_fragment(target.zh):
            entries[key] = polish_cache_entry(provider, model_label, target, "", time.strftime("%Y-%m-%d %H:%M:%S"), delete=True)
            delete_lines.add(target.line_index)
            stats["deleted"] += 1
            continue
        if not force and isinstance(cached, dict) and cached.get("delete") is True and target.style in OVERLAY_STYLES:
            delete_lines.add(target.line_index)
            stats["cache_hits"] += 1
            continue
        if not force and isinstance(cached, dict) and cached.get("polished_zh"):
            updates[target.line_index] = str(cached["polished_zh"])
            stats["cache_hits"] += 1
            continue
        if not force and is_already_polished(target, entries, provider, model_label):
            stats["already_polished"] += 1
            continue
        pending.append(target)
    return PolishFileState(
        ass_file=ass_file,
        lines=lines,
        targets=targets,
        cache_file=cache_file,
        cache=cache,
        entries=entries,
        updates=updates,
        delete_lines=delete_lines,
        pending=pending,
        stats=stats,
    )


def apply_polish_result(
    state: PolishFileState,
    target: PolishTarget,
    raw_zh,
    provider: str,
    model_label: str,
    now: str,
) -> None:
    key = cache_key(provider, model_label, target)
    if raw_zh is None:
        if target.en and not has_chinese(target.zh):
            state.stats["failed"] += 1
            return
        state.entries[key] = polish_cache_entry(provider, model_label, target, target.zh, now)
        state.stats["unchanged"] += 1
        return
    if raw_zh == DELETE_SENTINEL:
        if target.style in OVERLAY_STYLES:
            state.entries[key] = polish_cache_entry(provider, model_label, target, "", now, delete=True)
            state.delete_lines.add(target.line_index)
            state.stats["deleted"] += 1
        else:
            state.stats["failed"] += 1
        return
    clean_zh = clean_polished_zh(raw_zh, target)
    if not clean_zh:
        state.stats["failed"] += 1
        return
    if target.kind == "ocr" and ocr_is_noise_fragment(clean_zh):
        state.entries[key] = polish_cache_entry(provider, model_label, target, "", now, delete=True)
        state.delete_lines.add(target.line_index)
        state.stats["deleted"] += 1
        return
    state.entries[key] = polish_cache_entry(provider, model_label, target, clean_zh, now)
    state.updates[target.line_index] = clean_zh


def finalize_polish_file_state(
    state: PolishFileState,
    style_names: set[str],
    provider: str,
    model_label: str,
    base_label: str,
) -> None:
    normalized_lines = 0
    for target in state.targets:
        if target.line_index in state.delete_lines:
            continue
        zh = normalize_series_terms(state.updates.get(target.line_index, target.zh), target.en)
        new_line = replace_target_zh(state.lines[target.line_index], target, zh)
        if new_line and new_line != state.lines[target.line_index]:
            state.lines[target.line_index] = new_line
            normalized_lines += 1
            if target.line_index in state.updates:
                state.stats["changed"] += 1
    normalized_lines += normalize_bilingual_presentation(state.lines, style_names)
    if state.delete_lines:
        state.lines = [line for index, line in enumerate(state.lines) if index not in state.delete_lines]
    if normalized_lines or state.delete_lines:
        state.ass_file.write_text("\n".join(state.lines) + "\n", encoding="utf-8-sig")

    save_polish_cache(state.cache_file, state.cache, provider, model_label, base_label)


def polish_cache_entry(
    provider: str,
    model: str,
    target: PolishTarget,
    polished_zh: str,
    updated_at: str,
    *,
    delete: bool = False,
) -> dict:
    return {
        "provider": provider,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "style": target.style,
        "kind": target.kind,
        "en": target.en,
        "source_zh": target.zh,
        "polished_zh": polished_zh,
        "delete": bool(delete),
        "updated_at": updated_at,
    }


def normalize_bilingual_presentation(lines: list[str], style_names: set[str]) -> int:
    if not (style_names & set(BILINGUAL_STYLES)):
        return 0
    changed = 0
    for index, line in enumerate(lines):
        parts = split_dialogue(line)
        if not parts or parts[3].strip() not in BILINGUAL_STYLES:
            continue
        original_text = parts[9]
        split = split_bilingual_text(original_text)
        if split:
            zh_fragment, en_fragment = split
            zh = strip_presentation_tags(zh_fragment).strip()
            en = with_english_size_tag(en_fragment)
            new_text = zh + r"\N" + en if en else zh
        else:
            new_text = strip_presentation_tags(original_text).strip()
        if new_text != original_text:
            parts[9] = new_text
            lines[index] = ",".join(parts)
            changed += 1
    return changed


def extract_polish_targets(lines: list[str], style_names: set[str]) -> list[PolishTarget]:
    targets: list[PolishTarget] = []
    for line_index, line in enumerate(lines):
        parts = split_dialogue(line)
        if not parts:
            continue
        style = parts[3]
        if style not in style_names:
            continue
        kind = BILINGUAL_STYLES.get(style) or OVERLAY_STYLES.get(style)
        if not kind:
            continue
        text_field = parts[9]
        if style in BILINGUAL_STYLES:
            split = split_bilingual_text(text_field)
            if not split:
                continue
            zh_fragment, en_fragment = split
            zh = visible_ass_text(zh_fragment)
            en = visible_ass_text(en_fragment)
        else:
            zh = visible_ass_text(text_field)
            en = ""
        missing_zh_translation = bool(
            style in BILINGUAL_STYLES
            and en
            and not has_chinese(zh)
            and not re.search(r"[0-9A-Za-z]", zh)
        )
        if not has_chinese(zh) and not missing_zh_translation:
            continue
        targets.append(
            PolishTarget(
                line_index=line_index,
                style=style,
                kind=kind,
                start=parts[1],
                end=parts[2],
                text_field=text_field,
                zh=zh,
                en=en,
            )
        )
    targets.extend(extract_legacy_paired_targets(lines, style_names))
    return sorted(targets, key=lambda item: item.line_index)


def extract_legacy_paired_targets(lines: list[str], style_names: set[str]) -> list[PolishTarget]:
    """Pair legacy separate English/Chinese events with identical timestamps.

    Some finished bilingual ASS files keep the two languages as adjacent events
    instead of a single ``zh\\Nen`` event. Only exact-time pairs are accepted so
    unrelated overlays cannot become dialogue context. When a malformed group
    contains extra fragments, the nearest (last) events are preferred.
    """
    groups: dict[tuple[str, str], dict[str, list[tuple[int, list[str], str]]]] = {}
    enabled_zh_styles = LEGACY_PAIRED_ZH_STYLES & style_names
    if not enabled_zh_styles:
        return []

    for line_index, line in enumerate(lines):
        parts = split_dialogue(line)
        if not parts:
            continue
        style = parts[3]
        if style not in (LEGACY_PAIRED_SOURCE_STYLES | enabled_zh_styles):
            continue
        visible = visible_ass_text(parts[9])
        if not visible:
            continue
        group = groups.setdefault((parts[1], parts[2]), {"sources": [], "targets": []})
        if style in enabled_zh_styles and has_chinese(visible):
            group["targets"].append((line_index, parts, visible))
        elif style in LEGACY_PAIRED_SOURCE_STYLES and LATIN_RE.search(visible):
            group["sources"].append((line_index, parts, visible))

    targets: list[PolishTarget] = []
    for group in groups.values():
        pair_count = min(len(group["sources"]), len(group["targets"]))
        if pair_count <= 0:
            continue
        sources = group["sources"][-pair_count:]
        zh_events = group["targets"][-pair_count:]
        for (_source_index, _source_parts, en), (line_index, parts, zh) in zip(sources, zh_events):
            visible_pair = f"{zh} {en}"
            if CHANT_SYMBOL in visible_pair:
                kind = "chant"
            elif MUSIC_SYMBOL in visible_pair:
                kind = "lyric"
            else:
                kind = "dialogue"
            targets.append(
                PolishTarget(
                    line_index=line_index,
                    style=parts[3],
                    kind=kind,
                    start=parts[1],
                    end=parts[2],
                    text_field=parts[9],
                    zh=zh,
                    en=en,
                )
            )
    return targets


def request_polish_batch(
    client: OpenAICompatibleClient,
    model: str,
    batch: list[tuple[str, PolishTarget]],
    terminology: list[dict],
    mistranslation_hints: list[dict],
    phrase_hints: list[dict],
    temperature: float,
) -> dict[int, str] | None:
    payload = {
        "terminology": terminology,
        "mistranslation_hints": mistranslation_hints,
        "phrase_hints": phrase_hints,
        "items": [
            {
                "id": index,
                "file": ass_name,
                "kind": target.kind,
                "time": f"{target.start}-{target.end}",
                "source_en": target.en,
                "current_zh": target.zh,
                "needs_translation": bool(target.en and not has_chinese(target.zh)),
            }
            for index, (ass_name, target) in enumerate(batch, start=1)
        ],
    }
    system_prompt = (
        "You are a senior Simplified Chinese subtitle editor for finished TV subtitles. "
        "Polish the existing Chinese text, using the English source when present. "
        "Return strict JSON only: {\"items\":[{\"id\":1,\"zh\":\"...\"}]}. "
        "To save tokens, return only items whose Chinese text should change; omit ids that are already good. "
        "If needs_translation=true, current_zh is missing or only punctuation: translate source_en into concise "
        "Simplified Chinese and always return that id; never omit it. "
        "For OCR overlays or cultural notes only, if an item is useless OCR garbage, duplicated clutter, "
        "or an unnecessary/incorrect note, return {\"id\":1,\"delete\":true}; never delete dialogue or lyrics. "
        "Do not change ids, timing, style, speaker order, or meaning. "
        "Use natural spoken Chinese for dialogue, concise lyric Chinese for lyrics, "
        "compact text for OCR overlays, and informative but concise wording for cultural notes. "
        "For OCR overlays, prefer the shortest useful wording. "
        "For cultural notes, keep enough background to explain why the reference matters, "
        "usually about 36-60 Chinese characters, without becoming encyclopedic. "
        "If an OCR/note looks like repeated auxiliary context, keep the wording stable "
        "so the pipeline can de-duplicate later repeats cleanly. "
        "Fix translationese, stiff wording, wrong tone, and awkward literal phrasing. "
        "When current_zh contains literal or non-native phrasing, mistranslated collocations, "
        "or obvious OCR/MT artifacts, rewrite freely into natural Simplified Chinese rather than "
        "preserving word-for-word structure. "
        "Do not preserve bizarre compounds or semi-translated fragments just because some source "
        "words are visible; prefer the shortest natural Chinese a subtitle editor would actually use. "
        "For colloquial dialogue, jokes, ad copy, venue/show packaging, and spoken banter, prefer "
        "idiomatic Chinese over literal lexical matching. "
        "When terminology contains a matched entry, preferred_zh is the literal preferred subtitle term; "
        "the optional note only explains when to use it and must never be copied into a subtitle. "
        "Mistranslation_hints and phrase_hints contain reusable guidance, not replacement text; follow the "
        "guidance in context and never copy an entire guidance sentence into the subtitle. "
        "When phrase_hints contains a matched colloquial or industry phrase, treat its guidance as stronger guidance "
        "than word-by-word translation. "
        "When a line still reads like translated English syntax, rewrite the whole sentence into fluent "
        "spoken Chinese instead of preserving the original clause order. "
        "Preserve proper nouns and culture-specific terms unless a better established Chinese form exists. "
        "Do not add explanations, brackets, markdown, ASS tags, or extra line breaks. "
        "Keep each line concise enough for subtitles. Use Simplified Chinese only."
    )
    user_prompt = (
        "Polish current_zh for each item. Return only changed items; omitted ids keep current_zh, except every "
        "needs_translation=true item must be returned with a Chinese translation. "
        "For kind=ocr or kind=note, use delete=true only when the overlay should be removed from the finished ASS. "
        "Terminology, mistranslation_hints, and phrase_hints are optional context; use them only when relevant. "
        "Only terminology.preferred_zh is literal replacement wording; note and guidance fields are instructions.\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    debug_dir = os.environ.get(DEBUG_POLISH_DIR_ENV, "").strip()
    try:
        response = client.chat_json(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max(700, len(batch) * 45),
        )
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = parse_json_response(content)
    except Exception as exc:
        if debug_dir:
            write_polish_debug_artifact(debug_dir, payload, user_prompt, "", {}, error=str(exc))
        message = str(exc).encode("ascii", "backslashreplace").decode("ascii")
        print(f"  [warning] polish backend request failed: {message}")
        return None

    if debug_dir:
        write_polish_debug_artifact(debug_dir, payload, user_prompt, content, parsed)

    raw_items = parsed.get("items") if isinstance(parsed, dict) else parsed
    result: dict[int, str] = {}
    terminology_notes = {
        normalize_for_cache(item.get("note", ""))
        for item in terminology
        if isinstance(item, dict) and normalize_for_cache(item.get("note", ""))
    }
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                item_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            zh = item.get("zh")
            if isinstance(zh, str):
                if normalize_for_cache(zh) in terminology_notes:
                    continue
                result[item_id] = zh
            elif item.get("delete") is True:
                result[item_id] = DELETE_SENTINEL
    return result


def write_polish_debug_artifact(
    debug_dir: str | Path,
    payload: dict,
    user_prompt: str,
    raw_content: str,
    parsed,
    *,
    error: str = "",
) -> None:
    try:
        directory = Path(debug_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000:06d}"
        path = directory / f"polish_debug_{stamp}.json"
        artifact = {
            "payload": payload,
            "user_prompt": user_prompt,
            "raw_content": raw_content,
            "parsed": parsed,
            "error": error,
        }
        path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def replace_target_zh(line: str, target: PolishTarget, zh: str) -> str:
    parts = split_dialogue(line)
    if not parts:
        return line
    if target.style in BILINGUAL_STYLES:
        replaced = replace_bilingual_zh(parts[9], zh)
    elif target.style == "OCR_TRANSLATION":
        replaced = replace_leading_tagged_text(parts[9], break_zh(zh, max_chars=18))
    elif target.style == "EXPLANATION_NOTE":
        replaced = replace_leading_tagged_text(parts[9], break_zh(zh, max_chars=24))
    elif target.style in LEGACY_PAIRED_ZH_STYLES:
        replaced = leading_tags(parts[9]) + ass_escape(zh)
    else:
        return line
    parts[9] = replaced
    return ",".join(parts)


def split_dialogue(line: str) -> list[str] | None:
    if not line.startswith("Dialogue:"):
        return None
    parts = line.split(",", 9)
    return parts if len(parts) == 10 else None


def split_bilingual_text(text: str) -> tuple[str, str] | None:
    pieces = str(text or "").split(r"\N")
    en_index = None
    for index, piece in enumerate(pieces):
        if r"\fnArial" in piece:
            en_index = index
            break
    if en_index is None and len(pieces) >= 2:
        for index in range(1, len(pieces)):
            visible = visible_ass_text(r"\N".join(pieces[index:]))
            if visible and not has_chinese(visible):
                en_index = index
                break
    if en_index is None or en_index <= 0:
        return None
    return r"\N".join(pieces[:en_index]), r"\N".join(pieces[en_index:])


def visible_ass_text(text: str) -> str:
    text = ASS_TAG_RE.sub("", str(text or ""))
    text = normalize_music_symbols(text)
    text = text.replace(r"\N", "\n").replace("\r\n", "\n").replace("\r", "\n")
    parts = [part.strip() for part in text.split("\n") if part.strip()]
    return CJK_INTERNAL_SPACE_RE.sub("", " ".join(parts).strip())


def replace_bilingual_zh(text: str, zh: str) -> str:
    split = split_bilingual_text(text)
    if not split:
        return text
    zh_fragment, en_fragment = split
    visible_zh = visible_ass_text(zh)
    display_zh = (
        break_zh(zh, max_chars=BILINGUAL_ZH_WRAP_CHARS)
        if len(re.sub(r"\s+", "", visible_zh)) > MAX_BILINGUAL_ZH_CHARS
        else zh
    )
    return ass_escape(display_zh) + r"\N" + with_english_size_tag(en_fragment)


def replace_leading_tagged_text(text: str, zh: str) -> str:
    return strip_presentation_tags(leading_tags(text)) + ass_escape(zh)


def leading_tags(text: str) -> str:
    match = LEADING_ASS_TAGS_RE.match(str(text or ""))
    return match.group(1) if match else ""


def strip_presentation_tags(text: str) -> str:
    def clean_tag(match: re.Match) -> str:
        body = match.group(0)[1:-1]
        for pattern in PRESENTATION_TAG_COMMANDS:
            body = pattern.sub("", body)
        return "{" + body + "}" if body.strip() else ""

    return normalize_music_symbols(ASS_TAG_RE.sub(clean_tag, str(text or "")))


def with_english_size_tag(text: str) -> str:
    cleaned = strip_presentation_tags(text).strip()
    if not cleaned:
        return ""
    return EN_SIZE_TAG + cleaned


def normalize_music_symbols(text: str) -> str:
    return MOJIBAKE_MUSIC_MARK_RE.sub(MUSIC_SYMBOL, str(text or ""))


def normalize_series_terms(text: str, source_en: str = "") -> str:
    text = str(text or "")
    en = str(source_en or "").lower()
    if any(term in en for term in ("raina", "rayna")):
        text = re.sub(r"(?:蕾娜|瑞娜|莱娜|雷娜)[·・](?:詹姆斯|杰姆斯|杰恩斯|简)", "蕾娜·詹姆斯", text)
        text = re.sub(r"(?:瑞娜|莱娜|雷娜)", "蕾娜", text)
    if "fordham" in en or "jeff ford" in en:
        text = re.sub(r"杰夫[·・]?(?:福特汉姆|福特汉|福德汉|福德姆|福特姆|福特)", "杰夫·福特汉姆", text)
        text = re.sub(r"(?<!杰夫[·・])福(?:特|德)(?:汉姆|汉|德姆|特姆|德汉|特哈姆)?", "福特汉姆", text)
    if "edge hill" in en or "edgehill" in en:
        text = re.sub(r"Edge\s*Hill|Edgehill", "埃奇希尔", text, flags=re.IGNORECASE)
        text = re.sub(r"(?:埃奇山|伊奇山|边缘山|边希尔)", "埃奇希尔", text)
    if "scarlett" in en or "scarlet" in en:
        text = re.sub(r"Scarlett\s+O['’]Connor|Scarlet\s+O['’]Connor", "斯嘉丽·奥康纳", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<![A-Za-z])(?:Scarlett|Scarlet)(?![A-Za-z])", "斯嘉丽", text, flags=re.IGNORECASE)
    if "juliette" in en or "juliet" in en:
        text = re.sub(r"Juliette\s+Barnes|Juliet\s+Barnes", "朱丽叶·巴恩斯", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<![A-Za-z])(?:Juliette|Juliet)(?![A-Za-z])", "朱丽叶", text, flags=re.IGNORECASE)
    if "gunnar" in en or "gunner" in en:
        text = re.sub(r"(?:古纳|刚纳)", "冈纳", text)
    if "avery" in en:
        text = re.sub(r"艾弗里", "埃弗里", text)
    if any(term in en for term in ("deacon", "dakin", "beacon", "dickon")):
        text = re.sub(r"(?:戴肯|德肯|戴克|德克)(?=[，。？！、\\s]|$)", "迪肯", text)
        text = re.sub(r"(?:戴肯|德肯|戴克|德克)[·・](?:克莱伯恩|克莱布尔)", "迪肯·克莱伯恩", text)
    return text


def clean_polished_zh(value, target: PolishTarget) -> str:
    if not isinstance(value, str):
        return ""
    text = THINK_RE.sub("", value)
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    text = ASS_TAG_RE.sub("", text)
    text = normalize_music_symbols(text)
    text = text.replace(r"\N", "\n").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("{", "(").replace("}", ")")
    parts = [part.strip() for part in text.split("\n") if part.strip()]
    text = re.sub(r"\s+", " ", " ".join(parts)).strip().strip("\"'")
    text = preserve_style_marks(text, target)
    text = normalize_series_terms(text, target.en)
    if not text:
        return ""
    if has_chinese(target.zh) and not has_chinese(text):
        return ""
    if target.en and not has_chinese(target.zh) and not has_chinese(text):
        return ""
    # Guard against accidental explanations or whole-batch echoes.
    max_reasonable = max(len(target.zh) * 3 + 12, 80)
    if target.kind == "ocr":
        max_reasonable = max(len(target.zh) * 3 + 12, 64)
    elif target.kind == "note":
        max_reasonable = max(len(target.zh) * 4 + 24, 110)
    if len(text) > max_reasonable:
        return ""
    return text


def preserve_style_marks(text: str, target: PolishTarget) -> str:
    stripped = text.strip()
    original = target.zh.strip()
    if target.style == "BILINGUAL_MUSIC" and original.startswith(MUSIC_SYMBOL) and original.endswith(MUSIC_SYMBOL):
        core = stripped.strip(MUSIC_SYMBOL).strip()
        return f"{MUSIC_SYMBOL} {core} {MUSIC_SYMBOL}" if core else stripped
    if target.style == "BILINGUAL_CHANT" and original.startswith(CHANT_SYMBOL) and original.endswith(CHANT_SYMBOL):
        core = stripped.strip(CHANT_SYMBOL).strip()
        return f"{CHANT_SYMBOL} {core} {CHANT_SYMBOL}" if core else stripped
    return stripped


def parse_json_response(content: str):
    cleaned = THINK_RE.sub("", content or "").strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {}


def cache_key(provider: str, model: str, target: PolishTarget) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "provider": provider,
        "model": model,
        "style": target.style,
        "kind": target.kind,
        "en": normalize_for_cache(target.en),
        "zh": normalize_for_cache(target.zh),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_for_cache(text: str) -> str:
    text = re.sub(r"\s+", " ", normalize_music_symbols(str(text or "").replace(r"\N", " "))).strip()
    return CJK_INTERNAL_SPACE_RE.sub("", text)


def hint_guidance_text(item: dict) -> str:
    return str(item.get("guidance") or item.get("preferred_zh") or item.get("zh") or item.get("note") or "").strip()


def select_mistranslation_hints(targets: list[PolishTarget], limit: int = 24) -> list[dict]:
    haystack = "\n".join(f"{target.en}\n{target.zh}" for target in targets).lower()
    selected: list[dict] = []
    for item in COMMON_MISTRANSLATION_HINTS:
        term = str(item.get("term") or item.get("source_en") or "").strip()
        guidance = hint_guidance_text(item)
        if not term or not guidance:
            continue
        if term_matches_text(term, haystack):
            hint = {"term": term, "guidance": guidance}
            bad_zh = str(item.get("bad_zh") or "").strip()
            if bad_zh:
                hint["bad_zh"] = bad_zh
            selected.append(hint)
            if len(selected) >= limit:
                break
    return selected


def select_phrase_correction_hints(targets: list[PolishTarget], limit: int = 24) -> list[dict]:
    haystack = "\n".join(f"{target.en}\n{target.zh}" for target in targets).lower()
    selected: list[dict] = []
    for item in COMMON_PHRASE_CORRECTION_HINTS:
        phrase = str(item.get("phrase") or "").strip()
        guidance = hint_guidance_text(item)
        if not phrase or not guidance:
            continue
        if term_matches_text(phrase, haystack):
            hint = {"phrase": phrase, "guidance": guidance}
            bad_zh = str(item.get("bad_zh") or "").strip()
            if bad_zh:
                hint["bad_zh"] = bad_zh
            selected.append(hint)
            if len(selected) >= limit:
                break
    return selected


def ocr_is_noise_fragment(text: str) -> bool:
    normalized = normalize_for_cache(text)
    if not normalized:
        return True
    bare = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", normalized)
    if not bare:
        return True
    han_count = len(HAN_CHAR_RE.findall(bare))
    latin_count = len(re.findall(r"[A-Za-z]", bare))
    digit_count = len(re.findall(r"\d", bare))
    if len(bare) <= 1:
        return True
    if han_count and not latin_count and not digit_count:
        if bare in OCR_LOW_VALUE_SHORT_TEXTS or OCR_LOW_VALUE_COMPOUND_RE.fullmatch(bare):
            return True
        if han_count == 1:
            return True
        if han_count <= 3 and any(bare.endswith(suffix) for suffix in OCR_LOW_VALUE_SUFFIXES):
            return True
    if latin_count and not han_count:
        if len(bare) <= 2:
            return True
        letters_only = re.sub(r"[^A-Za-z]", "", bare)
        if letters_only:
            vowels = len(re.findall(r"[AEIOUaeiou]", letters_only))
            if len(letters_only) >= 4 and vowels == 0:
                return True
            if len(letters_only) <= 4 and vowels <= 1:
                return True
    if digit_count and not han_count and latin_count == 0 and len(bare) <= 3:
        return True
    return False


def term_matches_text(term: str, text: str) -> bool:
    term = normalize_for_cache(term)
    haystack = normalize_for_cache(text)
    if not term or not haystack:
        return False
    pattern = re.escape(term)
    pattern = re.sub(r"\\\s+", r"\\s+", pattern)
    return bool(re.search(rf"(?<![0-9A-Za-z]){pattern}(?![0-9A-Za-z])", haystack, re.IGNORECASE))


def is_already_polished(target: PolishTarget, entries: dict, provider: str, model: str) -> bool:
    target_zh = normalize_for_cache(target.zh)
    target_en = normalize_for_cache(target.en)
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("provider", "openai") != provider:
            continue
        if entry.get("model") != model:
            continue
        if entry.get("prompt_version") != PROMPT_VERSION:
            continue
        if entry.get("style") != target.style or entry.get("kind") != target.kind:
            continue
        if normalize_for_cache(entry.get("en", "")) != target_en:
            continue
        if normalize_for_cache(entry.get("polished_zh", "")) == target_zh:
            return True
    return False


def load_polish_cache(path: Path) -> dict:
    data = load_json(path, {})
    if not isinstance(data, dict):
        return {"entries": {}}
    if not isinstance(data.get("entries"), dict):
        data["entries"] = {}
    return data


def save_polish_cache(path: Path, cache: dict, provider: str, model: str, base_url: str) -> None:
    cache["provider"] = provider
    cache["model"] = model
    cache["base_url"] = base_url
    cache["prompt_version"] = PROMPT_VERSION
    cache["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def request_static_hint_review(
    client,
    model: str,
    payload: dict,
    temperature: float,
) -> dict:
    system_prompt = (
        "You are maintaining reusable Simplified Chinese subtitle correction dictionaries. "
        "Return strict JSON only: "
        "{\"mistranslation_hints\":[{\"term\":\"...\",\"source_en\":\"...\",\"bad_zh\":\"...\","
        "\"guidance\":\"...\",\"style_scope\":\"dialogue|lyric|ocr|note|all\","
        "\"confidence\":\"high|medium|low\",\"support_count\":1}],"
        "\"phrase_correction_hints\":[{\"phrase\":\"...\",\"source_en\":\"...\",\"bad_zh\":\"...\","
        "\"guidance\":\"...\",\"style_scope\":\"dialogue|lyric|all\","
        "\"confidence\":\"high|medium|low\",\"support_count\":1}],"
        "\"ocr_low_value_short_texts\":[\"...\"]}. "
        "Add only high-confidence, reusable rules that are likely to help future episodes. "
        "Use mistranslation_hints for proper nouns, fixed collocations, industry terms, and title/technical wording. "
        "Use phrase_correction_hints for frequent colloquial or industry phrases where the natural Chinese guidance is reusable. "
        "Do not add series-specific character names, fictional places, supernatural terms, or other exact canon wording "
        "to either common dictionary; those belong in subtitle_terminology.json and require explicit series scope. "
        "Never duplicate anything listed in existing.terminology_terms. "
        "The guidance field is an instruction for a future editor, not literal replacement text. "
        "For short colloquial phrases, provide several natural Chinese options in guidance when context decides the final wording. "
        "Set style_scope carefully: dialogue for spoken lines, lyric only for song-lyric wording, ocr for screen-text cleanup, all only when truly universal. "
        "Use bad_zh only for a recurring wrong translation pattern, not for every source line. "
        "Use ocr_low_value_short_texts only for short or low-value Chinese OCR fragments that should usually be deleted when shown alone. "
        "Do not repeat existing entries. Prefer a few precise additions over broad or risky guesses. "
        "Simplified Chinese only. No markdown or commentary."
    )
    user_prompt = "Review these cache-derived samples and propose dictionary additions:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    try:
        response = client.chat_json(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1600,
        )
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = parse_json_response(content)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        print(f"  [warning] static hint review failed: {exc}")
        return {}


def _clean_hint_support_count(value) -> int:
    try:
        return max(int(value or 1), 1)
    except (TypeError, ValueError):
        return 1


def _hint_preferred_text(item: dict) -> str:
    return clean_glossary_review_text(item.get("guidance") or item.get("preferred_zh") or item.get("zh"))


def _normalize_hint_dict(item: dict, key_field: str) -> dict | None:
    key_text = clean_glossary_review_text(item.get(key_field) or item.get("source_en") or item.get("term") or item.get("phrase"))
    value = _hint_preferred_text(item)
    if not key_text or not value:
        return None
    normalized = {key_field: key_text, "guidance": value}
    for field in ("source_en", "bad_zh", "style_scope", "confidence", "note"):
        field_value = clean_glossary_review_text(item.get(field))
        if field_value:
            normalized[field] = field_value
    normalized["support_count"] = _clean_hint_support_count(item.get("support_count") or item.get("count"))
    return normalized


def _merge_hint_record(existing: dict, addition: dict) -> None:
    existing["support_count"] = min(
        _clean_hint_support_count(existing.get("support_count")) + _clean_hint_support_count(addition.get("support_count")),
        99,
    )
    if addition.get("confidence") == "high" and existing.get("confidence") != "high":
        existing["confidence"] = "high"
    for field in ("source_en", "bad_zh", "guidance", "style_scope", "note"):
        if addition.get(field) and not existing.get(field):
            existing[field] = addition[field]


def _merge_hint_dicts(existing: list[dict], additions: list[dict], key_field: str) -> tuple[list[dict], int]:
    merged: list[dict] = []
    index_by_key: dict[str, int] = {}
    added = 0
    for source_name, items in (("existing", existing), ("additions", additions)):
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_hint_dict(item, key_field)
            if not normalized:
                continue
            key = normalize_for_cache(normalized.get(key_field, "")).lower()
            if not key:
                continue
            if key in index_by_key:
                _merge_hint_record(merged[index_by_key[key]], normalized)
                continue
            index_by_key[key] = len(merged)
            merged.append(normalized)
            if source_name == "additions":
                added += 1
    return merged, added


def _merge_hint_strings(existing: set[str], additions: list) -> tuple[list[str], int]:
    merged = set(existing)
    added = 0
    for item in additions:
        value = clean_glossary_review_text(item)
        if not value or value in merged:
            continue
        merged.add(value)
        added += 1
    return sorted(merged), added


def _collect_static_hint_samples(cache_files: list[Path]) -> dict:
    changed_items: list[dict] = []
    deleted_ocr: list[str] = []
    for path in cache_files:
        data = load_json(path, {})
        entries = data.get("entries") if isinstance(data, dict) else {}
        if not isinstance(entries, dict):
            continue
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("prompt_version") != PROMPT_VERSION:
                continue
            kind = str(entry.get("kind") or "").strip()
            source_zh = clean_glossary_review_text(entry.get("source_zh"))
            polished_zh = clean_glossary_review_text(entry.get("polished_zh"))
            en = clean_glossary_review_text(entry.get("en"))
            if kind == "ocr" and entry.get("delete") is True and source_zh and has_chinese(source_zh) and len(source_zh) <= 8:
                deleted_ocr.append(source_zh)
            if polished_zh and source_zh and polished_zh != source_zh:
                changed_items.append(
                    {
                        "kind": kind,
                        "en": en,
                        "source_zh": source_zh,
                        "polished_zh": polished_zh,
                    }
                )
    unique_deleted = sorted({item for item in deleted_ocr})[:80]
    unique_changed: list[dict] = []
    seen_changed: set[str] = set()
    for item in changed_items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen_changed:
            continue
        seen_changed.add(key)
        unique_changed.append(item)
        if len(unique_changed) >= 120:
            break
    return {"changed_items": unique_changed, "deleted_ocr": unique_deleted}


def curate_static_hint_files(
    cache_files: list[Path],
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    temperature: float,
    cache_dir: Path,
    cwd: Path,
    codex_command: str = "codex",
    codex_reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
) -> dict[str, int]:
    stats = {
        "samples": 0,
        "mistranslation_added": 0,
        "phrase_added": 0,
        "ocr_low_value_added": 0,
        "failed": 0,
        "skipped": 0,
    }
    samples = _collect_static_hint_samples(cache_files)
    stats["samples"] = len(samples["changed_items"]) + len(samples["deleted_ocr"])
    if not stats["samples"]:
        return stats

    effective_model = model or (DEFAULT_MODEL if provider == "openai" else DEFAULT_CODEX_MODEL)
    effective_reasoning = ""
    if provider == "codex-cli":
        effective_reasoning = codex_reasoning_effort or DEFAULT_CODEX_REASONING_EFFORT
    review_identity = f"{provider}:{effective_model}:{effective_reasoning}:{STATIC_HINT_REVIEW_PROMPT_VERSION}"
    review_digest_payload = {
        "provider": provider,
        "model": effective_model,
        "reasoning_effort": effective_reasoning,
        "prompt_version": STATIC_HINT_REVIEW_PROMPT_VERSION,
        "samples": samples,
    }
    review_digest = hashlib.sha256(
        json.dumps(review_digest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    review_state_file = cache_dir / "static_hint_review_state.json"
    review_state = load_json(review_state_file, {})
    if not isinstance(review_state, dict):
        review_state = {}
    review_entries = review_state.setdefault("entries", {})
    previous_review = review_entries.get(review_identity) if isinstance(review_entries, dict) else None
    if isinstance(previous_review, dict) and previous_review.get("digest") == review_digest:
        stats["skipped"] = 1
        return stats
    if not isinstance(review_entries, dict):
        review_entries = {}
        review_state["entries"] = review_entries

    terminology_data = load_json(cwd / SUBTITLE_TERMINOLOGY_FILE, {})
    terminology_entries = terminology_data.get("entries") if isinstance(terminology_data, dict) else []
    if not isinstance(terminology_entries, list):
        terminology_entries = []
    payload = {
        "prompt_version": STATIC_HINT_REVIEW_PROMPT_VERSION,
        "existing": {
            "mistranslation_terms": [item.get("term") for item in COMMON_MISTRANSLATION_HINTS if isinstance(item, dict)],
            "phrase_corrections": [item.get("phrase") for item in COMMON_PHRASE_CORRECTION_HINTS if isinstance(item, dict)],
            "terminology_terms": [
                item.get("source_en")
                for item in terminology_entries
                if isinstance(item, dict) and item.get("source_en")
            ],
            "ocr_low_value_short_texts": sorted(OCR_LOW_VALUE_SHORT_TEXTS),
        },
        "samples": samples,
    }
    client = build_polish_client(
        provider,
        base_url,
        api_key,
        timeout,
        cache_dir=cache_dir,
        cwd=cwd,
        codex_command=codex_command,
        codex_reasoning_effort=codex_reasoning_effort,
    )
    result = request_static_hint_review(client, effective_model, payload, temperature)
    if not result:
        stats["failed"] += 1
        return stats

    mistranslation_existing = load_json(_module_dir() / COMMON_MISTRANSLATION_HINTS_FILE, [])
    phrase_existing = load_json(_module_dir() / COMMON_PHRASE_CORRECTION_HINTS_FILE, [])
    ocr_existing = _load_hint_strings(OCR_LOW_VALUE_SHORT_TEXTS_FILE)

    mistranslation_merged, mistranslation_added = _merge_hint_dicts(
        mistranslation_existing if isinstance(mistranslation_existing, list) else [],
        result.get("mistranslation_hints") if isinstance(result.get("mistranslation_hints"), list) else [],
        "term",
    )
    phrase_merged, phrase_added = _merge_hint_dicts(
        phrase_existing if isinstance(phrase_existing, list) else [],
        result.get("phrase_correction_hints") if isinstance(result.get("phrase_correction_hints"), list) else [],
        "phrase",
    )
    ocr_merged, ocr_added = _merge_hint_strings(
        ocr_existing,
        result.get("ocr_low_value_short_texts") if isinstance(result.get("ocr_low_value_short_texts"), list) else [],
    )

    if mistranslation_added:
        (_module_dir() / COMMON_MISTRANSLATION_HINTS_FILE).write_text(json.dumps(mistranslation_merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if phrase_added:
        (_module_dir() / COMMON_PHRASE_CORRECTION_HINTS_FILE).write_text(json.dumps(phrase_merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if ocr_added:
        (_module_dir() / OCR_LOW_VALUE_SHORT_TEXTS_FILE).write_text(json.dumps(ocr_merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stats["mistranslation_added"] = mistranslation_added
    stats["phrase_added"] = phrase_added
    stats["ocr_low_value_added"] = ocr_added
    review_entries[review_identity] = {
        "digest": review_digest,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    review_state["version"] = 1
    review_state_file.parent.mkdir(parents=True, exist_ok=True)
    review_state_file.write_text(
        json.dumps(review_state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def glossary_review_targets(terms: list) -> list[int]:
    targets: list[int] = []
    for index, item in enumerate(terms):
        if not isinstance(item, dict):
            continue
        if item.get("reviewed") is True:
            continue
        targets.append(index)
    return targets


def clean_glossary_review_text(value) -> str:
    if not isinstance(value, str):
        return ""
    text = THINK_RE.sub("", value)
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    text = ASS_TAG_RE.sub("", text)
    text = text.replace(r"\N", " ").replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = text.replace("{", "(").replace("}", ")")
    return re.sub(r"\s+", " ", text).strip().strip("\"'")


def request_glossary_review_batch(
    client: OpenAICompatibleClient,
    model: str,
    batch: list[tuple[int, dict]],
    temperature: float,
) -> dict[int, dict]:
    payload = {
        "items": [
            {
                "id": offset,
                "term": str(item.get("term") or item.get("text") or "").strip(),
                "current_zh": str(item.get("zh") or item.get("note") or "").strip(),
            }
            for offset, item in batch
        ]
    }
    system_prompt = (
        "You are curating a Simplified Chinese subtitle glossary for TV cultural notes. "
        "Return strict JSON only: {\"items\":[{\"id\":1,\"keep\":true,\"term\":\"...\",\"zh\":\"...\"}]}. "
        "Keep only references that genuinely deserve an on-screen note for Chinese viewers: "
        "real people, works, songs, books, shows, sports teams/events, institutions, cultural customs, "
        "religious/historical/scientific concepts, notable slang or foreign phrases. "
        "Remove ordinary words, generic places, short ambiguous abbreviations, OCR fragments, store signs, "
        "credit/sponsor/cast boilerplate, main-character names, and entries whose identity is uncertain. "
        "For kept entries, rewrite zh into one useful compact note, usually 36-70 Chinese characters, "
        "explaining what it is and why the reference matters. "
        "Use established Chinese names when they exist. Simplified Chinese only. "
        "No markdown, brackets, ASS tags, or extra commentary."
    )
    user_prompt = "Review these glossary entries:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    try:
        response = client.chat_json(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max(1200, len(batch) * 120),
        )
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = parse_json_response(content)
    except Exception as exc:
        print(f"  [warning] glossary review request failed: {exc}")
        return {}

    raw_items = parsed.get("items") if isinstance(parsed, dict) else parsed
    result: dict[int, dict] = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                item_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            result[item_id] = item
    return result


def dedupe_glossary_terms(terms: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    output: list[dict] = []
    removed = 0
    for item in terms:
        key = normalize_for_cache(item.get("term") or item.get("text")).lower()
        if not key:
            removed += 1
            continue
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        output.append(item)
    return output, removed


def curate_glossary_file(
    base_dir: Path,
    work_dir: Path,
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    temperature: float,
    cache_dir: Path | None = None,
    codex_command: str = "codex",
    codex_reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
    batch_size: int = DEFAULT_GLOSSARY_REVIEW_BATCH_SIZE,
) -> dict[str, int]:
    stats = {"targets": 0, "reviewed": 0, "changed": 0, "removed": 0, "failed": 0}
    path = base_dir / "subtitle_glossary.json"
    if not path.exists():
        return stats
    data = load_json(path, {})
    if not isinstance(data, dict):
        return stats
    terms = data.get("terms")
    if not isinstance(terms, list):
        return stats
    indexes = glossary_review_targets(terms)
    stats["targets"] = len(indexes)
    if not indexes:
        return stats

    cache_dir = cache_dir or (work_dir / "polish_cache")
    effective_model = model or (DEFAULT_MODEL if provider == "openai" else DEFAULT_CODEX_MODEL)
    model_label = effective_model
    client = build_polish_client(
        provider,
        base_url,
        api_key,
        timeout,
        cache_dir=cache_dir,
        cwd=base_dir,
        codex_command=codex_command,
        codex_reasoning_effort=codex_reasoning_effort,
    )

    remove_indexes: set[int] = set()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    effective_batch_size = len(indexes) if batch_size <= 0 else max(batch_size, 1)
    for offset in range(0, len(indexes), effective_batch_size):
        batch_indexes = indexes[offset : offset + effective_batch_size]
        batch = [(local_id, terms[index]) for local_id, index in enumerate(batch_indexes, start=1)]
        result = request_glossary_review_batch(client, effective_model, batch, temperature)
        for local_id, index in enumerate(batch_indexes, start=1):
            review = result.get(local_id)
            if not isinstance(review, dict):
                stats["failed"] += 1
                continue
            stats["reviewed"] += 1
            if review.get("keep") is False:
                remove_indexes.add(index)
                stats["removed"] += 1
                continue
            item = terms[index] if isinstance(terms[index], dict) else {}
            term = clean_glossary_review_text(review.get("term")) or clean_glossary_review_text(item.get("term") or item.get("text"))
            zh = clean_glossary_review_text(review.get("zh") or review.get("note"))
            if not term or not zh or not has_chinese(zh):
                stats["failed"] += 1
                continue
            if len(zh) > 120:
                stats["failed"] += 1
                continue
            if term != item.get("term") or zh != item.get("zh") or item.get("reviewed") is not True:
                stats["changed"] += 1
            item["term"] = term
            item["zh"] = zh
            item["reviewed"] = True
            item["reviewed_by"] = "codex"
            item["review_prompt_version"] = GLOSSARY_REVIEW_PROMPT_VERSION
            item["review_model"] = model_label
            item["reviewed_at"] = now
            terms[index] = item

    kept_terms = [item for index, item in enumerate(terms) if index not in remove_indexes and isinstance(item, dict)]
    kept_terms, duplicates_removed = dedupe_glossary_terms(kept_terms)
    stats["removed"] += duplicates_removed
    if remove_indexes or duplicates_removed or stats["changed"]:
        data["terms"] = kept_terms
        data["updated_at"] = now
        data["review_prompt_version"] = GLOSSARY_REVIEW_PROMPT_VERSION
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def terminology_scope_matches(item: dict, target_kinds: set[str]) -> bool:
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
    return any(scopes & aliases.get(kind, {kind}) for kind in target_kinds)


def terminology_entry_matches(item: dict, text: str) -> bool:
    source_en = str(item.get("source_en") or item.get("term") or "").strip()
    raw_aliases = item.get("aliases") or []
    aliases = raw_aliases if isinstance(raw_aliases, list) else [raw_aliases]
    candidates = [source_en, *(str(alias).strip() for alias in aliases)]
    flags = 0 if item.get("case_sensitive") is True else re.IGNORECASE
    for candidate in candidates:
        if not candidate:
            continue
        pattern = re.escape(normalize_for_cache(candidate))
        pattern = re.sub(r"\\\s+", r"\\s+", pattern)
        if re.search(rf"(?<![0-9A-Za-z]){pattern}(?![0-9A-Za-z])", normalize_for_cache(text), flags):
            return True
    return False


def load_relevant_terminology(
    base_dir: Path,
    pairs: list[tuple[PolishFileState, PolishTarget]],
    limit: int = 32,
) -> list[dict]:
    data = load_json(base_dir / SUBTITLE_TERMINOLOGY_FILE, {})
    entries = data.get("entries") if isinstance(data, dict) else []
    if not isinstance(entries, list):
        return []
    haystack = "\n".join(f"{target.en}\n{target.zh}" for _, target in pairs)
    file_haystack = "\n".join(
        re.sub(r"[._-]+", " ", state.ass_file.name)
        for state, _ in pairs
    ).lower()
    target_kinds = {target.kind for _, target in pairs}
    selected: list[dict] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        if item.get("reviewed") is False:
            continue
        source_en = str(item.get("source_en") or item.get("term") or "").strip()
        preferred_zh = str(item.get("preferred_zh") or "").strip()
        if not source_en or not preferred_zh:
            continue
        series = str(item.get("series") or "").strip()
        if series and not term_matches_text(series, file_haystack):
            continue
        if not terminology_scope_matches(item, target_kinds):
            continue
        if terminology_entry_matches(item, haystack):
            entry = {"source_en": source_en, "preferred_zh": preferred_zh}
            note = str(item.get("note") or "").strip()
            if note:
                entry["note"] = note
            selected.append(entry)
            if len(selected) >= limit:
                break
    return selected


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def extract_codex_stdout_message(stdout: str) -> str:
    text = str(stdout or "").strip()
    if not text:
        return ""
    marker = "\ncodex\n"
    if marker in text:
        tail = text.rsplit(marker, 1)[1]
        return tail.split("\ntokens used", 1)[0].strip()
    return text


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default
