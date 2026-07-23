from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from pipeline_common import PADDLEOCR_SCRIPTS


BITMAP_SUBTITLE_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"}
SECONV_VERSION = "5.0.0"
SECONV_URL = (
    "https://github.com/SubtitleEdit/subtitleedit/releases/download/"
    f"{SECONV_VERSION}/SeConv-Windows-x64.zip"
)
NOCR_DB_URL = "https://raw.githubusercontent.com/SubtitleEdit/subtitleedit/main/Ocr/Latin.nocr"
ENGLISH_DIC_URL = (
    "https://raw.githubusercontent.com/SubtitleEdit/subtitleedit/main/"
    "Dictionaries/en_US.dic"
)
JOINED_WORD_PREFIXES = {
    "after",
    "battle",
    "danger",
    "deadly",
    "dear",
    "new",
    "never",
    "of",
    "off",
    "or",
    "other",
    "outpost",
    "over",
    "smaller",
    "snow",
    "for",
    "their",
    "together",
    "your",
}
COMMON_DICTIONARY_WORDS = {
    "for",
    "heard",
    "the",
}


def _configured_seconv() -> Path | None:
    configured = os.environ.get("SUB_SECONV_EXE", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.exists():
            return path.resolve()
    resolved = shutil.which("seconv")
    return Path(resolved).resolve() if resolved else None


def ensure_seconv(work_dir: Path) -> Path:
    configured = _configured_seconv()
    if configured:
        return configured

    tools_dir = work_dir / "tools"
    existing = sorted(tools_dir.glob("SeConv-*/seconv.exe"))
    if existing:
        return existing[-1]

    install_dir = tools_dir / f"SeConv-{SECONV_VERSION}"
    executable = install_dir / "seconv.exe"
    archive = tools_dir / f"SeConv-Windows-x64-{SECONV_VERSION}.zip"
    tools_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] Subtitle Edit SeConv {SECONV_VERSION}")
    urllib.request.urlretrieve(SECONV_URL, archive)
    install_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        package.extractall(install_dir)
    if not executable.exists():
        candidates = list(install_dir.rglob("seconv.exe"))
        if not candidates:
            raise RuntimeError(f"SeConv executable missing after extraction: {archive}")
        executable = candidates[0]
    return executable


def ensure_latin_nocr_db(work_dir: Path) -> Path:
    db_path = work_dir / "tools" / f"SeConv-{SECONV_VERSION}" / "Ocr" / "Latin.nocr"
    if db_path.exists():
        return db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    print("[download] Subtitle Edit Latin.nocr database")
    urllib.request.urlretrieve(NOCR_DB_URL, db_path)
    return db_path


def ensure_english_dictionary(work_dir: Path) -> Path:
    dictionary = work_dir / "tools" / "en_US.dic"
    if dictionary.exists():
        return dictionary
    dictionary.parent.mkdir(parents=True, exist_ok=True)
    print("[download] Subtitle Edit English OCR dictionary")
    urllib.request.urlretrieve(ENGLISH_DIC_URL, dictionary)
    return dictionary


def load_dictionary_words(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    words = {
        line.split("/", 1)[0].strip().lower()
        for line in lines[1:]
        if line.strip() and not line.startswith("#")
    }
    expanded = set(words) | COMMON_DICTIONARY_WORDS
    for word in words:
        if len(word) < 3:
            continue
        expanded.update({word + "s", word + "ed", word + "ing", word + "er"})
        if word.endswith("e"):
            expanded.update({word + "d", word[:-1] + "ing"})
        if word.endswith("y"):
            expanded.add(word[:-1] + "ies")
    return expanded


def split_joined_ocr_words(text: str, dictionary: set[str]) -> str:
    def replace(match: re.Match) -> str:
        token = match.group(0)
        lowered = token.lower()
        if lowered in dictionary or len(lowered) < 5:
            return token
        for position in range(2, len(lowered) - 1):
            left = lowered[:position]
            right = lowered[position:]
            if left not in JOINED_WORD_PREFIXES or left not in dictionary or right not in dictionary:
                continue
            if token.isupper():
                return f"{left.upper()} {right.upper()}"
            left_text = left.capitalize() if token[0].isupper() else left
            right_text = right.capitalize() if token[position].isupper() else right
            return f"{left_text} {right_text}"
        return token

    return re.sub(r"[A-Za-z]{5,}", replace, text)


def clean_nocr_artifacts(path: Path, work_dir: Path | None = None) -> None:
    text = path.read_text(encoding="utf-8-sig")
    # SeConv's common-error pass can interpret nOCR's generic unknown-glyph
    # marker as a music note. Music classification instead comes from SDH
    # labels/italic runs, so remove that ambiguous marker from dialogue.
    text = text.replace("\u266a", "").replace("*", "")
    while True:
        updated = re.sub(r"(?<=[A-Za-z])I(?=[a-z])", "l", text)
        if updated == text:
            break
        text = updated
    text = re.sub(r"\b[lL](?=(?:'m|'ll|'d|'ve|'re)\b)", "I", text)
    text = re.sub(r"\bl\b", "I", text)
    text = re.sub(
        r"\bI(?=(?:ies|ooks|ike|et|ive|egacy|eave|and)\b)",
        "l",
        text,
    )
    text = text.replace("AII", "All")
    text = re.sub(
        r"\.(</i>)?(\s+)(<i>)?(?=[a-z])",
        lambda match: f"{match.group(1) or ''}{match.group(2)}{match.group(3) or ''}",
        text,
    )
    text = re.sub(r"(?<=[a-z,])\s+In\b", " in", text)
    text = re.sub(r"(?<=-\s)[A-Z][A-Z0-9 .'\-]{1,30}:\s*", "", text)
    if work_dir is not None:
        dictionary = load_dictionary_words(ensure_english_dictionary(work_dir))
        text = split_joined_ocr_words(text, dictionary)
    path.write_text(text, encoding="utf-8-sig")


def extract_bitmap_subtitle(
    video: Path,
    stream: dict,
    out_srt: Path,
    work_dir: Path,
    *,
    force: bool = False,
) -> Path:
    if out_srt.exists() and not force:
        return out_srt
    codec = str(stream.get("codec_name", "")).lower()
    if codec not in BITMAP_SUBTITLE_CODECS:
        raise RuntimeError(f"unsupported bitmap subtitle codec: {codec or 'unknown'}")

    seconv = ensure_seconv(work_dir)
    nocr_db = ensure_latin_nocr_db(work_dir)
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    track_number = int(stream.get("index", 0)) + 1
    # SeConv currently logs --output-folder but writes MKV-track OCR beside the
    # source video. Give that transient file a pipeline-specific name and move
    # it into the work directory immediately after conversion.
    transient_name = f".__seconv_{out_srt.name}"
    transient_file = video.parent / transient_name
    env = os.environ.copy()
    if PADDLEOCR_SCRIPTS.exists():
        env["PATH"] = str(PADDLEOCR_SCRIPTS) + os.pathsep + env.get("PATH", "")
    command = [
        str(seconv),
        str(video),
        "subrip",
        f"--track-number:{track_number}",
        "--ocr-engine:nocr",
        "--ocr-language:en",
        f"--ocr-db:{nocr_db}",
        f"--output-folder:{out_srt.parent}",
        f"--output-filename:{transient_name}",
        "--overwrite",
    ]
    print(
        f"[bitmap-ocr] {video.name}: stream={stream.get('index')} "
        f"codec={codec} images -> {out_srt.name}"
    )
    result = subprocess.run(command, env=env)
    if result.returncode == 0 and transient_file.exists():
        shutil.move(str(transient_file), str(out_srt))
    if result.returncode != 0 or not out_srt.exists():
        print("[bitmap-ocr] nOCR unavailable/failed; falling back to PaddleOCR")
        paddle_command = [
            item
            for item in command
            if not item.startswith("--ocr-db:")
        ]
        paddle_command[paddle_command.index("--ocr-engine:nocr")] = "--ocr-engine:paddle"
        result = subprocess.run(paddle_command, env=env)
        if result.returncode == 0 and transient_file.exists():
            shutil.move(str(transient_file), str(out_srt))
    if result.returncode != 0:
        raise RuntimeError(
            f"SeConv bitmap subtitle OCR failed ({result.returncode}) for "
            f"{video.name} stream {stream.get('index')}"
        )
    if not out_srt.exists():
        raise RuntimeError(f"SeConv reported success but did not create: {out_srt}")
    clean_nocr_artifacts(out_srt, work_dir)
    return out_srt
