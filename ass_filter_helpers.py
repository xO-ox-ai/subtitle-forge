import re


ASS_TAG_RE = re.compile(r"\{[^}]*\}")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def ass_visible_text(line: str) -> str:
    text = str(line or "")
    if line.startswith("Dialogue:"):
        parts = line.split(",", 9)
        if len(parts) == 10:
            text = parts[9]
    text = ASS_TAG_RE.sub("", text)
    return text.replace(r"\N", "\n").strip()


def has_chinese(text: str) -> bool:
    return bool(HAN_RE.search(str(text or "")))
