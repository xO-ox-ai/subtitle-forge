import http.client
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from common import (
    StatusHeartbeat,
    filter_paths_by_stems,
    parse_common_args,
    selected_videos,
    work_dir_for,
    write_status,
)
from pipeline_common import PROJECT_ROOT, WHISPER_DIR, configure_environment, exe_path, int_env, resolve_project_path


configure_environment()


WORD_RE = re.compile(r"\S+")
DEFAULT_UPLOAD_CHUNK_MB = 128


def bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def upload_chunk_size() -> int:
    chunk_mb = max(1, int_env("WHISPER_UPLOAD_CHUNK_MB", DEFAULT_UPLOAD_CHUNK_MB))
    return chunk_mb * 1024 * 1024


def whisper_server_exe() -> str:
    return exe_path("whisper-server.exe", "WHISPER_SERVER_EXE", (WHISPER_DIR,))


def whisper_model_path() -> Path:
    for name in ("WHISPER_MODEL", "WHISPER_CPP_MODEL"):
        value = os.environ.get(name)
        if value:
            return resolve_project_path(value)
    exe = whisper_server_exe()
    resolved = Path(shutil.which(exe) or exe)
    if resolved.exists():
        candidate = resolved.parent / "ggml-large-v3.bin"
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "models" / "ggml-large-v3.bin"


def server_base_url() -> str:
    host = os.environ.get("WHISPER_SERVER_HOST", "127.0.0.1")
    port = int_env("WHISPER_SERVER_PORT", 8091)
    return f"http://{host}:{port}"


def inference_path() -> str:
    path = os.environ.get("WHISPER_INFERENCE_PATH", "/inference")
    return path if path.startswith("/") else f"/{path}"


def server_ready(timeout: int = 2) -> bool:
    try:
        with urlopen(server_base_url() + "/", timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def build_server_cmd(tmp_dir: Path) -> list[str]:
    model = whisper_model_path()
    if not model.exists():
        print(f"[error] Whisper model not found: {model}")
        sys.exit(1)

    host = os.environ.get("WHISPER_SERVER_HOST", "127.0.0.1")
    port = str(int_env("WHISPER_SERVER_PORT", 8091))
    language = os.environ.get("WHISPER_LANGUAGE") or os.environ.get("WHISPERX_LANGUAGE") or "en"
    cmd = [
        whisper_server_exe(),
        "-m",
        str(model),
        "--host",
        host,
        "--port",
        port,
        "--inference-path",
        inference_path(),
        "--tmp-dir",
        str(tmp_dir),
        "-l",
        language,
        "-t",
        str(int_env("WHISPER_THREADS", int_env("WHISPERX_THREADS", 16))),
        "-p",
        str(int_env("WHISPER_PROCESSORS", 1)),
        "-bo",
        str(int_env("WHISPER_BEST_OF", 5)),
        "-bs",
        str(int_env("WHISPER_BEAM_SIZE", 5)),
    ]
    if os.environ.get("WHISPER_DEVICE"):
        cmd.extend(["--device", os.environ["WHISPER_DEVICE"]])
    if bool_env("WHISPER_NO_GPU"):
        cmd.append("--no-gpu")
    if bool_env("WHISPER_VAD"):
        cmd.append("--vad")
        if os.environ.get("WHISPER_VAD_MODEL"):
            cmd.extend(["--vad-model", str(resolve_project_path(os.environ["WHISPER_VAD_MODEL"]))])
    if bool_env("WHISPER_SUPPRESS_NST", True):
        cmd.append("--suppress-nst")
    if bool_env("WHISPER_SERVER_CONVERT"):
        cmd.append("--convert")
    extra = os.environ.get("WHISPER_SERVER_EXTRA_ARGS")
    if extra:
        cmd.extend(shlex.split(extra, posix=False))
    return cmd


def start_whisper_server(work_dir: Path) -> subprocess.Popen | None:
    if server_ready(timeout=1):
        print("Reusing existing Whisper server")
        return None

    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = work_dir / "whisper_server_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_log = log_dir / "whisper_server.out.log"
    err_log = log_dir / "whisper_server.err.log"
    cmd = build_server_cmd(tmp_dir)

    print("Starting Whisper server...")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    out_handle = open(out_log, "a", encoding="utf-8")
    err_handle = open(err_log, "a", encoding="utf-8")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    below_normal = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
    creationflags |= below_normal
    proc = subprocess.Popen(
        cmd,
        stdout=out_handle,
        stderr=err_handle,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        env=os.environ.copy(),
    )
    out_handle.close()
    err_handle.close()

    timeout = int_env("WHISPER_SERVER_TIMEOUT", 300)
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            print(f"[error] Whisper server exited early; logs: {out_log}, {err_log}")
            sys.exit(1)
        if server_ready(timeout=2):
            print("Whisper server is ready")
            return proc
        print("Waiting for Whisper server...", end="\r")
        time.sleep(2)
    print()
    stop_whisper_server(proc)
    print(f"[error] Whisper server was not ready within {timeout}s; logs: {out_log}, {err_log}")
    sys.exit(1)


def stop_whisper_server(proc: subprocess.Popen | None) -> None:
    if not proc or proc.poll() is not None:
        return
    print("Stopping Whisper server...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def multipart_prefix(fields: dict[str, str], file_field: str, file_path: Path, boundary: str) -> list[bytes]:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode("utf-8"),
            b"Content-Type: audio/wav\r\n\r\n",
        ]
    )
    return chunks


def post_inference(wav: Path) -> dict:
    fields = {
        "response_format": "verbose_json",
    }
    language = os.environ.get("WHISPER_LANGUAGE") or os.environ.get("WHISPERX_LANGUAGE")
    if language:
        fields["language"] = language

    boundary = "----sub-whisper-" + uuid.uuid4().hex
    prefix = multipart_prefix(fields, "file", wav, boundary)
    suffix = [b"\r\n", f"--{boundary}--\r\n".encode("utf-8")]
    content_length = sum(len(part) for part in prefix) + wav.stat().st_size + sum(len(part) for part in suffix)
    parsed = urlparse(server_base_url())
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    timeout = int_env("WHISPER_REQUEST_TIMEOUT", 7200)

    connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = connection_cls(host, port, timeout=timeout)
    try:
        conn.putrequest("POST", inference_path())
        conn.putheader("Host", f"{host}:{port}")
        conn.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        conn.putheader("Content-Length", str(content_length))
        conn.endheaders()
        for part in prefix:
            conn.send(part)
        chunk_size = upload_chunk_size()
        with wav.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                conn.send(chunk)
        for part in suffix:
            conn.send(part)
        response = conn.getresponse()
        payload = response.read()
        if not (200 <= response.status < 300):
            detail = payload.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Whisper server returned HTTP {response.status} for {wav.name}: {detail}")
        return json.loads(payload.decode("utf-8"))
    except (OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Whisper server request failed for {wav.name}: {exc}") from exc
    finally:
        conn.close()


def parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    text = str(value).replace(",", ".")
    try:
        head, fraction = text.split(".", 1)
        hours, minutes, seconds = [int(part) for part in head.split(":")]
        millis = int((fraction + "000")[:3])
        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0
    except (ValueError, TypeError):
        return None


def seconds_from_offsets(item: dict, key: str, fallback: float | None = None) -> float:
    offsets = item.get("offsets") if isinstance(item, dict) else {}
    if isinstance(offsets, dict) and offsets.get(key) is not None:
        try:
            return max(0.0, float(offsets[key]) / 1000.0)
        except (TypeError, ValueError):
            pass
    timestamps = item.get("timestamps") if isinstance(item, dict) else {}
    if isinstance(timestamps, dict):
        parsed = parse_timestamp(timestamps.get("from" if key == "from" else "to"))
        if parsed is not None:
            return parsed
    return 0.0 if fallback is None else fallback


def is_special_token(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("[_") and stripped.endswith("]")


def fallback_words(text: str, start: float, end: float) -> list[dict]:
    words = [match.group(0) for match in WORD_RE.finditer(str(text or ""))]
    if not words:
        return []
    duration = max(0.05, end - start)
    step = duration / len(words)
    output = []
    for index, word in enumerate(words):
        word_start = start + step * index
        word_end = start + step * (index + 1)
        output.append({"word": word, "start": word_start, "end": word_end, "score": 0.0})
    return output


def token_words(tokens: list[dict], seg_start: float, seg_end: float, text: str) -> list[dict]:
    words: list[dict] = []
    current_text = ""
    current_start: float | None = None
    current_end: float | None = None
    current_scores: list[float] = []

    def flush() -> None:
        nonlocal current_text, current_start, current_end, current_scores
        word = current_text.strip()
        if word and current_start is not None and current_end is not None and current_end > current_start:
            score = sum(current_scores) / len(current_scores) if current_scores else 0.0
            words.append({"word": word, "start": current_start, "end": current_end, "score": score})
        current_text = ""
        current_start = None
        current_end = None
        current_scores = []

    for token in tokens:
        raw = str(token.get("word", token.get("text", "")))
        if not raw.strip() or is_special_token(raw):
            continue
        token_start = seconds_from_offsets(token, "from", seg_start)
        token_end = seconds_from_offsets(token, "to", token_start)
        if token.get("start") is not None:
            try:
                token_start = float(token["start"])
            except (TypeError, ValueError):
                pass
        if token.get("end") is not None:
            try:
                token_end = float(token["end"])
            except (TypeError, ValueError):
                pass
        if token_end <= token_start:
            if current_end is None:
                continue
            token_start = current_end
            token_end = min(seg_end, max(token_start + 0.01, current_end))
        starts_new_word = raw[:1].isspace() and current_text.strip()
        if starts_new_word:
            flush()
        if current_start is None:
            current_start = token_start
        current_end = token_end
        current_text += raw
        try:
            current_scores.append(float(token.get("probability", token.get("p", 0.0))))
        except (TypeError, ValueError):
            pass
    flush()
    return words or fallback_words(text, seg_start, seg_end)


def normalize_whisper_json(raw: dict) -> dict:
    language = (
        raw.get("language")
        or raw.get("detected_language")
        or (raw.get("result") or {}).get("language")
        or (raw.get("params") or {}).get("language")
        or os.environ.get("WHISPER_LANGUAGE")
        or os.environ.get("WHISPERX_LANGUAGE")
        or "en"
    )
    source_segments = raw.get("segments")
    if not isinstance(source_segments, list):
        source_segments = raw.get("transcription", []) if isinstance(raw.get("transcription"), list) else []

    segments = []
    word_segments = []
    for index, item in enumerate(source_segments, start=1):
        if not isinstance(item, dict):
            continue
        start = float(item.get("start")) if item.get("start") is not None else seconds_from_offsets(item, "from")
        end = float(item.get("end")) if item.get("end") is not None else seconds_from_offsets(item, "to", start + 1.0)
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            continue
        if end <= start:
            end = start + 0.5
        words_source = item.get("words") if isinstance(item.get("words"), list) else item.get("tokens") or []
        words = token_words(words_source, start, end, text)
        segment = {
            "id": item.get("id", index),
            "start": start,
            "end": end,
            "text": text,
            "words": words,
        }
        segments.append(segment)
        for word in words:
            flat_word = dict(word)
            flat_word["segment_id"] = segment["id"]
            word_segments.append(flat_word)

    return {
        "segments": segments,
        "word_segments": word_segments,
        "language": language,
        "source": "whisper-server",
    }


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_common_args("V2 STEP3: Whisper server transcription for selected videos.")
    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    vocals_dir = work_dir / "vocals"
    transcripts_dir = work_dir / "transcripts"
    raw_dir = work_dir / "transcripts_raw"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    stems = {video.stem for video in selected_videos(base_dir, args.chunk, args.target_stem)}
    wav_files = filter_paths_by_stems(sorted(vocals_dir.glob("*.wav")), stems)
    if not wav_files:
        print(f"[error] no vocals found in {vocals_dir}")
        sys.exit(1)

    pending = [wav for wav in wav_files if not (transcripts_dir / f"{wav.stem}.json").exists()]
    for wav in wav_files:
        if wav not in pending:
            print(f"[skip] transcript exists: {wav.stem}.json")
    if not pending:
        print("v2 step3 done; all selected transcript files already exist")
        return

    write_status(base_dir, "V2_STEP3", f"{len(pending)} files", "START Whisper server transcription")
    start = time.time()
    proc: subprocess.Popen | None = None
    try:
        for wav in pending:
            raw_file = raw_dir / f"{wav.stem}.json"
            out_file = transcripts_dir / f"{wav.stem}.json"
            print(f"[process] {wav.name}")
            write_status(base_dir, "V2_STEP3", wav.name, "Whisper server transcription")

            if not raw_file.exists():
                if proc is None and not server_ready(timeout=1):
                    proc = start_whisper_server(work_dir)
                with StatusHeartbeat(base_dir, "V2_STEP3", wav.name, "Whisper server inference"):
                    raw = post_inference(wav)
                save_json(raw_file, raw)
            else:
                print(f"[skip] raw whisper output exists: {raw_file.name}")
                raw = load_json(raw_file)

            try:
                normalized = normalize_whisper_json(raw)
            except Exception:
                print(f"[warn] invalid raw JSON, regenerating: {raw_file.name}")
                raw_file.unlink(missing_ok=True)
                if proc is None and not server_ready(timeout=1):
                    proc = start_whisper_server(work_dir)
                raw = post_inference(wav)
                save_json(raw_file, raw)
                normalized = normalize_whisper_json(raw)

            save_json(out_file, normalized)
            print(f"[done] transcript saved: {out_file}; segments={len(normalized.get('segments', []))}")
    finally:
        stop_whisper_server(proc)

    elapsed = time.time() - start
    write_status(base_dir, "V2_STEP3", f"{len(pending)} files", f"DONE Whisper server transcription ({elapsed:.1f}s)")
    print(f"v2 step3 done; transcripts saved in: {transcripts_dir}")


if __name__ == "__main__":
    main()
