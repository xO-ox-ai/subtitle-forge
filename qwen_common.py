import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pipeline_common import LLAMA_DIR, configure_environment, exe_path, int_env


configure_environment()


QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8080/v1")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "not-needed")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3-32b")
LLAMA_SERVER_EXE = exe_path("llama-server.exe", "LLAMA_SERVER_EXE", (LLAMA_DIR,))


def default_qwen_gguf() -> Path:
    if os.environ.get("QWEN_GGUF"):
        return Path(os.environ["QWEN_GGUF"])
    search_dirs: list[Path] = []
    server_path = shutil.which(str(LLAMA_SERVER_EXE)) or str(LLAMA_SERVER_EXE)
    try:
        search_dirs.append(Path(server_path).resolve().parent)
    except OSError:
        pass
    search_dirs.append(LLAMA_DIR)
    for directory in search_dirs:
        candidate = directory / "Qwen3-32B-Q4_K_M.gguf"
        if candidate.exists():
            return candidate
    return Path("Qwen3-32B-Q4_K_M.gguf")


QWEN_GGUF = default_qwen_gguf()


def server_ready(timeout: int = 5) -> bool:
    try:
        with urlopen(QWEN_BASE_URL.rstrip("/") + "/models", timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def parse_server_target(base_url: str) -> tuple[str, str]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or (443 if parsed.scheme == "https" else 80))
    return host, port


def build_server_cmd() -> list[str]:
    host, port = parse_server_target(QWEN_BASE_URL)
    return [
        str(LLAMA_SERVER_EXE),
        "-m", str(QWEN_GGUF),
        "--alias", QWEN_MODEL,
        "--host", host,
        "--port", port,
        "--ctx-size", str(int_env("QWEN_CTX_SIZE", 8192)),
        "--batch-size", str(int_env("QWEN_BATCH_SIZE", 2048)),
        "--ubatch-size", str(int_env("QWEN_UBATCH_SIZE", 512)),
        "--threads", str(int_env("QWEN_THREADS", 20)),
        "--threads-batch", str(int_env("QWEN_THREADS_BATCH", 20)),
        "--n-gpu-layers", os.environ.get("QWEN_GPU_LAYERS", "all"),
        "--main-gpu", os.environ.get("QWEN_MAIN_GPU", "0"),
        "--flash-attn", os.environ.get("QWEN_FLASH_ATTN", "on"),
        "--cache-type-k", os.environ.get("QWEN_CACHE_TYPE_K", "q8_0"),
        "--cache-type-v", os.environ.get("QWEN_CACHE_TYPE_V", "q8_0"),
        "--parallel", str(int_env("QWEN_PARALLEL", 1)),
        "--reasoning", os.environ.get("QWEN_REASONING", "off"),
        "--reasoning-budget", os.environ.get("QWEN_REASONING_BUDGET", "0"),
        "--cont-batching",
        "--no-webui",
    ]


def start_qwen_server(log_file: Path) -> subprocess.Popen:
    if not Path(LLAMA_SERVER_EXE).exists() and shutil.which(str(LLAMA_SERVER_EXE)) is None:
        print(f"[error] llama-server not found: {LLAMA_SERVER_EXE}")
        sys.exit(1)
    if not QWEN_GGUF.exists():
        print(f"[error] Qwen GGUF model not found: {QWEN_GGUF}")
        sys.exit(1)

    cmd = build_server_cmd()
    print("Starting Qwen3 llama-server...")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_file, "a", encoding="utf-8")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    below_normal = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
    creationflags |= below_normal
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        env=os.environ.copy(),
    )
    log.close()
    return proc


def wait_for_server(timeout: int = 300) -> bool:
    print("Checking Qwen3 service...")
    start = time.time()
    while time.time() - start < timeout:
        if server_ready(timeout=5):
            print("Qwen3 service is ready")
            return True
        print("Waiting for Qwen3 service...", end="\r")
        time.sleep(5)
    print()
    return False


def stop_qwen_server(proc: subprocess.Popen | None) -> None:
    if not proc or proc.poll() is not None:
        return
    print("Stopping Qwen3 llama-server...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


class QwenHTTPClient:
    def __init__(self, base_url: str = QWEN_BASE_URL, api_key: str = QWEN_API_KEY, timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create_chat_completion))

    def create_chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                )
            ],
            raw=data,
        )


def get_client(base_dir: Path) -> tuple[QwenHTTPClient, subprocess.Popen | None]:
    client = QwenHTTPClient(timeout=int_env("QWEN_REQUEST_TIMEOUT", 600))
    if server_ready(timeout=3):
        print("Reusing existing Qwen3 service")
        return client, None

    log_file = base_dir / "temp" / "qwen_server.log"
    proc = start_qwen_server(log_file)
    timeout = int_env("QWEN_SERVER_TIMEOUT", 300)
    if not wait_for_server(timeout=timeout):
        stop_qwen_server(proc)
        print(f"[error] Qwen3 service was not ready within {timeout}s; log: {log_file}")
        sys.exit(1)
    return client, proc


def parse_json_response(content: str, fallback):
    cleaned = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return fallback


def chat_json(client: QwenHTTPClient, system_prompt: str, user_prompt: str, fallback, max_tokens: int = 512):
    try:
        resp = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        content = (resp.choices[0].message.content or "").strip()
        return parse_json_response(content, fallback)
    except Exception as exc:
        print(f"  [warning] Qwen request failed: {exc}")
        return fallback
