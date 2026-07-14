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

from pipeline_common import PROJECT_ROOT, LLAMA_DIR, configure_environment, exe_path, int_env, resolve_project_path


configure_environment()


QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8080/v1")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "not-needed")
DEFAULT_QWEN_PROFILE = "80b"
QWEN_PROFILES = {
    "32b": {
        "model": "qwen3-32b",
        "filename": "Qwen3-32B-Q4_K_M.gguf",
        "ctx_size": 8192,
        "batch_size": 2048,
        "ubatch_size": 512,
        "threads": 20,
        "threads_batch": 20,
        "gpu_layers": "all",
        "flash_attn": "on",
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "parallel": 1,
        "reasoning": "off",
        "reasoning_budget": 0,
        "cache_ram": 4096,
    },
    "80b": {
        "model": "qwen3-next-80b-a3b-instruct",
        "filename": "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf",
        "ctx_size": 8192,
        "batch_size": 1024,
        "ubatch_size": 256,
        "threads": 16,
        "threads_batch": 16,
        "gpu_layers": "16",
        "flash_attn": "on",
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "parallel": 1,
        "reasoning": "off",
        "reasoning_budget": 0,
        "cache_ram": 2048,
    },
}
QWEN_PROFILE = ""
QWEN_MODEL = ""
QWEN_GGUF = Path()
LLAMA_SERVER_EXE = exe_path("llama-server.exe", "LLAMA_SERVER_EXE", (LLAMA_DIR,))


def _profile_env_name(profile: str, setting: str) -> str:
    return f"QWEN_{profile.upper()}_{setting}"


def _profile_setting(profile: str, setting: str):
    config = QWEN_PROFILES[profile]
    profile_value = os.environ.get(_profile_env_name(profile, setting.upper()))
    if profile_value not in {None, ""}:
        return profile_value
    # Preserve legacy generic overrides for the original 32B profile. The 80B
    # profile intentionally requires profile-specific overrides so an old
    # QWEN_GPU_LAYERS=all setting cannot exhaust a 24GB desktop GPU.
    if profile == "32b":
        generic_value = os.environ.get(f"QWEN_{setting.upper()}")
        if generic_value not in {None, ""}:
            return generic_value
    return config[setting]


def default_qwen_gguf(profile: str) -> Path:
    profile_env = os.environ.get(_profile_env_name(profile, "GGUF"))
    if profile_env:
        return resolve_project_path(profile_env)
    if profile == "32b" and os.environ.get("QWEN_GGUF"):
        return resolve_project_path(os.environ["QWEN_GGUF"])
    search_dirs: list[Path] = []
    server_path = shutil.which(str(LLAMA_SERVER_EXE)) or str(LLAMA_SERVER_EXE)
    try:
        search_dirs.append(Path(server_path).resolve().parent)
    except OSError:
        pass
    search_dirs.append(LLAMA_DIR)
    search_dirs.append(PROJECT_ROOT / "models")
    for directory in search_dirs:
        candidate = directory / QWEN_PROFILES[profile]["filename"]
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "models" / QWEN_PROFILES[profile]["filename"]


def configure_qwen_profile(profile: str | None = None, gguf: str | Path | None = None) -> dict:
    global QWEN_PROFILE, QWEN_MODEL, QWEN_GGUF
    selected = str(profile or os.environ.get("QWEN_PROFILE") or DEFAULT_QWEN_PROFILE).strip().lower()
    if selected not in QWEN_PROFILES:
        raise ValueError(f"Unknown Qwen profile: {selected}; choose one of {sorted(QWEN_PROFILES)}")
    QWEN_PROFILE = selected
    QWEN_MODEL = str(_profile_setting(selected, "model"))
    QWEN_GGUF = resolve_project_path(str(gguf)) if gguf else default_qwen_gguf(selected)
    return {"profile": QWEN_PROFILE, "model": QWEN_MODEL, "gguf": str(QWEN_GGUF)}


configure_qwen_profile()


def server_model_ids(timeout: int = 5) -> set[str]:
    try:
        with urlopen(QWEN_BASE_URL.rstrip("/") + "/models", timeout=timeout) as resp:
            if not 200 <= resp.status < 300:
                return set()
            payload = json.loads(resp.read().decode("utf-8"))
            return {
                str(item.get("id", ""))
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            }
    except Exception:
        return set()


def server_ready(timeout: int = 5, expected_model: str | None = None) -> bool:
    models = server_model_ids(timeout)
    return bool(models and (not expected_model or expected_model in models))


def parse_server_target(base_url: str) -> tuple[str, str]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or (443 if parsed.scheme == "https" else 80))
    return host, port


def build_server_cmd() -> list[str]:
    host, port = parse_server_target(QWEN_BASE_URL)
    profile = QWEN_PROFILE
    return [
        str(LLAMA_SERVER_EXE),
        "-m", str(QWEN_GGUF),
        "--alias", QWEN_MODEL,
        "--host", host,
        "--port", port,
        "--ctx-size", str(_profile_setting(profile, "ctx_size")),
        "--batch-size", str(_profile_setting(profile, "batch_size")),
        "--ubatch-size", str(_profile_setting(profile, "ubatch_size")),
        "--threads", str(_profile_setting(profile, "threads")),
        "--threads-batch", str(_profile_setting(profile, "threads_batch")),
        "--n-gpu-layers", str(_profile_setting(profile, "gpu_layers")),
        "--main-gpu", os.environ.get("QWEN_MAIN_GPU", "0"),
        "--flash-attn", str(_profile_setting(profile, "flash_attn")),
        "--cache-type-k", str(_profile_setting(profile, "cache_type_k")),
        "--cache-type-v", str(_profile_setting(profile, "cache_type_v")),
        "--parallel", str(_profile_setting(profile, "parallel")),
        "--reasoning", str(_profile_setting(profile, "reasoning")),
        "--reasoning-budget", str(_profile_setting(profile, "reasoning_budget")),
        "--cache-ram", str(_profile_setting(profile, "cache_ram")),
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
        if server_ready(timeout=5, expected_model=QWEN_MODEL):
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
    available_models = server_model_ids(timeout=3)
    if QWEN_MODEL in available_models:
        print(f"Reusing existing Qwen3 service: profile={QWEN_PROFILE} model={QWEN_MODEL}")
        return client, None
    if available_models:
        print(
            f"[error] Qwen service already runs a different model at {QWEN_BASE_URL}: "
            f"available={sorted(available_models)}, requested={QWEN_MODEL}"
        )
        sys.exit(1)

    log_file = base_dir / "temp" / "qwen_server.log"
    print(f"Qwen profile={QWEN_PROFILE} model={QWEN_MODEL} gguf={QWEN_GGUF}")
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
        raise RuntimeError(
            f"Qwen request failed for profile={QWEN_PROFILE} model={QWEN_MODEL}; "
            "the current output was not written. Restart with --qwen-profile 32b if the 80b service is unstable."
        ) from exc
