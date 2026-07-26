from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class DownloadableModel:
    key: str
    title: str
    description: str
    filename: str
    size_label: str
    url: str
    profile: str = ""
    sha256: str = ""


MODEL_CATALOG = {
    "qwen32": DownloadableModel(
        key="qwen32",
        title="Qwen3 32B · Q4_K_M",
        description="通用推荐。约 32GB 内存起步，12–24GB 显存可部分卸载。",
        filename="Qwen3-32B-Q4_K_M.gguf",
        size_label="约 19.8 GB",
        url=(
            "https://huggingface.co/Qwen/Qwen3-32B-GGUF/resolve/main/"
            "Qwen3-32B-Q4_K_M.gguf?download=true"
        ),
        profile="32b",
    ),
    "qwen80": DownloadableModel(
        key="qwen80",
        title="Qwen3-Next 80B A3B · Q4_K_M",
        description="高质量推荐。建议 64GB 内存和 24GB 显存。",
        filename="Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf",
        size_label="约 48.4 GB",
        url=(
            "https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF/resolve/main/"
            "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf?download=true"
        ),
        profile="80b",
    ),
    "whisper": DownloadableModel(
        key="whisper",
        title="Whisper.cpp large-v3",
        description="无字幕视频的英文语音识别模型。",
        filename="ggml-large-v3.bin",
        size_label="约 3.1 GB",
        url=(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
            "ggml-large-v3.bin?download=true"
        ),
        sha256="64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
    ),
}


class DownloadCancelled(RuntimeError):
    pass


class ModelDownloader:
    def __init__(self) -> None:
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def download(
        self,
        model: DownloadableModel,
        destination_dir: Path,
        token: str = "",
        progress=None,
    ) -> Path:
        self.cancel_event.clear()
        destination_dir.mkdir(parents=True, exist_ok=True)
        target = destination_dir / model.filename
        partial = target.with_suffix(target.suffix + ".part")
        if target.is_file():
            if not model.sha256 or self._sha256(target) == model.sha256.lower():
                if progress:
                    progress(target.stat().st_size, target.stat().st_size)
                return target
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "SubtitleForge/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(model.url, headers=headers)
        try:
            response = urlopen(request, timeout=60)
        except HTTPError as exc:
            if exc.code == 416 and partial.exists():
                partial.replace(target)
                return target
            raise

        status = getattr(response, "status", 200)
        if offset and status != 206:
            offset = 0
        content_range = response.headers.get("Content-Range", "")
        match = re.search(r"/(\d+)$", content_range)
        if match:
            total = int(match.group(1))
        else:
            remaining = int(response.headers.get("Content-Length") or 0)
            total = offset + remaining if remaining else 0

        mode = "ab" if offset and status == 206 else "wb"
        downloaded = offset
        try:
            with response, partial.open(mode) as output:
                while True:
                    if self.cancel_event.is_set():
                        raise DownloadCancelled("下载已暂停，可稍后继续")
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(downloaded, total)
                output.flush()
                os.fsync(output.fileno())
            if model.sha256 and self._sha256(partial) != model.sha256.lower():
                corrupt = partial.with_suffix(partial.suffix + ".corrupt")
                if corrupt.exists():
                    corrupt.unlink()
                partial.replace(corrupt)
                raise RuntimeError(f"SHA256 校验失败，文件已保留为：{corrupt}")
            partial.replace(target)
            if progress:
                progress(target.stat().st_size, target.stat().st_size)
            return target
        except Exception:
            raise

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while True:
                if self.cancel_event.is_set():
                    raise DownloadCancelled("校验已暂停")
                chunk = source.read(4 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
