from __future__ import annotations

import os
import queue
import subprocess
import threading
from collections.abc import Callable

from .planner import PipelineCommand


class PipelineRunner:
    def __init__(
        self,
        on_output: Callable[[str], None],
        on_status: Callable[[PipelineCommand, str], None],
        on_done: Callable[[bool, str], None],
    ) -> None:
        self.on_output = on_output
        self.on_status = on_status
        self.on_done = on_done
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self, commands: list[PipelineCommand]) -> None:
        if self.running:
            raise RuntimeError("已有任务正在运行")
        self._cancel.clear()
        self._thread = threading.Thread(target=self._run, args=(commands,), daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()
        process = self._process
        if not process or process.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            else:
                process.terminate()
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass

    def _run(self, commands: list[PipelineCommand]) -> None:
        try:
            for index, command in enumerate(commands, start=1):
                if self._cancel.is_set():
                    self.on_done(False, "任务已取消")
                    return
                self.on_status(command, "运行中")
                self.on_output(f"\n▶ 任务 {index}/{len(commands)}\n{command.display}\n\n")
                creationflags = 0
                if os.name == "nt":
                    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                self._process = subprocess.Popen(
                    command.argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=command.env,
                    creationflags=creationflags,
                )
                assert self._process.stdout is not None
                for line in iter(self._process.stdout.readline, ""):
                    self.on_output(line)
                    if self._cancel.is_set():
                        break
                return_code = self._process.wait()
                self._process = None
                if self._cancel.is_set():
                    self.on_status(command, "已取消")
                    self.on_done(False, "任务已取消")
                    return
                if return_code != 0:
                    self.on_status(command, "失败")
                    self.on_done(False, f"{command.task.path.name} 执行失败，退出码 {return_code}")
                    return
                self.on_status(command, "完成")
            self.on_done(True, f"执行完成，共 {len(commands)} 个任务")
        except Exception as exc:
            self.on_done(False, f"运行失败：{exc}")
        finally:
            self._process = None
