from __future__ import annotations

import threading
import webbrowser
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    HORIZONTAL,
    LEFT,
    RIGHT,
    VERTICAL,
    BooleanVar,
    Canvas,
    DoubleVar,
    Frame,
    StringVar,
    Text,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
    ttk,
)

from .config import AppSettings, PYTHON_ENV_NAMES, TOOL_ENV_NAMES, load_settings, save_settings
from .environment import CheckResult, HardwareInfo, recommend_qwen, run_environment_checks
from .models import DownloadCancelled, MODEL_CATALOG, ModelDownloader
from .paths import config_path, models_dir
from .planner import PipelineCommand, VideoTask, build_pipeline_commands, scan_video_tasks
from .runner import PipelineRunner


BG = "#f5f7fb"
SURFACE = "#ffffff"
SIDEBAR = "#111827"
SIDEBAR_MUTED = "#9ca3af"
TEXT = "#111827"
MUTED = "#6b7280"
ACCENT = "#4f46e5"
ACCENT_HOVER = "#4338ca"
SUCCESS = "#059669"
WARNING = "#d97706"
DANGER = "#dc2626"
BORDER = "#e5e7eb"


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


class ScrollFrame(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = Canvas(self, background=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=VERTICAL, command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Page.TFrame")
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill="y")
        self.inner.bind("<Configure>", self._sync_scroll)
        self.canvas.bind("<Configure>", self._sync_width)

    def _sync_scroll(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_width(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)


class SubtitleForgeApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.settings = load_settings()
        self.tasks: list[VideoTask] = []
        self.task_rows: dict[str, VideoTask] = {}
        self.pages: dict[str, ttk.Frame] = {}
        self.nav_buttons: dict[str, ttk.Button] = {}
        self.env_results: list[CheckResult] = []
        self.hardware = HardwareInfo(0, "", 0)
        self.downloader = ModelDownloader()
        self.download_thread: threading.Thread | None = None
        self.runner = PipelineRunner(self._runner_output, self._runner_status, self._runner_done)

        self.root.title("Subtitle Forge")
        self.root.geometry("1280x820")
        self.root.minsize(1080, 700)
        self.root.configure(background=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_styles()
        self._build_shell()
        self._build_create_page()
        self._build_environment_page()
        self._build_models_page()
        self._build_settings_page()
        self.show_page("create")
        self.root.after(250, self.check_environment)

    def _build_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", font=("Microsoft YaHei UI", 10), foreground=TEXT)
        style.configure("Page.TFrame", background=BG)
        style.configure("Surface.TFrame", background=SURFACE)
        style.configure("Sidebar.TFrame", background=SIDEBAR)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Microsoft YaHei UI", 22, "bold"))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED, font=("Microsoft YaHei UI", 10))
        style.configure("CardTitle.TLabel", background=SURFACE, foreground=TEXT, font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("CardText.TLabel", background=SURFACE, foreground=MUTED, font=("Microsoft YaHei UI", 9))
        style.configure("SidebarTitle.TLabel", background=SIDEBAR, foreground="#ffffff", font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("SidebarText.TLabel", background=SIDEBAR, foreground=SIDEBAR_MUTED, font=("Microsoft YaHei UI", 9))
        style.configure(
            "Nav.TButton",
            background=SIDEBAR,
            foreground="#d1d5db",
            padding=(18, 13),
            anchor="w",
            borderwidth=0,
            font=("Microsoft YaHei UI", 10),
        )
        style.map("Nav.TButton", background=[("active", "#1f2937")], foreground=[("active", "#ffffff")])
        style.configure(
            "NavActive.TButton",
            background="#312e81",
            foreground="#ffffff",
            padding=(18, 13),
            anchor="w",
            borderwidth=0,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map("NavActive.TButton", background=[("active", "#3730a3")])
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="#ffffff",
            bordercolor=ACCENT,
            padding=(15, 9),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map("Accent.TButton", background=[("active", ACCENT_HOVER)], foreground=[("active", "#ffffff")])
        style.configure("Secondary.TButton", padding=(12, 8), background="#eef2ff", foreground="#3730a3")
        style.map("Secondary.TButton", background=[("active", "#e0e7ff")])
        style.configure("Danger.TButton", padding=(12, 8), background="#fee2e2", foreground="#991b1b")
        style.map("Danger.TButton", background=[("active", "#fecaca")])
        style.configure("Card.TLabelframe", background=SURFACE, bordercolor=BORDER, relief="solid", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background=SURFACE, foreground=TEXT, font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Treeview", rowheight=30, fieldbackground=SURFACE, background=SURFACE, bordercolor=BORDER)
        style.configure("Treeview.Heading", background="#f9fafb", foreground="#374151", font=("Microsoft YaHei UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#e0e7ff")], foreground=[("selected", TEXT)])
        style.configure("TEntry", padding=7)
        style.configure("TCombobox", padding=6)
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor="#e5e7eb")

    def _build_shell(self) -> None:
        shell = ttk.Frame(self.root, style="Page.TFrame")
        shell.pack(fill=BOTH, expand=True)
        sidebar = ttk.Frame(shell, width=214, style="Sidebar.TFrame")
        sidebar.pack(side=LEFT, fill="y")
        sidebar.pack_propagate(False)
        brand = ttk.Frame(sidebar, style="Sidebar.TFrame")
        brand.pack(fill="x", padx=18, pady=(24, 22))
        ttk.Label(brand, text="Subtitle Forge", style="SidebarTitle.TLabel").pack(anchor="w")
        ttk.Label(brand, text="本地双语字幕工作台", style="SidebarText.TLabel").pack(anchor="w", pady=(4, 0))
        nav_items = [
            ("create", "▸  制作字幕"),
            ("environment", "✓  环境检测"),
            ("models", "↓  模型中心"),
            ("settings", "⚙  设置"),
        ]
        for key, label in nav_items:
            button = ttk.Button(sidebar, text=label, style="Nav.TButton", command=lambda value=key: self.show_page(value))
            button.pack(fill="x", padx=10, pady=2)
            self.nav_buttons[key] = button
        ttk.Label(
            sidebar,
            text="Python 与依赖由便携运行时提供\n模型按硬件配置下载",
            style="SidebarText.TLabel",
            justify=LEFT,
        ).pack(side="bottom", anchor="w", padx=20, pady=20)
        self.content = ttk.Frame(shell, style="Page.TFrame")
        self.content.pack(side=LEFT, fill=BOTH, expand=True)

    def _page(self, key: str) -> ttk.Frame:
        page = ttk.Frame(self.content, style="Page.TFrame")
        page.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.pages[key] = page
        return page

    def _page_header(self, page, title: str, subtitle: str) -> ttk.Frame:
        header = ttk.Frame(page, style="Page.TFrame")
        header.pack(fill="x", padx=28, pady=(24, 16))
        ttk.Label(header, text=title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text=subtitle, style="Subtitle.TLabel").pack(anchor="w", pady=(5, 0))
        return header

    def show_page(self, key: str) -> None:
        self.pages[key].tkraise()
        for name, button in self.nav_buttons.items():
            button.configure(style="NavActive.TButton" if name == key else "Nav.TButton")
        if key == "environment" and not self.env_results:
            self.check_environment()

    def _build_create_page(self) -> None:
        page = self._page("create")
        self._page_header(
            page,
            "制作字幕",
            "选择素材目录，自动递归扫描视频并匹配最合适的流水线。",
        )
        source_card = ttk.Frame(page, style="Surface.TFrame", padding=16)
        source_card.pack(fill="x", padx=28, pady=(0, 12))
        ttk.Label(source_card, text="素材目录", style="CardTitle.TLabel").pack(anchor="w")
        row = ttk.Frame(source_card, style="Surface.TFrame")
        row.pack(fill="x", pady=(10, 0))
        self.media_dir_var = StringVar(value=self.settings.last_media_dir)
        ttk.Entry(row, textvariable=self.media_dir_var).pack(side=LEFT, fill="x", expand=True)
        ttk.Button(row, text="浏览…", style="Secondary.TButton", command=self.choose_media_dir).pack(side=LEFT, padx=(8, 0))
        self.scan_button = ttk.Button(row, text="扫描并生成计划", style="Accent.TButton", command=self.scan_media)
        self.scan_button.pack(side=LEFT, padx=(8, 0))

        toolbar = ttk.Frame(page, style="Page.TFrame")
        toolbar.pack(fill="x", padx=28, pady=(0, 8))
        self.plan_summary_var = StringVar(value="尚未扫描")
        ttk.Label(toolbar, textvariable=self.plan_summary_var, style="Subtitle.TLabel").pack(side=LEFT)
        actions = ttk.Frame(toolbar, style="Page.TFrame")
        actions.pack(side=RIGHT)
        for text, command in [
            ("全选", lambda: self._set_all_selected(True)),
            ("取消全选", lambda: self._set_all_selected(False)),
            ("切换 OCR", lambda: self._toggle_selected_field("use_ocr")),
            ("切换云端精修", lambda: self._toggle_selected_field("use_polish")),
            ("切换清理", lambda: self._toggle_selected_field("cleanup")),
        ]:
            ttk.Button(actions, text=text, style="Secondary.TButton", command=command).pack(side=LEFT, padx=(6, 0))

        table_card = ttk.Frame(page, style="Surface.TFrame", padding=1)
        table_card.pack(fill=BOTH, expand=True, padx=28, pady=(0, 10))
        columns = ("selected", "file", "flow", "ocr", "polish", "cleanup", "status")
        self.plan_tree = ttk.Treeview(table_card, columns=columns, show="headings", selectmode="extended")
        headings = {
            "selected": "执行",
            "file": "视频文件",
            "flow": "自动匹配流程",
            "ocr": "画面 OCR",
            "polish": "云端精修",
            "cleanup": "完成后清理",
            "status": "状态",
        }
        widths = {"selected": 55, "file": 270, "flow": 330, "ocr": 85, "polish": 90, "cleanup": 100, "status": 80}
        for key in columns:
            self.plan_tree.heading(key, text=headings[key])
            self.plan_tree.column(key, width=widths[key], minwidth=45, anchor="w" if key in {"file", "flow"} else "center")
        yscroll = ttk.Scrollbar(table_card, orient=VERTICAL, command=self.plan_tree.yview)
        xscroll = ttk.Scrollbar(table_card, orient=HORIZONTAL, command=self.plan_tree.xview)
        self.plan_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.plan_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        table_card.rowconfigure(0, weight=1)
        table_card.columnconfigure(0, weight=1)
        self.plan_tree.bind("<Double-1>", self._tree_double_click)
        self.plan_tree.tag_configure("failed", foreground=DANGER)
        self.plan_tree.tag_configure("done", foreground=SUCCESS)
        self.plan_tree.tag_configure("running", background="#eef2ff")

        bottom = ttk.Frame(page, style="Page.TFrame")
        bottom.pack(fill="x", padx=28, pady=(0, 22))
        ttk.Button(bottom, text="查看命令计划", style="Secondary.TButton", command=self.preview_commands).pack(side=LEFT)
        ttk.Button(bottom, text="查看运行日志", style="Secondary.TButton", command=self.show_log_window).pack(side=LEFT, padx=(8, 0))
        self.stop_button = ttk.Button(bottom, text="停止", style="Danger.TButton", command=self.stop_pipeline, state="disabled")
        self.stop_button.pack(side=RIGHT)
        self.run_button = ttk.Button(bottom, text="开始制作字幕", style="Accent.TButton", command=self.start_pipeline)
        self.run_button.pack(side=RIGHT, padx=(0, 8))

        self.log_window: Toplevel | None = None
        self.log_text: Text | None = None

    def choose_media_dir(self) -> None:
        initial = self.media_dir_var.get() or str(Path.home())
        selected = filedialog.askdirectory(title="选择包含视频的目录", initialdir=initial)
        if selected:
            self.media_dir_var.set(selected)
            self.scan_media()

    def scan_media(self) -> None:
        raw = self.media_dir_var.get().strip()
        if not raw:
            self.choose_media_dir()
            return
        base_dir = Path(raw).expanduser()
        if not base_dir.is_dir():
            messagebox.showerror("目录无效", f"找不到目录：\n{base_dir}")
            return
        self.scan_button.configure(state="disabled", text="正在探测字幕轨…")
        self.plan_summary_var.set("正在递归扫描视频并调用 ffprobe 探测字幕轨…")

        def worker():
            try:
                tasks = scan_video_tasks(base_dir, self.settings)
                self.root.after(0, lambda target=base_dir, values=tasks: self._scan_finished(target, values, ""))
            except Exception as exc:
                self.root.after(0, lambda target=base_dir, error=str(exc): self._scan_finished(target, [], error))

        threading.Thread(target=worker, daemon=True).start()

    def _scan_finished(self, base_dir: Path, tasks: list[VideoTask], error: str) -> None:
        self.scan_button.configure(state="normal", text="扫描并生成计划")
        if error:
            self.plan_summary_var.set("扫描失败")
            messagebox.showerror("扫描失败", error)
            return
        self.settings.last_media_dir = str(base_dir.resolve())
        save_settings(self.settings)
        self.tasks = tasks
        self.task_rows.clear()
        for item in self.plan_tree.get_children():
            self.plan_tree.delete(item)
        for index, task in enumerate(tasks):
            item_id = f"task-{index}"
            self.task_rows[item_id] = task
            self.plan_tree.insert("", END, iid=item_id, values=self._task_values(task))
        groups: dict[str, int] = {}
        for task in tasks:
            groups[task.group] = groups.get(task.group, 0) + 1
        group_summary = " · ".join(f"{count} 个{task_label}" for task_label, count in self._compact_groups(groups))
        self.plan_summary_var.set(f"发现 {len(tasks)} 个视频" + (f" · {group_summary}" if group_summary else ""))
        if not tasks:
            messagebox.showinfo("没有找到视频", "所选目录及其子目录中没有受支持的视频文件。")

    def _compact_groups(self, groups: dict[str, int]):
        labels = {
            "direct_bilingual_done": "已生成双语",
            "ready_bilingual": "已有 ASS",
            "external_mono": "外部字幕",
            "embedded_bilingual": "内嵌中英",
            "embedded_only": "内嵌英文",
            "no_subtitles": "无字幕",
        }
        return [(labels.get(key, key), value) for key, value in groups.items() if value]

    def _task_values(self, task: VideoTask):
        base = Path(self.media_dir_var.get()).resolve(strict=False)
        try:
            name = str(task.path.relative_to(base))
        except ValueError:
            name = str(task.path)
        return (
            "✓" if task.selected else "—",
            name,
            task.flow,
            "✓" if task.use_ocr else "—",
            "✓" if task.use_polish else "—",
            "✓" if task.cleanup else "—",
            task.status,
        )

    def _refresh_task(self, task: VideoTask) -> None:
        for item_id, candidate in self.task_rows.items():
            if candidate is task:
                tag = "failed" if task.status == "失败" else "done" if task.status == "完成" else "running" if task.status == "运行中" else ""
                self.plan_tree.item(item_id, values=self._task_values(task), tags=(tag,) if tag else ())
                break

    def _tree_double_click(self, event) -> None:
        row = self.plan_tree.identify_row(event.y)
        column = self.plan_tree.identify_column(event.x)
        task = self.task_rows.get(row)
        if not task:
            return
        field_by_column = {"#1": "selected", "#4": "use_ocr", "#5": "use_polish", "#6": "cleanup"}
        field = field_by_column.get(column)
        if field:
            setattr(task, field, not getattr(task, field))
            self._refresh_task(task)

    def _set_all_selected(self, selected: bool) -> None:
        for task in self.tasks:
            task.selected = selected
            self._refresh_task(task)

    def _toggle_selected_field(self, field: str) -> None:
        rows = list(self.plan_tree.selection()) or list(self.task_rows)
        for row in rows:
            task = self.task_rows.get(row)
            if task:
                setattr(task, field, not getattr(task, field))
                self._refresh_task(task)

    def _commands(self, dry_run: bool = False) -> list[PipelineCommand]:
        raw = self.media_dir_var.get().strip()
        return build_pipeline_commands(Path(raw), self.tasks, self.settings, dry_run=dry_run) if raw else []

    def preview_commands(self) -> None:
        commands = self._commands(dry_run=True)
        if not commands:
            messagebox.showinfo("没有执行项", "请先扫描目录并选择至少一个视频。")
            return
        dialog = Toplevel(self.root)
        dialog.title("执行计划")
        dialog.geometry("980x620")
        text = Text(dialog, wrap="word", font=("Cascadia Mono", 9), padx=14, pady=14)
        scroll = ttk.Scrollbar(dialog, orient=VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side=LEFT, fill=BOTH, expand=True)
        scroll.pack(side=RIGHT, fill="y")
        for index, command in enumerate(commands, 1):
            text.insert(END, f"任务 {index} · {command.task.path.name}\n{command.display}\n\n")
        text.configure(state="disabled")

    def show_log_window(self) -> None:
        if self.log_window and self.log_window.winfo_exists():
            self.log_window.deiconify()
            self.log_window.lift()
            return
        self.log_window = Toplevel(self.root)
        self.log_window.title("Subtitle Forge 运行日志")
        self.log_window.geometry("1000x650")
        self.log_text = Text(
            self.log_window,
            background="#0b1020",
            foreground="#d1d5db",
            insertbackground="#ffffff",
            font=("Cascadia Mono", 9),
            wrap="word",
            padx=12,
            pady=12,
        )
        scrollbar = ttk.Scrollbar(self.log_window, orient=VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill="y")

    def start_pipeline(self) -> None:
        commands = self._commands()
        if not commands:
            messagebox.showinfo("没有执行项", "请先扫描目录并选择至少一个视频。")
            return
        if any(task.use_polish for task in self.tasks if task.selected) and not self.settings.cloud_polish_available:
            for task in self.tasks:
                if task.selected:
                    task.use_polish = False
                    self._refresh_task(task)
            commands = self._commands()
            messagebox.showinfo("已关闭云端精修", "没有填写云端 API Key，本次不会调用第九步云端大模型。")
        self.show_log_window()
        assert self.log_text is not None
        self.log_text.delete("1.0", END)
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.scan_button.configure(state="disabled")
        try:
            self.runner.start(commands)
        except Exception as exc:
            self._runner_done(False, str(exc))

    def stop_pipeline(self) -> None:
        if self.runner.running and messagebox.askyesno("停止任务", "确定停止当前流水线及其子进程吗？"):
            self.runner.cancel()

    def _runner_output(self, text: str) -> None:
        self.root.after(0, lambda value=text: self._append_log(value))

    def _append_log(self, value: str) -> None:
        if self.log_text and self.log_text.winfo_exists():
            self.log_text.insert(END, value)
            self.log_text.see(END)

    def _runner_status(self, command: PipelineCommand, status: str) -> None:
        def update():
            command.task.status = status
            self._refresh_task(command.task)

        self.root.after(0, update)

    def _runner_done(self, success: bool, message: str) -> None:
        def update():
            self.run_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.scan_button.configure(state="normal")
            self._append_log(f"\n{'✓' if success else '✕'} {message}\n")
            if success:
                messagebox.showinfo("执行完成", message)
            else:
                messagebox.showwarning("执行结束", message)

        self.root.after(0, update)

    def _build_environment_page(self) -> None:
        page = self._page("environment")
        header = self._page_header(page, "环境检测", "检查完整流水线所需的工具、隔离运行时、模型与账户配置。")
        self.env_check_button = ttk.Button(header, text="重新检测", style="Accent.TButton", command=self.check_environment)
        self.env_check_button.pack(side=RIGHT, anchor="e")

        summary = ttk.Frame(page, style="Page.TFrame")
        summary.pack(fill="x", padx=28, pady=(0, 14))
        self.hardware_vars = [StringVar(value="检测中…") for _ in range(3)]
        cards = [
            ("GPU", self.hardware_vars[0]),
            ("系统内存", self.hardware_vars[1]),
            ("Qwen 建议", self.hardware_vars[2]),
        ]
        for index, (title, variable) in enumerate(cards):
            card = ttk.Frame(summary, style="Surface.TFrame", padding=16)
            card.grid(row=0, column=index, sticky="nsew", padx=(0, 10 if index < 2 else 0))
            ttk.Label(card, text=title, style="CardText.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=variable, style="CardTitle.TLabel", wraplength=310).pack(anchor="w", pady=(7, 0))
            summary.columnconfigure(index, weight=1)

        card = ttk.Frame(page, style="Surface.TFrame", padding=1)
        card.pack(fill=BOTH, expand=True, padx=28, pady=(0, 22))
        columns = ("category", "name", "state", "detail")
        self.env_tree = ttk.Treeview(card, columns=columns, show="headings")
        for key, label, width in [
            ("category", "类别", 110),
            ("name", "检查项", 180),
            ("state", "状态", 90),
            ("detail", "说明 / 路径", 650),
        ]:
            self.env_tree.heading(key, text=label)
            self.env_tree.column(key, width=width, anchor="w" if key == "detail" else "center")
        scroll = ttk.Scrollbar(card, orient=VERTICAL, command=self.env_tree.yview)
        self.env_tree.configure(yscrollcommand=scroll.set)
        self.env_tree.pack(side=LEFT, fill=BOTH, expand=True)
        scroll.pack(side=RIGHT, fill="y")
        self.env_tree.tag_configure("ok", foreground=SUCCESS)
        self.env_tree.tag_configure("warning", foreground=WARNING)
        self.env_tree.tag_configure("error", foreground=DANGER)

    def check_environment(self) -> None:
        if not hasattr(self, "env_check_button"):
            return
        self.env_check_button.configure(state="disabled", text="检测中…")

        def worker():
            try:
                hardware, results = run_environment_checks(self.settings)
                self.root.after(
                    0,
                    lambda detected=hardware, values=results: self._environment_finished(detected, values, ""),
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda error=str(exc): self._environment_finished(HardwareInfo(0, "", 0), [], error),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _environment_finished(self, hardware: HardwareInfo, results: list[CheckResult], error: str) -> None:
        self.env_check_button.configure(state="normal", text="重新检测")
        if error:
            messagebox.showerror("环境检测失败", error)
            return
        self.hardware = hardware
        self.env_results = results
        profile, reason = recommend_qwen(hardware)
        self.hardware_vars[0].set(f"{hardware.gpu_name}\n{hardware.vram_gb:.1f} GB 显存" if hardware.vram_gb else hardware.gpu_name)
        self.hardware_vars[1].set(f"{hardware.ram_gb:.1f} GB")
        self.hardware_vars[2].set(f"{profile.upper()} · {reason}")
        self.model_recommendation_var.set(reason)
        for item in self.env_tree.get_children():
            self.env_tree.delete(item)
        state_text = {"ok": "✓ 正常", "warning": "△ 可选/未配置", "error": "✕ 缺失"}
        for index, result in enumerate(results):
            self.env_tree.insert(
                "",
                END,
                iid=f"check-{index}",
                values=(result.category, result.name, state_text[result.state], result.detail),
                tags=(result.state,),
            )

    def _build_models_page(self) -> None:
        page = self._page("models")
        self._page_header(page, "模型中心", "根据硬件选择本地模型；支持断点续传，下载完成后自动写入配置。")
        recommendation = ttk.Frame(page, style="Surface.TFrame", padding=16)
        recommendation.pack(fill="x", padx=28, pady=(0, 14))
        ttk.Label(recommendation, text="硬件建议", style="CardTitle.TLabel").pack(anchor="w")
        self.model_recommendation_var = StringVar(value="正在检测硬件…")
        ttk.Label(recommendation, textvariable=self.model_recommendation_var, style="CardText.TLabel", wraplength=900).pack(anchor="w", pady=(6, 0))

        destination = ttk.Frame(page, style="Surface.TFrame", padding=14)
        destination.pack(fill="x", padx=28, pady=(0, 14))
        ttk.Label(destination, text="模型保存目录", style="CardText.TLabel").pack(anchor="w")
        destination_row = ttk.Frame(destination, style="Surface.TFrame")
        destination_row.pack(fill="x", pady=(6, 0))
        self.model_dir_var = StringVar(value=str(models_dir()))
        ttk.Entry(destination_row, textvariable=self.model_dir_var).pack(side=LEFT, fill="x", expand=True)
        ttk.Button(destination_row, text="选择…", style="Secondary.TButton", command=self.choose_model_dir).pack(side=LEFT, padx=(8, 0))

        cards = ttk.Frame(page, style="Page.TFrame")
        cards.pack(fill=BOTH, expand=True, padx=28)
        self.model_buttons: dict[str, ttk.Button] = {}
        for index, model in enumerate(MODEL_CATALOG.values()):
            card = ttk.Frame(cards, style="Surface.TFrame", padding=18)
            card.grid(row=index, column=0, sticky="ew", pady=(0, 10))
            text_frame = ttk.Frame(card, style="Surface.TFrame")
            text_frame.pack(side=LEFT, fill="x", expand=True)
            ttk.Label(text_frame, text=model.title, style="CardTitle.TLabel").pack(anchor="w")
            ttk.Label(
                text_frame,
                text=f"{model.description}  ·  {model.size_label}",
                style="CardText.TLabel",
                wraplength=760,
            ).pack(anchor="w", pady=(5, 0))
            button = ttk.Button(card, text="下载 / 继续", style="Accent.TButton", command=lambda key=model.key: self.download_model(key))
            button.pack(side=RIGHT)
            self.model_buttons[model.key] = button
        cards.columnconfigure(0, weight=1)
        progress_card = ttk.Frame(page, style="Surface.TFrame", padding=14)
        progress_card.pack(fill="x", padx=28, pady=(2, 22))
        self.download_status_var = StringVar(value="尚未开始下载")
        ttk.Label(progress_card, textvariable=self.download_status_var, style="CardText.TLabel").pack(anchor="w")
        progress_row = ttk.Frame(progress_card, style="Surface.TFrame")
        progress_row.pack(fill="x", pady=(8, 0))
        self.download_progress_var = DoubleVar(value=0)
        self.download_progress = ttk.Progressbar(progress_row, variable=self.download_progress_var, maximum=100)
        self.download_progress.pack(side=LEFT, fill="x", expand=True)
        self.download_cancel_button = ttk.Button(
            progress_row,
            text="暂停",
            style="Danger.TButton",
            command=self.cancel_download,
            state="disabled",
        )
        self.download_cancel_button.pack(side=LEFT, padx=(10, 0))

    def choose_model_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择模型保存目录", initialdir=self.model_dir_var.get())
        if selected:
            self.model_dir_var.set(selected)

    def download_model(self, key: str) -> None:
        if self.download_thread and self.download_thread.is_alive():
            messagebox.showinfo("正在下载", "请先等待当前下载完成，或点击暂停。")
            return
        model = MODEL_CATALOG[key]
        destination = Path(self.model_dir_var.get()).expanduser()
        if not messagebox.askyesno(
            "确认下载",
            f"将下载 {model.title}（{model.size_label}）到：\n{destination}\n\n确认继续吗？",
        ):
            return
        for button in self.model_buttons.values():
            button.configure(state="disabled")
        self.download_cancel_button.configure(state="normal")
        self.download_status_var.set(f"正在连接：{model.title}")
        self.download_progress_var.set(0)

        def progress(downloaded: int, total: int):
            self.root.after(0, lambda: self._download_progress(model.title, downloaded, total))

        def worker():
            try:
                path = self.downloader.download(model, destination, self.settings.hf_token, progress)
                self.root.after(0, lambda model_key=model.key, target=path: self._download_finished(model_key, target, ""))
            except DownloadCancelled as exc:
                self.root.after(
                    0,
                    lambda model_key=model.key, error=str(exc): self._download_finished(model_key, None, error),
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda model_key=model.key, error=f"下载失败：{exc}": self._download_finished(model_key, None, error),
                )

        self.download_thread = threading.Thread(target=worker, daemon=True)
        self.download_thread.start()

    def _download_progress(self, title: str, downloaded: int, total: int) -> None:
        percent = (downloaded / total * 100) if total else 0
        self.download_progress_var.set(percent)
        suffix = f" / {human_bytes(total)} · {percent:.1f}%" if total else ""
        self.download_status_var.set(f"{title}：{human_bytes(downloaded)}{suffix}")

    def cancel_download(self) -> None:
        self.downloader.cancel()
        self.download_cancel_button.configure(state="disabled")
        self.download_status_var.set("正在暂停…")

    def _download_finished(self, key: str, path: Path | None, message: str) -> None:
        for button in self.model_buttons.values():
            button.configure(state="normal")
        self.download_cancel_button.configure(state="disabled")
        if path:
            model = MODEL_CATALOG[key]
            if key == "qwen32":
                self.settings.qwen_32b_path = str(path)
                self.settings.qwen_profile = "32b"
            elif key == "qwen80":
                self.settings.qwen_80b_path = str(path)
                self.settings.qwen_profile = "80b"
            else:
                self.settings.whisper_model_path = str(path)
            save_settings(self.settings)
            self._load_settings_vars()
            self.download_progress_var.set(100)
            self.download_status_var.set(f"下载完成：{path}")
            self.check_environment()
        else:
            self.download_status_var.set(message)
            if message.startswith("下载失败"):
                messagebox.showerror("模型下载失败", message)

    def _build_settings_page(self) -> None:
        page = self._page("settings")
        self._page_header(page, "设置", "自定义工具、隔离运行时、模型和账户；保存后仅作用于 Subtitle Forge。")
        scroll = ScrollFrame(page)
        scroll.pack(fill=BOTH, expand=True, padx=28, pady=(0, 12))
        body = scroll.inner
        self.tool_vars = {name: StringVar() for name in TOOL_ENV_NAMES}
        self.python_vars = {name: StringVar() for name in PYTHON_ENV_NAMES}
        self.settings_vars = {
            "qwen_profile": StringVar(),
            "qwen_32b_path": StringVar(),
            "qwen_80b_path": StringVar(),
            "whisper_model_path": StringVar(),
            "hf_token": StringVar(),
            "polish_provider": StringVar(),
            "polish_api_key": StringVar(),
            "polish_base_url": StringVar(),
            "polish_model": StringVar(),
            "proxy": StringVar(),
            "cache_root": StringVar(),
            "ocr_interval": StringVar(),
        }
        tool_card = ttk.LabelFrame(body, text="外部工具路径", style="Card.TLabelframe", padding=14)
        tool_card.pack(fill="x", pady=(0, 12))
        labels = {
            "ffmpeg": "FFmpeg",
            "ffprobe": "FFprobe",
            "llama-server": "llama-server",
            "whisper-server": "whisper-server",
            "codex": "Codex CLI（可选）",
            "seconv": "Subtitle Edit seconv（可选）",
        }
        for row, name in enumerate(TOOL_ENV_NAMES):
            self._path_setting_row(tool_card, row, labels[name], self.tool_vars[name], executable=True)

        runtime_card = ttk.LabelFrame(body, text="隔离 Python 运行时（便携完整版会自动填写）", style="Card.TLabelframe", padding=14)
        runtime_card.pack(fill="x", pady=(0, 12))
        for row, name in enumerate(PYTHON_ENV_NAMES):
            self._path_setting_row(runtime_card, row, name, self.python_vars[name], executable=True)

        model_card = ttk.LabelFrame(body, text="本地模型", style="Card.TLabelframe", padding=14)
        model_card.pack(fill="x", pady=(0, 12))
        ttk.Label(model_card, text="Qwen 配置", background=SURFACE).grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        ttk.Combobox(
            model_card,
            textvariable=self.settings_vars["qwen_profile"],
            values=("32b", "80b"),
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w", pady=5)
        self._path_setting_row(model_card, 1, "Qwen 32B GGUF", self.settings_vars["qwen_32b_path"])
        self._path_setting_row(model_card, 2, "Qwen 80B GGUF", self.settings_vars["qwen_80b_path"])
        self._path_setting_row(model_card, 3, "Whisper large-v3", self.settings_vars["whisper_model_path"])

        account_card = ttk.LabelFrame(body, text="账户与云端精修", style="Card.TLabelframe", padding=14)
        account_card.pack(fill="x", pady=(0, 12))
        self._text_setting_row(account_card, 0, "Hugging Face Token", self.settings_vars["hf_token"], secret=True)
        link_row = ttk.Frame(account_card, style="Surface.TFrame")
        link_row.grid(row=1, column=1, sticky="w", pady=(0, 8))
        for label, url in [
            ("注册 HF", "https://huggingface.co/join"),
            ("创建 Token", "https://huggingface.co/settings/tokens"),
            ("接受说话人模型条款", "https://huggingface.co/pyannote/speaker-diarization-3.1"),
            ("接受分割模型条款", "https://huggingface.co/pyannote/segmentation-3.0"),
        ]:
            ttk.Button(link_row, text=label, style="Secondary.TButton", command=lambda value=url: webbrowser.open(value)).pack(side=LEFT, padx=(0, 6))
        ttk.Label(account_card, text="云端提供方", background=SURFACE).grid(row=2, column=0, sticky="w", padx=(0, 10), pady=5)
        ttk.Combobox(
            account_card,
            textvariable=self.settings_vars["polish_provider"],
            values=("openai", "codex-cli"),
            state="readonly",
            width=18,
        ).grid(row=2, column=1, sticky="w", pady=5)
        self._text_setting_row(account_card, 3, "API Key", self.settings_vars["polish_api_key"], secret=True)
        api_links = ttk.Frame(account_card, style="Surface.TFrame")
        api_links.grid(row=4, column=1, sticky="w", pady=(0, 8))
        ttk.Button(
            api_links,
            text="注册 OpenAI",
            style="Secondary.TButton",
            command=lambda: webbrowser.open("https://platform.openai.com/signup"),
        ).pack(side=LEFT, padx=(0, 6))
        ttk.Button(
            api_links,
            text="管理 API Keys",
            style="Secondary.TButton",
            command=lambda: webbrowser.open("https://platform.openai.com/api-keys"),
        ).pack(side=LEFT)
        self._text_setting_row(account_card, 5, "Base URL", self.settings_vars["polish_base_url"])
        self._text_setting_row(account_card, 6, "模型名称", self.settings_vars["polish_model"])
        ttk.Label(
            account_card,
            text="未填写 API Key 时，任务计划会自动关闭第九步的云端大模型调用。",
            style="CardText.TLabel",
        ).grid(row=7, column=1, sticky="w", pady=(6, 0))

        advanced = ttk.LabelFrame(body, text="缓存与网络", style="Card.TLabelframe", padding=14)
        advanced.pack(fill="x", pady=(0, 12))
        self._path_setting_row(advanced, 0, "缓存根目录", self.settings_vars["cache_root"], directory=True)
        self._text_setting_row(advanced, 1, "代理（留空直连）", self.settings_vars["proxy"])
        self._text_setting_row(advanced, 2, "OCR 抽帧间隔（秒）", self.settings_vars["ocr_interval"])
        ttk.Label(
            advanced,
            text=f"配置文件：{config_path()}（Token 使用当前 Windows 用户的 DPAPI 加密）",
            style="CardText.TLabel",
        ).grid(row=3, column=1, sticky="w", pady=(6, 0))

        footer = ttk.Frame(page, style="Page.TFrame")
        footer.pack(fill="x", padx=28, pady=(0, 18))
        ttk.Button(footer, text="保存设置并重新检测", style="Accent.TButton", command=self.save_settings_from_ui).pack(side=RIGHT)
        self._load_settings_vars()

    def _path_setting_row(
        self,
        parent,
        row: int,
        label: str,
        variable: StringVar,
        executable: bool = False,
        directory: bool = False,
    ) -> None:
        ttk.Label(parent, text=label, background=SURFACE).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=5)

        def choose():
            if directory:
                value = filedialog.askdirectory(title=f"选择{label}", initialdir=variable.get() or str(Path.home()))
            else:
                filetypes = [("可执行文件", "*.exe"), ("所有文件", "*.*")] if executable else [("模型文件", "*.gguf *.bin"), ("所有文件", "*.*")]
                value = filedialog.askopenfilename(title=f"选择{label}", initialdir=str(Path(variable.get()).parent) if variable.get() else str(Path.home()), filetypes=filetypes)
            if value:
                variable.set(value)

        ttk.Button(parent, text="选择…", style="Secondary.TButton", command=choose).grid(row=row, column=2, padx=(8, 0), pady=5)
        parent.columnconfigure(1, weight=1)

    def _text_setting_row(self, parent, row: int, label: str, variable: StringVar, secret: bool = False) -> None:
        ttk.Label(parent, text=label, background=SURFACE).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
        ttk.Entry(parent, textvariable=variable, show="●" if secret else "").grid(row=row, column=1, columnspan=2, sticky="ew", pady=5)
        parent.columnconfigure(1, weight=1)

    def _load_settings_vars(self) -> None:
        if not hasattr(self, "tool_vars"):
            return
        for name, variable in self.tool_vars.items():
            variable.set(self.settings.tool_paths.get(name, ""))
        for name, variable in self.python_vars.items():
            variable.set(self.settings.python_paths.get(name, ""))
        for name, variable in self.settings_vars.items():
            variable.set(str(getattr(self.settings, name)))

    def save_settings_from_ui(self) -> None:
        try:
            self.settings.tool_paths = {name: variable.get().strip() for name, variable in self.tool_vars.items()}
            self.settings.python_paths = {name: variable.get().strip() for name, variable in self.python_vars.items()}
            for name, variable in self.settings_vars.items():
                value = variable.get().strip()
                if name == "ocr_interval":
                    number = float(value)
                    if number <= 0:
                        raise ValueError("OCR 抽帧间隔必须大于 0")
                    self.settings.ocr_interval = number
                else:
                    setattr(self.settings, name, value)
            path = save_settings(self.settings)
            self.check_environment()
            messagebox.showinfo("设置已保存", f"配置已保存到：\n{path}")
        except Exception as exc:
            messagebox.showerror("无法保存设置", str(exc))

    def _on_close(self) -> None:
        active_download = bool(self.download_thread and self.download_thread.is_alive())
        if self.runner.running or active_download:
            if not messagebox.askyesno("退出 Subtitle Forge", "仍有运行或下载任务，确定停止并退出吗？"):
                return
            self.runner.cancel()
            self.downloader.cancel()
        self.root.destroy()


def run_app(smoke_test: bool = False) -> None:
    root = Tk()
    app = SubtitleForgeApp(root)
    if smoke_test:
        root.after(700, root.destroy)
    root.mainloop()
