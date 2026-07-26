# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path(SPECPATH)
pipeline_data = []
for source in sorted(project_root.glob("*.py")):
    if source.name != "subtitle_forge_ui.py":
        pipeline_data.append((str(source), "pipeline"))
for source in sorted(project_root.glob("*.json")):
    pipeline_data.append((str(source), "pipeline"))
pipeline_data.append((str(project_root / "VERSION"), "pipeline"))


a = Analysis(
    ["subtitle_forge_ui.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=pipeline_data,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="SubtitleForge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
