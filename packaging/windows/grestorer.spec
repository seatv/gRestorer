# -*- mode: python ; coding: utf-8 -*-
# gRestorer CLI PyInstaller spec (Windows) — entrypoint-safe (no relative-import crash)

import argparse
import fnmatch
import pathlib
import shutil
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

def get_project_root() -> pathlib.Path:
    root = pathlib.Path(".").absolute()
    assert (root / "pyproject.toml").exists(), "Run PyInstaller from the repo root (pyproject.toml must exist)."
    return root

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="nvidia", choices=["nvidia", "cpu", "intel"])
    p.add_argument("--skip-ffmpeg", action="store_true", help="Do not bundle ffmpeg/ffprobe (expect them on PATH).")
    p.add_argument("--name", default="grestorer", help="Base exe/folder name (default: grestorer)")
    return p.parse_args()

args = parse_args()
TARGET = args.target.lower()
NAME = args.name

project_root = get_project_root()

# --- Bundled binaries (ffmpeg/ffprobe) ---
binaries = []

def _which_or_none(exe):
    try:
        return shutil.which(exe)
    except Exception:
        return None

if not args.skip_ffmpeg:
    ffmpeg = _which_or_none("ffmpeg.exe")
    ffprobe = _which_or_none("ffprobe.exe")
    if ffmpeg:
        binaries.append((ffmpeg, "bin"))
    if ffprobe:
        binaries.append((ffprobe, "bin"))

# --- Intel XPU runtime loader DLLs (best effort) ---
def get_intel_xpu_runtime_libs():
    if TARGET != "intel":
        return []
    found = []
    patterns = ["ur_win*.dll", "ur_loader.dll", "ur_adapter_level_zero.dll"]
    candidates = [project_root / "venv", project_root / ".venv", pathlib.Path(sys.prefix)]
    for base in candidates:
        if not base.exists():
            continue
        for p in base.rglob("*.dll"):
            name = p.name.lower()
            if any(fnmatch.fnmatch(name, pat.lower()) for pat in patterns):
                found.append((str(p), "."))
    seen = set()
    out = []
    for src, dst in found:
        key = src.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((src, dst))
    return out

binaries += get_intel_xpu_runtime_libs()

# --- Data files ---
datas = []
datas += collect_data_files("ultralytics", include_py_files=False)

# --- Hidden imports ---
hiddenimports = []
hiddenimports += collect_submodules("gRestorer")
hiddenimports += collect_submodules("ultralytics")
hiddenimports += collect_submodules("cv2")
hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("PyNvVideoCodec")

# --- Dynamic libs ---
binaries += collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("cv2")
binaries += collect_dynamic_libs("PyNvVideoCodec")

runtime_hooks = [str(project_root / "packaging" / "windows" / "pyinstaller_runtime_hook_grestorer.py")]

# IMPORTANT:
# Do NOT use gRestorer/cli/main.py as the entry script; it uses relative imports.
# Use a tiny bootstrap entrypoint that imports gRestorer.cli.main as a package module.
entry_script = str(project_root / "packaging" / "windows" / "grestorer_entrypoint.py")

a = Analysis(
    [entry_script],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name=NAME,
)
