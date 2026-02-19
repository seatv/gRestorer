## gRestorer Windows Packaging (CLI)

This folder builds a **portable onedir** Windows release using **PyInstaller**.

### Prereqs (build machine)
- Windows 10/11
- A working gRestorer repo checkout
- A Python venv that already has *your desired torch variant* installed:
  - CUDA build: torch + cu12x wheels
  - CPU build: cpu wheels
  - Intel XPU build: intel/ipex wheels (and runtime available)

Optional (recommended):
- `ffmpeg.exe` and `ffprobe.exe` in PATH (will be bundled into `dist\grestorer\bin`)

### Quick build
From the **repo root**:

```powershell
# 1) Activate your release venv first (same one you run grestorer from)
# .\venv\Scripts\Activate.ps1

# 2) Build
powershell -ExecutionPolicy Bypass .\packaging\windows\package_executable.ps1 -Target nvidia
```

Targets:
- `nvidia` (default): CUDA-friendly build (still runs on CPU if CUDA unavailable)
- `cpu`: CPU-only environment build
- `intel`: Intel XPU environment build (copies UR/Level-Zero loader DLLs if found)

### Output
- `dist\grestorer\grestorer.exe`
- `dist\grestorer\bin\ffmpeg.exe` (if bundled)
- `dist\grestorer\config.json` (template)

### Notes
- The bundled `config.json` is a **template**. Users should pass `--config` or edit it.
- A `grestorer.cmd` wrapper is included so double-click runs with the correct working directory.
