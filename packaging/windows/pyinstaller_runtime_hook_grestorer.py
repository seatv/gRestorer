import os
import sys
from pathlib import Path

base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))

# Ensure bundled binaries (ffmpeg, ffprobe, torch/cv2 dlls, etc.) resolve first.
paths = [str(base), str(base / "bin")]

# If Intel XPU loader DLL is present, keep system PATH too (Level-Zero/driver stack often lives there).
intel_marker = base / "ur_adapter_level_zero.dll"
if intel_marker.exists():
    system_path = os.environ.get("PATH", "")
    if system_path:
        paths.append(system_path)

os.environ["PATH"] = os.pathsep.join(paths)

# Convenience: allow code to detect its "home" if needed later.
os.environ.setdefault("GRESTORER_HOME", str(base))
