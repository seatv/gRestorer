import subprocess

def probe_avg_frame_rate_rational(path: str) -> str:
    # Returns e.g. "10280/343" or "60/1"
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=nw=1:nk=1",
        path
    ]
    out = subprocess.check_output(cmd, text=True).strip()

    if not out or out == "0/0":
        # fallback to r_frame_rate
        se = cmd.index("-show_entries")
        cmd[se + 1] = "stream=r_frame_rate"
        out = subprocess.check_output(cmd, text=True).strip()

    return out

import json
import subprocess
from fractions import Fraction

import subprocess

def probe_format_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        path
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)
