"""
Video encoder using NVIDIA PyNvVideoCodec.

Provides hardware-accelerated video encoding with GPU input.

Key detail:
- If you write raw HEVC bytes to a file, that file is NOT an MP4 container even if it ends with .mp4.
  Players will guess timing → choppy / wrong FPS is common.
- PyNvVideoCodec can mux (via Lavf/libavformat) when you pass container="mp4" or "mkv".
  We infer that from the output extension (or you can override via constructor).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import PyNvVideoCodec as nvc


def _infer_container(output_path: str | Path) -> Optional[str]:
    """
    Infer container muxing from file extension.

    Returns:
      "mp4" | "mkv" | None

    Note: "mp4" here means an ISO BMFF container (mov/mp4 family).
    """
    suf = Path(output_path).suffix.lower()
    if suf in (".mp4", ".m4v", ".mov"):
        return "mp4"
    if suf == ".mkv":
        return "mkv"
    return None


class Encoder:
    """
    Hardware-accelerated video encoder using NVENC.

    Encodes video from GPU memory surfaces.
    Expects BGRA format [H, W, 4] input (mapped to "ARGB" encoder format).

    If container muxing is enabled, the bytes returned from Encode()/EndEncode()
    are a real MP4/MKV stream (Lavf muxed), suitable for normal playback.
    """

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "hevc",
        preset: str = "P7",
        profile: str = "main",
        qp: int = 23,
        gpu_id: int = 0,
        container: str | None = None,  # NEW: "mp4" / "mkv" / None (raw)
    ):
        self.output_path = str(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec = str(codec).lower()
        self.preset = str(preset)
        self.profile = str(profile)
        self.qp = int(qp)
        self.gpu_id = int(gpu_id)

        # NEW: muxing selection
        self.container: str | None = container if container is not None else _infer_container(self.output_path)

        # Suppress PyNvVideoCodec warnings (like preset warnings)
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning)

        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid encoder resolution {self.width}x{self.height}")

        # Open output file (we always write bytes ourselves)
        self._file = open(self.output_path, "wb")

        # Encoder configuration
        # CRITICAL: "ARGB" format expects BGRA memory layout (little-endian)
        enc_opts = {
            "codec": self.codec,       # "hevc" or "h264"
            "preset": self.preset,     # "P1"-"P7" (case-sensitive)
            "profile": self.profile,   # "main", "main10", etc.
            "qp": str(self.qp),        # Quantization parameter
            "fps": f"{self.fps:g}",    # Framerate
        }

        # GOP (keyframe interval) - every 2 seconds
        gop_frames = max(1, int(round(self.fps * 2.0)))
        enc_opts["gop"] = str(gop_frames)
        enc_opts["idrperiod"] = str(gop_frames)

        print(f"[Encoder] Creating: {self.output_path}")
        print(f"[Encoder] Resolution: {self.width}x{self.height} @ {self.fps:.2f} fps")
        print(f"[Encoder] Codec: {self.codec}, Preset: {self.preset}, QP: {self.qp}")
        print(f"[Encoder] Format: ARGB (expects BGRA memory layout)")
        print(f"[Encoder] GOP: {gop_frames} frames")
        if self.container:
            print(f"[Encoder] Container muxing: {self.container} (Lavf/libavformat inside PyNvVideoCodec)")
        else:
            print(f"[Encoder] Container muxing: NONE (raw elementary bitstream)")

        fmt = "ARGB"  # word-ordered => BGRA byte order on little-endian

        # NEW: pass container kwarg when requested
        if self.container:
            self._encoder = nvc.CreateEncoder(
                self.width,
                self.height,
                fmt,
                False,  # use_cpu_input_buffer=False -> GPU input
                container=self.container,
                **enc_opts,
            )
        else:
            self._encoder = nvc.CreateEncoder(
                self.width,
                self.height,
                fmt,
                False,  # use_cpu_input_buffer=False -> GPU input
                **enc_opts,
            )

        self._frames_encoded = 0

    def encode_frame(self, frame: Any) -> None:
        if frame is None:
            return

        try:
            bitstream = self._encoder.Encode(frame)
        except Exception as e:
            print(f"[Encoder] ERROR: Encode failed: {e}")
            raise

        if bitstream:
            self._file.write(bytearray(bitstream))
            self._frames_encoded += 1

    def encode_frames(self, frames: Iterable[Any]) -> None:
        for frame in frames:
            self.encode_frame(frame)

    def flush(self) -> None:
        print(f"[Encoder] Flushing... ({self._frames_encoded} frames encoded)")
        try:
            tail = self._encoder.EndEncode()
        except Exception as e:
            print(f"[Encoder] ERROR: EndEncode failed: {e}")
            return

        if tail:
            self._file.write(bytearray(tail))

    def close(self) -> None:
        print(f"[Encoder] Closing output file ({self._frames_encoded} total frames)")
        try:
            self._file.close()
        except Exception:
            pass

    @property
    def frames_encoded(self) -> int:
        return self._frames_encoded

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self.close()

    def __repr__(self) -> str:
        return (
            f"Encoder(path='{self.output_path}', "
            f"{self.width}x{self.height}, "
            f"{self.codec}, {self._frames_encoded} frames, "
            f"container={self.container!r})"
        )
