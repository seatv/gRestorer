"""
Video encoder using NVIDIA PyNvVideoCodec.

Design goals for gRestorer:
- GPU input (BGRA) -> NVENC bitstream (H.264/HEVC)
- Stable CFR timing metadata for MP4/MKV so ffprobe/player don't guess wrong FPS
- Avoid non-monotonic DTS/PTS issues that cause flicker / warnings when extracting frames

Strategy:
- Encode to an elementary stream (raw .h264/.hevc) for maximum compatibility.
- If the requested output is a container (.mp4/.mkv), remux at close with ffmpeg using stream copy
  and generated timestamps.

Why not rely only on PyNvVideoCodec internal muxing?
- In practice, MP4 timing metadata can vary across versions/drivers/players and may yield
  avg_frame_rate mismatches or missing duration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import subprocess

import PyNvVideoCodec as nvc


def _infer_container(output_path: str | Path) -> Optional[str]:
    """Infer container muxing from file extension."""
    suf = Path(output_path).suffix.lower()
    if suf in (".mp4", ".m4v", ".mov"):
        return "mp4"
    if suf == ".mkv":
        return "mkv"
    return None


def _raw_ext_for_codec(codec: str) -> str:
    c = codec.lower()
    if c in ("hevc", "h265"):
        return ".hevc"
    if c in ("h264", "avc"):
        return ".h264"
    # fall back to generic
    return ".bit"


def _ffmpeg_input_format(codec: str) -> str:
    c = codec.lower()
    if c in ("hevc", "h265"):
        return "hevc"
    if c in ("h264", "avc"):
        return "h264"
    raise ValueError(f"Unsupported codec for ffmpeg remux: {codec!r}")


class Encoder:
    """
    Hardware-accelerated video encoder using NVENC (PyNvVideoCodec).

    Encodes from GPU memory surfaces.
    Expects BGRA format [H, W, 4] input (mapped to "ARGB" encoder format).

    Parameters
    ----------
    output_path:
        Final output file. If extension is .mp4 or .mkv, we will remux at close.
    codec:
        "hevc" or "h264"
    preset/profile/qp:
        NVENC tuning knobs.
    bframes:
        We default to 0 to avoid DTS/PTS reorder issues when remuxing raw streams.
        (Disabling B-frames also tends to reduce latency and can improve stability.)
    ffmpeg_path:
        Path to ffmpeg executable used for remux (default: "ffmpeg")
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
        container: str | None = None,
        bframes: int = 0,
        ffmpeg_path: str = "ffmpeg",
        input_path: str | Path | None = None,
    ):
        self.output_path = str(output_path)
        self.input_path = str(input_path) if input_path else None
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec = str(codec).lower()
        self.preset = str(preset)
        self.profile = str(profile)
        self.qp = int(qp)
        self.gpu_id = int(gpu_id)
        self.bframes = int(bframes)
        self.ffmpeg_path = str(ffmpeg_path)

        self.container: str | None = container if container is not None else _infer_container(self.output_path)

        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid encoder resolution {self.width}x{self.height}")
        if self.fps <= 0:
            raise ValueError(f"Invalid fps: {self.fps}")

        # Decide where we write bytes
        self._raw_path = self.output_path
        self._needs_remux = False
        if self.container in ("mp4", "mkv"):
            self._needs_remux = True
            self._raw_path = self.output_path + _raw_ext_for_codec(self.codec)

        self._file = open(self._raw_path, "wb")

        # Encoder configuration
        # CRITICAL: "ARGB" format expects BGRA memory layout (little-endian)
        enc_opts: dict[str, str] = {
            "codec": self.codec,        # "hevc" or "h264"
            "preset": self.preset,      # "P1"-"P7" (case-sensitive)
            "profile": self.profile,    # "main", "main10", etc.
            "qp": str(self.qp),         # Quantization parameter
            "fps": f"{self.fps:g}",     # Framerate
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
        if self._needs_remux:
            print(f"[Encoder] Container: {self.container} (ffmpeg remux at close)")
            print(f"[Encoder] Raw bitstream: {self._raw_path}")
        else:
            print(f"[Encoder] Container: NONE (raw elementary bitstream)")

        fmt = "ARGB"  # word-ordered => BGRA byte order on little-endian

        # Try to set B-frames if supported; fall back cleanly if not.
        self._encoder = None
        last_err: Exception | None = None
        for key in ("bf", "bframes"):
            try:
                opts_try = dict(enc_opts)
                if self.bframes >= 0:
                    opts_try[key] = str(self.bframes)
                self._encoder = nvc.CreateEncoder(self.width, self.height, fmt, False, **opts_try)
                last_err = None
                if self.bframes >= 0:
                    print(f"[Encoder] B-frames: {self.bframes} (key='{key}')")
                break
            except Exception as e:
                last_err = e
                self._encoder = None

        if self._encoder is None:
            # Fall back without any bframe option
            try:
                self._encoder = nvc.CreateEncoder(self.width, self.height, fmt, False, **enc_opts)
                if self.bframes >= 0:
                    print(f"[Encoder] B-frames: (option not supported by this PyNvVideoCodec build)")
            except Exception as e:
                print(f"[Encoder] ERROR: CreateEncoder failed: {e}")
                if last_err is not None:
                    print(f"[Encoder] (Last bframes option error was: {last_err})")
                raise

        self._frames_encoded = 0

    def encode_frame(self, frame: Any) -> None:
        if frame is None:
            return
        bitstream = self._encoder.Encode(frame)
        if bitstream:
            self._file.write(bytearray(bitstream))
            self._frames_encoded += 1

    def encode_frames(self, frames: Iterable[Any]) -> None:
        for fr in frames:
            self.encode_frame(fr)

    def flush(self) -> None:
        print(f"[Encoder] Flushing... ({self._frames_encoded} frames encoded)")
        try:
            tail = self._encoder.EndEncode()
        except Exception as e:
            print(f"[Encoder] ERROR: EndEncode failed: {e}")
            return
        if tail:
            self._file.write(bytearray(tail))

    def _remux_with_ffmpeg(self, input_path: str | None = None) -> None:
        """
        Remux raw HEVC bitstream (.hevc) into a playable MP4 container.
        Optionally copies audio and subtitle tracks from the original input.
        """
        import subprocess
        from pathlib import Path
        import shlex

        hevc_path = Path(self.output_path).with_suffix(".mp4.hevc")
        mp4_path = Path(self.output_path)
        input_video = Path(input_path) if input_path else None

        if not hevc_path.exists():
            print(f"[Encoder] Remux skipped: {hevc_path} not found")
            return

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-loglevel", "error",
            "-fflags", "+genpts",
            "-r", str(self.fps),
            "-f", "hevc",
            "-i", str(hevc_path),
        ]

        if input_video and input_video.exists():
            # Add original input as second input to copy its audio/subs
            cmd += ["-i", str(input_video)]
            cmd += [
                "-map", "0:v:0", "-c:v", "copy",
                "-map", "1:a?", "-c:a", "copy",
                "-map", "1:s?", "-c:s", "copy",
            ]
        else:
            # Video-only (no audio input)
            cmd += ["-an", "-c:v", "copy"]

        cmd += [
            "-movflags", "+faststart",
            "-video_track_timescale", "90000",
            str(mp4_path),
        ]

        print("[Encoder] Remux:", " ".join(shlex.quote(x) for x in cmd))

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Encoder] Remux failed: {e}")
        else:
            # Clean up the raw HEVC if remux succeeded
            try:
                hevc_path.unlink()
            except OSError:
                pass

    def close(self) -> None:
        # Ensure trailer bytes are written
        self.flush()
        print(f"[Encoder] Closing output file ({self._frames_encoded} total frames)")
        try:
            self._file.close()
        except Exception:
            pass

        # Remux if needed
        self._remux_with_ffmpeg(self.input_path)

    @property
    def frames_encoded(self) -> int:
        return self._frames_encoded

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return (
            f"Encoder(path='{self.output_path}', "
            f"{self.width}x{self.height}, "
            f"{self.codec}, {self._frames_encoded} frames, "
            f"container={self.container!r}, needs_remux={self._needs_remux})"
        )
