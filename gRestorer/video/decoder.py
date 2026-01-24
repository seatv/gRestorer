from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import os
import subprocess
from fractions import Fraction

import numpy as np
import torch
import PyNvVideoCodec as nvc


@dataclass
class VideoMetadata:
    width: int
    height: int
    bit_depth: int
    num_frames: int
    fps: Optional[float]
    duration: Optional[float]
    bitrate: Optional[float]
    codec_name: Optional[str]


class Decoder:
    """
    GPU-first video decoder.

    - Primary backend: PyNvVideoCodec (NVDEC) outputting RGBP in device memory.
    - Fallback backend: ffmpeg CPU decode to raw RGB24 (HWC uint8).

    Why the fallback matters:
      Some NVDEC-capable GPUs (e.g., Tesla P4) cannot decode frames wider than 4096px.
      Many VR SBS sources are 4320x2160. NVDEC will fail with:
        "Resolution not supported on this GPU ... Max Supported 4096x4096"
      In that case we transparently fall back to ffmpeg CPU decode so the pipeline can proceed.
    """

    def __init__(
        self,
        input_path: str,
        gpu_id: int = 0,
        batch_size: int = 80,
        trim_negative_pts: bool = True,
    ) -> None:
        self.input_path = str(Path(input_path))
        self.gpu_id = int(gpu_id)
        self.batch_size = int(batch_size)

        # Allow an escape hatch for debugging / flaky sources.
        # (No CLI flag required; set env var to force CPU decode.)
        self._force_cpu = os.environ.get("GR_FORCE_CPU_DECODE", "").strip() in ("1", "true", "True", "YES", "yes")

        self.backend: str = "nvdec"
        self.metadata: VideoMetadata

        self._decoder: Any = None  # NVDEC decoder
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._ffmpeg_frame_size: int = 0

        # Accounting for trimming / prefetch
        self._raw_num_frames: int = 0
        self._frames_read: int = 0
        self._prefetch: List[Any] = []
        self._trim_prefix: int = 0
        self._trim_negative_pts: bool = bool(trim_negative_pts)

        # Init backend
        if self._force_cpu:
            self.backend = "ffmpeg-cpu"
            self._init_ffmpeg_cpu_backend()
        else:
            try:
                # Output RGBP (planar RGB) for fast GPU interop (we convert to HWC later)
                self._decoder = nvc.SimpleDecoder(
                    enc_file_path=self.input_path,
                    gpu_id=self.gpu_id,
                    output_color_type=nvc.OutputColorType.RGBP,
                    use_device_memory=True,
                    decoder_cache_size=self.batch_size,
                    need_scanned_stream_metadata=False,
                )
                self.backend = "nvdec"
            except Exception as e:
                if self._looks_like_nvdec_unsupported(e):
                    print(
                        f"[Decoder] NVDEC unsupported for this stream on GPU {self.gpu_id}; "
                        f"falling back to ffmpeg CPU decode. ({e})"
                    )
                    self.backend = "ffmpeg-cpu"
                    self._decoder = None
                    self._init_ffmpeg_cpu_backend()
                else:
                    raise

        # Extract metadata
        if self.backend == "nvdec":
            meta = self._decoder.get_stream_metadata()
            self.metadata = VideoMetadata(
                width=int(getattr(meta, "width", 0) or 0),
                height=int(getattr(meta, "height", 0) or 0),
                bit_depth=int(getattr(meta, "bit_depth", 8) or 8),
                num_frames=int(getattr(meta, "num_frames", 0) or 0),
                fps=float(getattr(meta, "average_fps", getattr(meta, "fps", 0)) or 0) or None,
                duration=getattr(meta, "duration_in_seconds", None),
                bitrate=float(getattr(meta, "bitrate", 0) or 0) or None,
                codec_name=getattr(meta, "codec_name", None),
            )
        else:
            # _init_ffmpeg_cpu_backend() fills self.metadata
            pass

        self._raw_num_frames = int(self.metadata.num_frames or 0)

        # Optional: trim negative-PTS preroll for NVDEC backend
        if self._trim_negative_pts and self.backend == "nvdec":
            # NOTE: This can throw on some streams; if it does and it's an unsupported-res issue,
            # fall back to CPU decode.
            try:
                self._prime_to_first_nonneg_pts()
            except Exception as e:
                if self._looks_like_nvdec_unsupported(e):
                    print(
                        f"[Decoder] NVDEC failed during preroll trim; falling back to ffmpeg CPU decode. ({e})"
                    )
                    self.backend = "ffmpeg-cpu"
                    self._decoder = None
                    self._init_ffmpeg_cpu_backend()
                    self._raw_num_frames = int(self.metadata.num_frames or 0)
                else:
                    raise

        # Status
        fps_s = f"{self.metadata.fps:.2f}" if self.metadata.fps else "?"
        nf_s = str(self.metadata.num_frames) if self.metadata.num_frames else "?"
        print(f"[Decoder] Initialized ({self.backend}): {self.metadata.width}x{self.metadata.height}, {nf_s} frames, {fps_s} fps")
        if self.backend == "nvdec":
            print(f"[Decoder] Output: RGB [H,W,3] on GPU {self.gpu_id}")
        else:
            print(f"[Decoder] Output: RGB [H,W,3] on CPU (ffmpeg)")

    # -------------------------
    # Public API used by pipeline
    # -------------------------
    @property
    def num_frames(self) -> int:
        if self._raw_num_frames <= 0:
            return 0
        return max(0, self._raw_num_frames - self._trim_prefix)

    def is_complete(self) -> bool:
        if self.backend != "nvdec":
            # ffmpeg: complete when process ended and no more stdout
            return self._ffmpeg_proc is None
        if self._raw_num_frames <= 0:
            return False
        return self._frames_read >= self._raw_num_frames

    def read_batch(self) -> List[Any]:
        """Read next batch of frames (surfaces or tensors depending on backend)."""
        if self.backend != "nvdec":
            n = self.batch_size
            out: List[torch.Tensor] = []
            for _ in range(n):
                fr = self._ffmpeg_read_frame()
                if fr is None:
                    # EOF: shut down proc deterministically
                    self.close()
                    break
                self._frames_read += 1
                out.append(fr)
            return out

        # NVDEC path: get_batch_frames (GPU surfaces)
        n = self.batch_size

        if self._raw_num_frames > 0:
            remaining_raw = self._raw_num_frames - self._frames_read
            if remaining_raw <= 0 and not self._prefetch:
                return []
            if remaining_raw > 0:
                n = min(n, remaining_raw)

        if self._prefetch:
            frames = self._prefetch
            self._prefetch = []
            if len(frames) > n:
                out = frames[:n]
                self._prefetch = frames[n:]
                return out
            return frames

        frames = self._decoder.get_batch_frames(n)
        if not frames:
            return []
        self._frames_read += len(frames)
        return frames

    def close(self) -> None:
        """Explicitly release decoder resources."""
        # ffmpeg backend
        if self.backend != "nvdec":
            try:
                if self._ffmpeg_proc is not None:
                    try:
                        if self._ffmpeg_proc.stdout:
                            self._ffmpeg_proc.stdout.close()
                    except Exception:
                        pass
                    try:
                        if self._ffmpeg_proc.stderr:
                            self._ffmpeg_proc.stderr.close()
                    except Exception:
                        pass
                    try:
                        self._ffmpeg_proc.terminate()
                    except Exception:
                        pass
                    try:
                        self._ffmpeg_proc.wait(timeout=2)
                    except Exception:
                        pass
            finally:
                self._ffmpeg_proc = None
            return

        # NVDEC backend: drop references so Python can destroy underlying objects
        for attr in ("_decoder", "_demuxer", "_reader", "_ctx", "_stream"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # -------------------------
    # NVDEC helpers
    # -------------------------
    @staticmethod
    def _looks_like_nvdec_unsupported(e: Exception) -> bool:
        msg = str(e)
        return (
            ("Resolution not supported" in msg)
            or ("Error code : 801" in msg)
            or ("PyNvVCExceptionUnsupported" in msg)
        )

    def _frame_pts(self, frame: Any) -> Optional[int]:
        try:
            p = frame.pts()
            if p is None:
                return None
            if hasattr(p, "value"):
                return int(p.value)
            return int(p)
        except Exception:
            return None

    def _prime_to_first_nonneg_pts(self) -> None:
        """
        Some files have an edit-list / preroll region with negative PTS.
        Those frames are not meant to be presented, and encoding them causes
        duration inflation + audio drift. We trim the negative-PTS prefix.
        """
        if self._raw_num_frames <= 0:
            return

        scan_batch = max(8, min(128, self.batch_size))
        while True:
            remaining = self._raw_num_frames - self._frames_read
            if remaining <= 0:
                return

            n = min(scan_batch, remaining)
            frames = self._decoder.get_batch_frames(n)
            if not frames:
                return

            self._frames_read += len(frames)  # RAW frames consumed from decoder

            first_ok = None
            for i, fr in enumerate(frames):
                pts = self._frame_pts(fr)
                if pts is None:
                    # If PTS is unavailable, we can't trim safely.
                    return
                if pts >= 0:
                    first_ok = i
                    break

            if first_ok is None:
                # whole batch is negative PTS
                self._trim_prefix += len(frames)
                continue

            # Found first non-negative PTS inside this batch:
            self._trim_prefix += first_ok
            self._prefetch = frames[first_ok:]

            if self._trim_prefix > 0:
                presented = max(0, self._raw_num_frames - self._trim_prefix)
                print(f"[Decoder] Trimmed {self._trim_prefix} negative-PTS preroll frames (presented={presented})")
            return

    # -------------------------
    # ffmpeg CPU fallback backend
    # -------------------------
    def _ffprobe(self) -> VideoMetadata:
        """Probe basic video metadata using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,codec_name,bit_rate",
            "-of", "default=nw=1",
            self.input_path,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffprobe failed:\n{p.stderr}")

        kv: dict[str, str] = {}
        for ln in p.stdout.splitlines():
            if "=" in ln:
                k, v = ln.strip().split("=", 1)
                kv[k] = v

        w = int(kv.get("width", "0") or 0)
        h = int(kv.get("height", "0") or 0)

        fps = None
        afr = kv.get("avg_frame_rate")
        if afr and afr != "0/0":
            try:
                fps = float(Fraction(afr))
            except Exception:
                fps = None

        codec = kv.get("codec_name") or None

        bitrate = None
        try:
            bitrate = float(kv.get("bit_rate")) if kv.get("bit_rate") else None
        except Exception:
            bitrate = None

        return VideoMetadata(
            width=w,
            height=h,
            bit_depth=8,
            num_frames=0,  # unknown without costly count_frames
            fps=fps,
            duration=None,
            bitrate=bitrate,
            codec_name=codec,
        )

    def _init_ffmpeg_cpu_backend(self) -> None:
        """Start an ffmpeg process that outputs raw RGB24 frames on stdout."""
        self.metadata = self._ffprobe()
        if not self.metadata.width or not self.metadata.height:
            raise RuntimeError("ffprobe did not return width/height; cannot CPU-decode")

        w = int(self.metadata.width)
        h = int(self.metadata.height)
        self._ffmpeg_frame_size = w * h * 3  # rgb24

        print(f"[Decoder] Backend: ffmpeg-cpu  output=RGB24(HWC,u8)  {w}x{h}")

        # -vsync 0 avoids frame duplication/drop when dumping rawvideo
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-fflags", "+genpts",
            "-i", self.input_path,
            "-an", "-sn", "-dn",
            "-vsync", "0",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8,
        )

    def _ffmpeg_read_frame(self) -> torch.Tensor | None:
        """Read one RGB24 frame from ffmpeg stdout.

        Returns:
          HWC uint8 CPU tensor (shares memory with a per-frame bytearray), or None at EOF.

        Note:
          We intentionally avoid np.frombuffer(bytes) -> non-writable NumPy arrays, which triggers
          a PyTorch warning. Using a per-frame bytearray keeps the buffer writable without adding
          an extra full-frame copy.
        """
        if self._ffmpeg_proc is None or self._ffmpeg_proc.stdout is None:
            return None

        # Read exactly one frame into a writable buffer.
        buf = bytearray(self._ffmpeg_frame_size)
        view = memoryview(buf)
        got = 0
        while got < self._ffmpeg_frame_size:
            n = self._ffmpeg_proc.stdout.readinto(view[got:])
            if not n:
                return None
            got += int(n)

        arr = np.frombuffer(buf, dtype=np.uint8).reshape(
            (int(self.metadata.height), int(self.metadata.width), 3)
        )
        return torch.from_numpy(arr)
