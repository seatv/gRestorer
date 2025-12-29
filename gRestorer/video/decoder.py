"""
Video decoder using NVIDIA PyNvVideoCodec.

Provides hardware-accelerated video decoding with GPU output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

import PyNvVideoCodec as nvc


@dataclass
class VideoMetadata:
    """Metadata for decoded video stream."""
    width: int
    height: int
    bit_depth: int
    num_frames: int
    fps: Optional[float] = None
    duration: Optional[float] = None
    bitrate: Optional[float] = None
    codec_name: Optional[str] = None


class Decoder:
    """
    Hardware-accelerated video decoder using NVDEC.
    
    Decodes video to RGB format [H, W, 3] on GPU.
    Uses PyNvVideoCodec for direct GPU memory output.
    """
    
    def __init__(
        self,
        input_path: str | Path,
        gpu_id: int = 0,
        batch_size: int = 80
    ):
        """
        Initialize decoder.
        
        Args:
            input_path: Path to input video file
            gpu_id: CUDA device ID
            batch_size: Number of frames to decode per batch
        """
        self.input_path = str(input_path)
        self.gpu_id = gpu_id
        self.batch_size = max(1, batch_size)
        
        # Suppress PyNvVideoCodec warnings (like INVALID INDEX)
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Create PyNvVideoCodec decoder
        # Output RGB format for processing (will convert to BGR immediately)
        self._decoder = nvc.SimpleDecoder(
            enc_file_path=self.input_path,
            gpu_id=self.gpu_id,
            output_color_type=nvc.OutputColorType.RGBP,  # RGB output from NVDEC
            use_device_memory=True,  # Keep frames on GPU
            decoder_cache_size=self.batch_size,
            need_scanned_stream_metadata=True
        )
        
        # Extract metadata
        meta = self._decoder.get_stream_metadata()
        self.metadata = VideoMetadata(
            width=getattr(meta, 'width', 0),
            height=getattr(meta, 'height', 0),
            bit_depth=getattr(meta, 'bit_depth', 8),
            num_frames=getattr(meta, 'num_frames', 0),
            fps=getattr(meta, 'average_fps', getattr(meta, 'fps', None)),
            duration=getattr(meta, 'duration_in_seconds', None),
            bitrate=getattr(meta, 'bitrate', None),
            codec_name=getattr(meta, 'codec_name', None)
        )
        
        self._frames_read = 0
        
        self._raw_num_frames = self.metadata.num_frames
        self._trim_prefix = 0
        self._prefetch: list = []
        self._trim_negative_pts = True

        if self._trim_negative_pts:
            self._prime_to_first_nonneg_pts()

        
        print(f"[Decoder] Initialized: {self.metadata.width}x{self.metadata.height}, "
              f"{self.metadata.num_frames} frames, {self.metadata.fps:.2f} fps")
        print(f"[Decoder] Output: RGB [H,W,3] on GPU {self.gpu_id}")

    def read_batch(self) -> List:
        """Read next batch of frames (presented frames only). Returns [] at EOF."""
        n = self.batch_size

        # Use RAW frame count for decoder EOF protection.
        if self._raw_num_frames > 0:
            remaining_raw = self._raw_num_frames - self._frames_read
            if remaining_raw <= 0 and not self._prefetch:
                return []
            n = min(n, max(0, remaining_raw)) if remaining_raw > 0 else n

        out: list = []

        # Serve any prefetched frames first
        if self._prefetch:
            take = min(n, len(self._prefetch))
            out.extend(self._prefetch[:take])
            self._prefetch = self._prefetch[take:]
            n -= take

        # Then pull more from NVDEC
        if n > 0:
            frames = self._decoder.get_batch_frames(n)
            if frames:
                self._frames_read += len(frames)  # RAW frames consumed
                out.extend(frames)

        if not out:
            return []

        # Safety: if any negative-PTS frames slip through (shouldn't), drop them.
        if self._trim_negative_pts:
            filtered = []
            for fr in out:
                pts = self._frame_pts(fr)
                if pts is None or pts >= 0:
                    filtered.append(fr)
            out = filtered

        return out

    @property
    def num_frames(self) -> int:
        """Presented frame count (raw minus trimmed negative-PTS prefix)."""
        if self._raw_num_frames > 0:
            return max(0, self._raw_num_frames - self._trim_prefix)
        return self.metadata.num_frames


    @property
    def frames_read(self) -> int:
        """Number of frames read so far."""
        return self._frames_read

    @property
    def is_complete(self) -> bool:
        """Check if all frames have been read."""
        return (self._raw_num_frames > 0) and (self._frames_read >= self._raw_num_frames)
    
    def __repr__(self) -> str:
        return (f"Decoder(path='{self.input_path}', "
                f"{self.metadata.width}x{self.metadata.height}, "
                f"{self.metadata.num_frames} frames)")


    def _frame_pts(self, fr) -> int | None:
        try:
            return int(fr.getPTS())
        except Exception:
            try:
                return int(getattr(fr, "timestamp"))
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

        # Read forward until we hit first frame with PTS >= 0.
        # Keep the first non-neg batch tail in _prefetch so we don't lose it.
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


    def close(self) -> None:
        """
        PyNvVideoCodec decoder objects typically free resources on GC.
        We still provide an explicit close() so the pipeline can deterministically
        drop references and release GPU memory sooner.
        """
        # Drop references so Python can destroy underlying objects
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
