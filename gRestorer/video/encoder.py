"""
Video encoder using NVIDIA PyNvVideoCodec.

Provides hardware-accelerated video encoding with GPU input.
"""

from __future__ import annotations

from typing import Any, Iterable
from pathlib import Path

import PyNvVideoCodec as nvc


class Encoder:
    """
    Hardware-accelerated video encoder using NVENC.
    
    Encodes video from GPU memory surfaces.
    Expects BGRA format [H, W, 4] input (mapped to "ARGB" encoder format).
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
        gpu_id: int = 0
    ):
        """
        Initialize encoder.
        
        Args:
            output_path: Path to output video file
            width: Frame width
            height: Frame height
            fps: Frames per second
            codec: Video codec ('hevc' or 'h264')
            preset: NVENC preset (P1-P7, higher is better quality)
            profile: Codec profile
            qp: Quantization parameter (lower = better quality, 0 = lossless)
            gpu_id: CUDA device ID
        """
        self.output_path = str(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec.lower()
        self.preset = preset
        self.profile = profile
        self.qp = qp
        self.gpu_id = gpu_id
        
        # Suppress PyNvVideoCodec warnings (like preset warnings)
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Open output file
        self._file = open(self.output_path, 'wb')
        
        # Encoder configuration
        # CRITICAL: "ARGB" format expects BGRA memory layout (see Table 2)
        enc_opts = {
            'codec': self.codec,      # 'hevc' or 'h264'
            'preset': self.preset,    # P1-P7
            'profile': self.profile,  # 'main', 'main10', etc.
            'qp': str(self.qp),       # Quantization parameter
            'fps': f"{self.fps:g}",   # Framerate
        }
        
        # Set GOP (keyframe interval) - every 2 seconds
        gop_frames = max(1, int(round(self.fps * 2.0)))
        enc_opts['gop'] = str(gop_frames)
        enc_opts['idrperiod'] = str(gop_frames)
        
        print(f"[Encoder] Creating: {output_path}")
        print(f"[Encoder] Resolution: {width}x{height} @ {fps:.2f} fps")
        print(f"[Encoder] Codec: {self.codec}, Preset: {preset}, QP: {qp}")
        print(f"[Encoder] Format: ARGB (expects BGRA memory layout)")
        print(f"[Encoder] GOP: {gop_frames} frames")
        
        # Create PyNvVideoCodec encoder
        # Format "ARGB" expects BGRA byte order in memory (little-endian)
        fmt = "ARGB"  # word-ordered => BGRA byte order on little-endian
        self._encoder = nvc.CreateEncoder(
            self.width,
            self.height,
            fmt,
            False,   # use_cpu_input_buffer=False -> GPU input
            **enc_opts
        )
        
        self._frames_encoded = 0
    
    def encode_frame(self, frame: Any) -> None:
        """
        Encode a single frame.
        
        Args:
            frame: GPU surface [H, W, 4] in BGRA format
                   Compatible with PyNvVideoCodec/DLPack
        """
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
        """
        Encode multiple frames.
        
        Args:
            frames: Iterable of GPU surfaces [H, W, 4] in BGRA format
        """
        for frame in frames:
            self.encode_frame(frame)
    
    def flush(self) -> None:
        """
        Flush encoder buffer and write remaining frames.
        
        Must be called before closing to ensure all frames are written.
        """
        print(f"[Encoder] Flushing... ({self._frames_encoded} frames encoded)")
        
        try:
            tail = self._encoder.EndEncode()
        except Exception as e:
            print(f"[Encoder] ERROR: EndEncode failed: {e}")
            return
        
        if tail:
            self._file.write(bytearray(tail))
    
    def close(self) -> None:
        """Close output file."""
        print(f"[Encoder] Closing output file ({self._frames_encoded} total frames)")
        try:
            self._file.close()
        except Exception:
            pass
    
    @property
    def frames_encoded(self) -> int:
        """Number of frames encoded so far."""
        return self._frames_encoded
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self.close()
    
    def __repr__(self) -> str:
        return (f"Encoder(path='{self.output_path}', "
                f"{self.width}x{self.height}, "
                f"{self.codec}, {self._frames_encoded} frames)")