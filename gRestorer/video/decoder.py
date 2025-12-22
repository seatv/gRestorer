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
        
        print(f"[Decoder] Initialized: {self.metadata.width}x{self.metadata.height}, "
              f"{self.metadata.num_frames} frames, {self.metadata.fps:.2f} fps")
        print(f"[Decoder] Output: RGB [H,W,3] on GPU {self.gpu_id}")
    
    def read_batch(self) -> List:
        """
        Read next batch of frames.

        Returns:
            List of RGB frames as GPU surfaces [H, W, 3].
            Empty list if no more frames.

        Note:
            Frames are DLPack-compatible GPU memory that can be wrapped
            with torch.from_dlpack() for zero-copy access.
        """
        # If container reports a finite frame count, avoid calling into decoder past EOF.
        frames = None
        if not self.is_complete:
            frames = self._decoder.get_batch_frames(self.batch_size)

            if frames:
                self._frames_read += len(frames)

        return frames


    @property
    def num_frames(self) -> int:
        """Total number of frames in video."""
        return self.metadata.num_frames

    @property
    def frames_read(self) -> int:
        """Number of frames read so far."""
        return self._frames_read

    @property
    def is_complete(self) -> bool:
        """Check if all frames have been read."""
        return (self.metadata.num_frames > 0) and (self._frames_read >= self.metadata.num_frames)
    
    def __repr__(self) -> str:
        return (f"Decoder(path='{self.input_path}', "
                f"{self.metadata.width}x{self.metadata.height}, "
                f"{self.metadata.num_frames} frames)")

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
