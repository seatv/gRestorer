from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass
class FaceSwapBackendResult:
    # New strict aligned-space contract
    swapped_face_f32: np.ndarray
    pred_src_mask_f32: Optional[np.ndarray]
    pred_dst_mask_f32: Optional[np.ndarray]
    aligned_target_f32: Optional[np.ndarray]
    target_landmarks_aligned: Optional[np.ndarray]
    source_landmarks_aligned: Optional[np.ndarray]
    roi_to_aligned: np.ndarray
    aligned_to_roi: np.ndarray
    aligned_size: int
    quality: Optional[float] = None
    backend: Optional[str] = None
    debug: Optional[dict[str, Any]] = None

    # Transitional compatibility helpers so existing debug code keeps working.
    @staticmethod
    def _f32_to_u8(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if x is None:
            return None
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HWC image, got shape={tuple(arr.shape)}")
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0.0, 255.0).round().astype(np.uint8)

    @property
    def aligned_swapped_bgr_u8(self) -> np.ndarray:
        return self._f32_to_u8(self.swapped_face_f32)

    @property
    def aligned_target_bgr_u8(self) -> Optional[np.ndarray]:
        return self._f32_to_u8(self.aligned_target_f32)

    @property
    def aligned_backend_mask_f32(self) -> Optional[np.ndarray]:
        return self.pred_src_mask_f32


@dataclass
class FaceCompositorConfig:
    mask_mode: str = "geom"
    geom_expand: float = 1.00
    mask_erode: int = 0
    mask_dilate: int = 0
    mask_blur: int = 3
    blend_mode: str = "alpha"
    color_transfer: str = "none"
    face_scale: float = 0.0
    debug: bool = False

@dataclass
class FaceOcclusionMaskResult:
    keep_mask_f32: np.ndarray
    aligned_input_bgr_u8: Optional[np.ndarray] = None
    aligned_raw_mask_f32: Optional[np.ndarray] = None
    aligned_keep_mask_f32: Optional[np.ndarray] = None

__all__ = ["FaceSwapBackendResult", "FaceCompositorConfig", "FaceOcclusionMaskResult"]
