from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

@dataclass
class FaceSwapBackendResult:
    aligned_swapped_bgr_u8: np.ndarray
    aligned_target_bgr_u8: Optional[np.ndarray]
    aligned_backend_mask_f32: Optional[np.ndarray]
    roi_to_aligned: np.ndarray
    aligned_to_roi: np.ndarray
    aligned_size: int
    debug: Optional[dict[str, Any]] = None


@dataclass
class FaceCompositorConfig:
    mask_mode: str = "geom_backend_intersection"
    geom_expand: float = 1.05
    mask_erode: int = 0
    mask_dilate: int = 2
    mask_blur: int = 5
    blend_mode: str = "alpha"
    color_transfer: str = "none"
    face_scale: float = 0.0
    debug: bool = False

__all__ = ["FaceSwapBackendResult", "FaceCompositorConfig"]
    