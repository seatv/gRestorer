from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FaceOcclusionMaskResult:
    keep_mask_f32: np.ndarray
    aligned_input_bgr_u8: Optional[np.ndarray] = None
    aligned_raw_mask_f32: Optional[np.ndarray] = None
    aligned_keep_mask_f32: Optional[np.ndarray] = None


__all__ = ["FaceOcclusionMaskResult"]
