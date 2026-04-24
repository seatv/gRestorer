from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple

import cv2
import numpy as np
import torch


@dataclass
class MosaicPasteDebug:
    enabled: bool
    out_dir: Path
    frames: Set[int]
    start: int = -1
    end: int = -1

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        if self.enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def should_dump(self, frame_num: int) -> bool:
        if not self.enabled:
            return False
        frame_num = int(frame_num)
        if self.frames:
            return frame_num in self.frames
        if self.start >= 0 and frame_num < self.start:
            return False
        if self.end >= 0 and frame_num > self.end:
            return False
        return True

    def dump(
        self,
        *,
        frame_num: int,
        clip_id: int,
        crop_box: Tuple[int, int, int, int],
        original_roi: torch.Tensor,
        restored_roi: torch.Tensor,
        resized_mask: torch.Tensor,
        legacy_alpha: torch.Tensor,
        final_roi: torch.Tensor,
        actual_alpha: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.should_dump(frame_num):
            return

        stem = self.out_dir / f"f{int(frame_num):06d}_clip{int(clip_id):04d}"
        meta = {
            "frame_num": int(frame_num),
            "clip_id": int(clip_id),
            "crop_box_tlbr": [int(v) for v in crop_box],
            "roi_hw": [int(original_roi.shape[0]), int(original_roi.shape[1])],
        }
        (stem.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        self._imwrite(stem.with_name(stem.name + "_orig_roi.png"), self._to_bgr_u8(original_roi))
        self._imwrite(stem.with_name(stem.name + "_restored_roi.png"), self._to_bgr_u8(restored_roi))
        self._imwrite(stem.with_name(stem.name + "_mask_resized.png"), self._to_gray_u8(resized_mask))
        self._imwrite(stem.with_name(stem.name + "_alpha_legacy.png"), self._to_alpha_u8(legacy_alpha))
        if actual_alpha is not None:
            self._imwrite(stem.with_name(stem.name + "_alpha_actual.png"), self._to_alpha_u8(actual_alpha))
        self._imwrite(stem.with_name(stem.name + "_final_roi.png"), self._to_bgr_u8(final_roi))

    @staticmethod
    def _imwrite(path: Path, img: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), img):
            raise RuntimeError(f"Failed to write debug image: {path}")

    @staticmethod
    def _to_cpu_numpy(x: torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @classmethod
    def _to_bgr_u8(cls, x: torch.Tensor) -> np.ndarray:
        arr = cls._to_cpu_numpy(x)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HWC3 image, got shape={arr.shape}")
        if np.issubdtype(arr.dtype, np.floating):
            vmax = float(np.nanmax(arr)) if arr.size else 0.0
            if vmax <= 1.5:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
            arr = np.rint(arr).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @classmethod
    def _to_gray_u8(cls, x: torch.Tensor) -> np.ndarray:
        arr = cls._to_cpu_numpy(x)
        arr = np.squeeze(arr)
        if np.issubdtype(arr.dtype, np.floating):
            vmax = float(np.nanmax(arr)) if arr.size else 0.0
            if vmax <= 1.5:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
            arr = np.rint(arr).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @classmethod
    def _to_alpha_u8(cls, x: torch.Tensor) -> np.ndarray:
        arr = cls._to_cpu_numpy(x)
        arr = np.squeeze(arr)
        arr = np.clip(arr, 0.0, 1.0)
        return np.rint(arr * 255.0).astype(np.uint8)
