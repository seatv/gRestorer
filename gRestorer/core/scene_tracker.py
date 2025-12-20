# SPDX-FileCopyrightText: gRestorer Authors
# SPDX-License-Identifier: AGPL-3.0

"""SceneTracker + Clip creation, aligned with LADA 0.9.x semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .scene import Box, Clip, Scene, crop_box_to_target_v3, _union_box


@dataclass(frozen=True)
class TrackerConfig:
    clip_size: int = 256
    max_clip_length: int = 30
    pad_mode: str = "reflect"
    border_size: float = 0.06
    max_box_expansion_factor: float = 1.0


class SceneTracker:
    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self._scenes: List[Scene] = []
        self._scene_counter = 0
        self._clip_counter = 0

    @property
    def active_scenes(self) -> int:
        return len(self._scenes)

    @property
    def clip_counter(self) -> int:
        return self._clip_counter

    def _new_scene(self, frame_num: int) -> Scene:
        s = Scene(id=self._scene_counter, start_frame=frame_num)
        self._scene_counter += 1
        return s

    def _compute_crop(
        self,
        frame_bgr_u8: torch.Tensor,
        roi_box: Box,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Box, torch.Tensor, Optional[torch.Tensor]]:
        h, w = int(frame_bgr_u8.shape[0]), int(frame_bgr_u8.shape[1])
        crop_box, _scale = crop_box_to_target_v3(
            roi_box,
            img_h=h,
            img_w=w,
            target_hw=(self.cfg.clip_size, self.cfg.clip_size),
            max_box_expansion_factor=self.cfg.max_box_expansion_factor,
            border_size=self.cfg.border_size,
        )
        t, l, b, r = crop_box
        crop_img = frame_bgr_u8[t : b + 1, l : r + 1, :].clone()

        crop_mask_out: Optional[torch.Tensor] = None
        if roi_mask is not None:
            crop_mask_out = roi_mask[t : b + 1, l : r + 1].clone()

        return crop_box, crop_img, crop_mask_out

    def ingest_frame(
        self,
        frame_num: int,
        frame_bgr_u8: torch.Tensor,
        roi_boxes: Sequence[Box],
        roi_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> List[Box]:
        """Update scenes with detections for ONE frame. Returns tracked ROI boxes for overlay."""
        if roi_masks is not None and len(roi_masks) != len(roi_boxes):
            raise ValueError("roi_masks length must match roi_boxes length")

        for i, box in enumerate(roi_boxes):
            mask = roi_masks[i] if roi_masks is not None else None

            matched = None
            for s in self._scenes:
                if s.belongs(box):
                    matched = s
                    break

            if matched is None:
                matched = self._new_scene(frame_num)
                self._scenes.append(matched)

            # Same-frame merge: union ROI, recompute crop from union ROI.
            if matched.frame_end == frame_num:
                union_roi = _union_box(matched.roi_boxes[-1], box)
                crop_box, crop_img, crop_mask = self._compute_crop(frame_bgr_u8, union_roi, None)
                matched.merge_same_frame(
                    roi_box=union_roi,
                    crop_box=crop_box,
                    crop_img=crop_img,
                    crop_mask=crop_mask,
                )
            else:
                crop_box, crop_img, crop_mask = self._compute_crop(frame_bgr_u8, box, None)
                matched.add_frame(
                    frame_num=frame_num,
                    roi_box=box,
                    crop_box=crop_box,
                    crop_img=crop_img,
                    crop_mask=crop_mask,
                )

        overlay_boxes = [s.roi_boxes[-1] for s in self._scenes if s.frame_end == frame_num]
        return overlay_boxes

    def flush_completed(self, frame_num: int, *, eof: bool = False) -> List[Clip]:
        """Flush completed scenes and return created clips (sorted by frame_start)."""
        completed: List[Scene] = []

        for s in self._scenes:
            if s in completed:
                continue
            if eof or (s.frame_end < frame_num) or (len(s) >= self.cfg.max_clip_length):
                completed.append(s)
                # LADA ordering rule: also flush scenes that started earlier.
                for other in self._scenes:
                    if other in completed:
                        continue
                    if other.frame_start < s.frame_start:
                        completed.append(other)

        if not completed:
            return []

        completed.sort(key=lambda sc: sc.frame_start)

        clips: List[Clip] = []
        for s in completed:
            clip = Clip(
                scene=s,
                clip_id=self._clip_counter,
                clip_size=self.cfg.clip_size,
                pad_mode=self.cfg.pad_mode,
            )
            self._clip_counter += 1
            clips.append(clip)

        for s in completed:
            if s in self._scenes:
                self._scenes.remove(s)

        return clips

    def flush_eof(self, next_frame_num: int) -> List[Clip]:
        return self.flush_completed(next_frame_num, eof=True)
