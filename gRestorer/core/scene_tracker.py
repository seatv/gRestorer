from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict

import torch

from gRestorer.core.scene import Box, Clip, Scene, crop_box_to_target_v3


@dataclass
class TrackerConfig:
    clip_size: int = 256
    max_clip_length: int = 30
    pad_mode: str = "reflect"
    border_size: float = 0.06
    max_box_expansion_factor: float = 1.0
    debug: bool = False

    # If True and the detector provides per-pixel masks, we'll use them for
    # clip masks (more LADA-faithful compositing). If False, clip masks will
    # be simple rectangle-box masks.
    use_seg_masks: bool = True

    # NOTE: LADA unions multiple detections for the same scene in the same frame.
    # This can occasionally create a single larger ROI if two boxes briefly overlap.
    # TODO: consider adding a *debug-only* metric for "weak overlap" merges so we can
    #       inspect detector jitter. We should not change LADA semantics by default.


@dataclass
class TrackerStepResult:
    overlay_boxes: List[Box]
    new_clips: List[Clip]
    active_scenes: int
    t_track: float
    t_clip_build: float


def _union_box(a: Box, b: Box) -> Box:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


class SceneTracker:
    """Track per-frame detections into LADA-style Scenes, then emit Clips."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self._scenes: List[Scene] = []
        self._scene_counter: int = 0
        self._clip_counter: int = 0

    def reset(self) -> None:
        self._scenes.clear()
        self._scene_counter = 0
        self._clip_counter = 0

    @property
    def scenes_active(self) -> int:
        return len(self._scenes)

    # Back-compat alias used by the CLI pipeline.
    @property
    def active_scenes(self) -> int:
        return self.scenes_active

    def min_active_start(self) -> Optional[int]:
        """Earliest start-frame among active scenes (None if no active scenes)."""
        if not self._scenes:
            return None
        return min(s.frame_start for s in self._scenes)

    def clips_emitted(self) -> int:
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
    ) -> Tuple[Box, torch.Tensor, torch.Tensor]:
        """Compute LADA crop_to_box_v3 crop box and slice crop from the frame.

        We *always* create a per-crop mask on the frame device.

        - Base mask is a rectangle mask derived from roi_box.
        - If roi_mask is provided (per-pixel segmentation) and use_seg_masks=True,
          we OR it into the crop mask. If roi_mask is on CPU at full resolution,
          we slice *only the crop region* on CPU and transfer just that to the
          frame device (avoids copying full HxW masks each frame).
        """
        h, w = int(frame_bgr_u8.shape[0]), int(frame_bgr_u8.shape[1])
        crop_box, _scale = crop_box_to_target_v3(
            roi_box,
            img_h=h,
            img_w=w,
            target_hw=(self.cfg.clip_size, self.cfg.clip_size),
            max_box_expansion_factor=self.cfg.max_box_expansion_factor,
            border_size=float(self.cfg.border_size),
        )
        t, l, b, r = crop_box
        crop_img = frame_bgr_u8[t: b + 1, l: r + 1, :].clone()

        # Fast box mask in crop coordinates
        crop_h = int(b - t + 1)
        crop_w = int(r - l + 1)
        crop_mask_out = torch.zeros((crop_h, crop_w), device=frame_bgr_u8.device, dtype=torch.uint8)

        rt, rl, rb, rr = roi_box
        it = max(t, rt)
        il = max(l, rl)
        ib = min(b, rb)
        ir = min(r, rr)
        if ib >= it and ir >= il:
            crop_mask_out[it - t: ib - t + 1, il - l: ir - l + 1] = 255

        # Optional segmentation mask (LADA uses masks for blend boundaries).
        # We keep the actual transfer minimal: crop on CPU first, then move the crop.
        if roi_mask is not None and self.cfg.use_seg_masks:
            try:
                if roi_mask.device == frame_bgr_u8.device:
                    if roi_mask.shape == (h, w):
                        m = roi_mask[t: b + 1, l: r + 1]
                        crop_mask_out = torch.maximum(crop_mask_out, m.to(dtype=torch.uint8))
                    elif roi_mask.shape == (crop_h, crop_w):
                        crop_mask_out = torch.maximum(crop_mask_out, roi_mask.to(dtype=torch.uint8))

                elif roi_mask.device.type == "cpu":
                    # Full-res CPU mask (H,W) or already-cropped CPU mask (crop_h,crop_w).
                    if roi_mask.shape == (h, w):
                        m_cpu = roi_mask[t: b + 1, l: r + 1]
                    elif roi_mask.shape == (crop_h, crop_w):
                        m_cpu = roi_mask
                    else:
                        m_cpu = None

                    if m_cpu is not None:
                        m = m_cpu.to(device=frame_bgr_u8.device, dtype=torch.uint8)
                        crop_mask_out = torch.maximum(crop_mask_out, m)

                else:
                    # Unexpected device (e.g. different GPU). Ignore.
                    pass

            except Exception:
                # Mask usage is best-effort; don't crash the pipeline.
                pass

        return crop_box, crop_img, crop_mask_out

    def step_frame(
            self,
            frame_num: int,
            frame_bgr_u8: torch.Tensor,
            roi_boxes: Sequence[Box],
            roi_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> TrackerStepResult:
        """Ingest one frame's detections, update scenes, and flush completed scenes."""
        if roi_masks is not None and len(roi_masks) != len(roi_boxes):
            raise ValueError("roi_masks length must match roi_boxes length")

        t0 = time.perf_counter()

        # Update scenes with detections.
        for i, box in enumerate(roi_boxes):
            mask = None
            if self.cfg.use_seg_masks and roi_masks is not None:
                mask = roi_masks[i]

            matched: Optional[Scene] = None
            for s in self._scenes:
                if s.belongs(box):
                    matched = s
                    break

            if matched is None:
                matched = self._new_scene(frame_num)
                self._scenes.append(matched)

            if matched.frame_end == frame_num:
                # Same-frame merge: union ROI and recompute crop from union.
                union_roi = _union_box(matched.roi_boxes[-1], box)
                crop_box, crop_img, crop_mask = self._compute_crop(frame_bgr_u8, union_roi, None)
                matched.merge_same_frame(
                    roi_box=union_roi,
                    crop_box=crop_box,
                    crop_img=crop_img,
                    crop_mask=crop_mask,
                )
            else:
                crop_box, crop_img, crop_mask = self._compute_crop(frame_bgr_u8, box, mask)
                matched.add_frame(
                    frame_num=frame_num,
                    roi_box=box,
                    crop_box=crop_box,
                    crop_img=crop_img,
                    crop_mask=crop_mask,
                )

        # Any scenes not updated in this frame are completed (gap).
        completed_gap: List[Scene] = [s for s in self._scenes if s.frame_end < frame_num]

        # And any scenes that reached max length are completed.
        completed_maxlen: List[Scene] = [s for s in self._scenes if len(s) >= self.cfg.max_clip_length]

        # LADA-faithful rule: when a scene completes, also complete any scenes that started earlier.
        # This guarantees deterministic clip ordering and prevents early-started scenes from
        # blocking drain forever when later scenes end first.
        completed_scenes: List[Scene] = []
        reason_by_scene_id: Dict[int, str] = {}
        for s in completed_gap:
            reason_by_scene_id[s.id] = "gap"
        for s in completed_maxlen:
            reason_by_scene_id.setdefault(s.id, "max_len")

        for current_scene in list(self._scenes):
            is_done = (current_scene.frame_end < frame_num) or (len(current_scene) >= self.cfg.max_clip_length)
            if not is_done:
                continue

            if current_scene not in completed_scenes:
                completed_scenes.append(current_scene)

            for other_scene in self._scenes:
                if other_scene is current_scene:
                    continue
                if other_scene.frame_start < current_scene.frame_start and other_scene not in completed_scenes:
                    completed_scenes.append(other_scene)
                    reason_by_scene_id.setdefault(other_scene.id, "cascade")

        # LADA: complete in ascending start-frame order.
        completed_unique: List[Scene] = sorted(completed_scenes, key=lambda s: s.frame_start)

        # Remove completed from active list.
        if completed_unique:
            completed_ids = {s.id for s in completed_unique}
            self._scenes = [s for s in self._scenes if s.id not in completed_ids]

        t1 = time.perf_counter()

        # reason_by_scene_id is built alongside the LADA-style completion selection above.

        new_clips: List[Clip] = []
        t_clip_build = 0.0
        for s in completed_unique:
            tb0 = time.perf_counter()
            new_clips.append(
                s.to_clip(
                    clip_id=self._clip_counter,
                    clip_size=self.cfg.clip_size,
                    pad_mode=self.cfg.pad_mode,
                )
            )

            if self.cfg.debug:
                why = reason_by_scene_id.get(s.id, "?")
                roi_xyxy = s.roi_boxes[-1] if s.roi_boxes else (0, 0, 0, 0)
                print(
                    f"[Clip] clip_id={self._clip_counter:5d} scene_id={s.id:4d} why={why:10s} "
                    f"frames={s.frame_start:5d}-{s.frame_end:5d} len={len(s):3d} roi_xyxy={roi_xyxy}"
                )
            self._clip_counter += 1
            tb1 = time.perf_counter()
            t_clip_build += (tb1 - tb0)

        t2 = time.perf_counter()

        overlay_boxes: List[Box] = []
        # For debug/visualization we want the boxes corresponding to *this frame's* overlays.
        # Crucially, this must include scenes that *complete on this frame* (e.g. max_len),
        # otherwise you see missing overlays at f=29,59,... even though detections exist.
        #
        # NOTE: At this point, any scenes that completed have already been removed from
        # self._scenes, so we must also include completed_unique.
        for s in self._scenes:
            if s.frame_end == frame_num and s.roi_boxes:
                overlay_boxes.append(s.roi_boxes[-1])
        for s in completed_unique:
            if s.frame_end == frame_num and s.roi_boxes:
                overlay_boxes.append(s.roi_boxes[-1])

        return TrackerStepResult(
            overlay_boxes=overlay_boxes,
            new_clips=new_clips,
            active_scenes=len(self._scenes),
            t_track=(t1 - t0),
            t_clip_build=t_clip_build,
        )

    def flush_eof(self, *_: object) -> List[Clip]:
        """Flush all remaining scenes at end-of-file."""
        clips: List[Clip] = []
        for s in self._scenes:
            clips.append(
                s.to_clip(
                    clip_id=self._clip_counter,
                    clip_size=self.cfg.clip_size,
                    pad_mode=self.cfg.pad_mode,
                )
            )
            self._clip_counter += 1

        self._scenes.clear()
        return clips

    # --- Compatibility helpers (older pipeline revisions) ---
    def ingest_frame(self, frame_num: int, frame_bgr_u8: torch.Tensor, roi_boxes: Sequence[Box]) -> List[Box]:
        """Compat: older pipelines expect an ingest_frame() that returns overlay boxes."""
        res = self.step_frame(frame_num, frame_bgr_u8, roi_boxes)
        return res.overlay_boxes

    def flush_completed(self, current_frame: int) -> List[Clip]:
        """Compat: older pipelines called flush_completed() explicitly.

        In the current implementation, completion is handled inside step_frame().
        This is therefore a no-op and returns an empty list.
        """
        return []


__all__ = [
    "TrackerConfig",
    "TrackerStepResult",
    "SceneTracker",
]
