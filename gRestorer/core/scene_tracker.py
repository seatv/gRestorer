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

    # --- Crop stabilization (addresses ROI wobble/breathing) ---
    # Quantize crop boxes to a pixel grid (e.g. 8px) to eliminate 0.5/1px jitter.
    # This is applied to the *crop box* (not the detector ROI), so it doesn't change
    # detection semantics; it only stabilizes the region we feed the restorer.
    crop_quant_px: int = 8

    # If True, keep the previous crop box when the new crop would only move a small amount
    # and the current ROI still fits inside the previous crop.
    crop_sticky: bool = True

    # Maximum allowed per-edge movement (pixels) for sticky to keep the previous crop.
    crop_sticky_pad_px: int = 8

    # Expand the previous ROI box by this pad when deciding scene membership. This reduces
    # scene fragmentation when a box 'jitters' by a few pixels and barely loses overlap.
    match_pad_px: int = 8

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


def _box_overlap_strict(a: Box, b: Box) -> bool:
    """Strict overlap: touching edges is NOT overlap (matches LADA semantics)."""
    at, al, ab, ar = a
    bt, bl, bb, br = b
    if ar <= bl or br <= al:
        return False
    if ab <= bt or bb <= at:
        return False
    return True


def _box_overlap_pad(a: Box, b: Box, pad: int) -> bool:
    """Strict overlap after expanding both boxes by `pad` pixels."""
    if pad <= 0:
        return _box_overlap_strict(a, b)
    at, al, ab, ar = a
    bt, bl, bb, br = b
    a2 = (at - pad, al - pad, ab + pad, ar + pad)
    b2 = (bt - pad, bl - pad, bb + pad, br + pad)
    return _box_overlap_strict(a2, b2)


def _roi_inside_crop(roi: Box, crop: Box) -> bool:
    rt, rl, rb, rr = roi
    ct, cl, cb, cr = crop
    return (rt >= ct) and (rl >= cl) and (rb <= cb) and (rr <= cr)


def _quantize_crop_box(crop: Box, img_h: int, img_w: int, q: int) -> Box:
    """Quantize crop edges to a q-pixel grid.
    - top/left are floored
    - bottom/right are ceiled
    This tends to slightly expand crops, trading a bit of compute for stability.
    """
    if q <= 1:
        t, l, b, r = crop
        return (max(0, t), max(0, l), min(img_h - 1, b), min(img_w - 1, r))

    t, l, b, r = crop

    t2 = (t // q) * q
    l2 = (l // q) * q

    # Inclusive bottom/right: quantize (b+1) and subtract 1.
    b2 = (((b + 1 + q - 1) // q) * q) - 1
    r2 = (((r + 1 + q - 1) // q) * q) - 1

    # Clamp within bounds
    t2 = max(0, t2)
    l2 = max(0, l2)
    b2 = min(img_h - 1, b2)
    r2 = min(img_w - 1, r2)

    # Ensure valid box
    if b2 < t2:
        b2 = min(img_h - 1, t2)
    if r2 < l2:
        r2 = min(img_w - 1, l2)

    return (int(t2), int(l2), int(b2), int(r2))


class SceneTracker:
    """Track per-frame detections into LADA-style Scenes, then emit Clips."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self._scenes: List[Scene] = []
        self._scene_counter: int = 0
        self._clip_counter: int = 0

    def _belongs_scene(self, s: Scene, roi_box: Box) -> bool:
        """Scene membership with optional padding to reduce jitter-induced fragmentation."""
        if not s.roi_boxes:
            return False
        pad = int(getattr(self.cfg, "match_pad_px", 0) or 0)
        if pad > 0:
            return _box_overlap_pad(s.roi_boxes[-1], roi_box, pad)
        return s.belongs(roi_box)

    def _stabilize_crop_box(
            self,
            *,
            roi_box: Box,
            base_crop_box: Box,
            prev_crop_box: Optional[Box],
            img_h: int,
            img_w: int,
    ) -> Box:
        """Apply quantization + sticky crop stabilization.

        Order:
          1) Quantize crop to a pixel grid (crop_quant_px).
          2) If sticky enabled and ROI still fits inside prev crop, keep prev crop when
             the new crop differs only slightly (crop_sticky_pad_px).
        """
        q = int(getattr(self.cfg, "crop_quant_px", 0) or 0)
        crop = _quantize_crop_box(base_crop_box, img_h=img_h, img_w=img_w, q=q) if q > 1 else base_crop_box

        if not bool(getattr(self.cfg, "crop_sticky", False)):
            return crop
        if prev_crop_box is None:
            return crop

        # Only keep the old crop if the current ROI is still fully inside it.
        if not _roi_inside_crop(roi_box, prev_crop_box):
            return crop

        pad = int(getattr(self.cfg, "crop_sticky_pad_px", 0) or 0)
        if pad <= 0:
            return crop

        dt = abs(int(crop[0]) - int(prev_crop_box[0]))
        dl = abs(int(crop[1]) - int(prev_crop_box[1]))
        db = abs(int(crop[2]) - int(prev_crop_box[2]))
        dr = abs(int(crop[3]) - int(prev_crop_box[3]))
        if max(dt, dl, db, dr) <= pad:
            return prev_crop_box

        return crop

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
            *,
            force_crop_box: Optional[Box] = None,
    ) -> Tuple[Box, torch.Tensor, torch.Tensor]:
        """Compute LADA crop_to_box_v3 crop box and slice crop from the frame.

        We create a per-crop mask on the frame device.

        - If roi_mask is provided (per-pixel segmentation) and use_seg_masks=True, we use it
          as the clip mask (cropped to crop_box). If roi_mask is on CPU at full resolution,
          we slice *only the crop region* on CPU and transfer just that to the frame device
          (avoids copying full HxW masks each frame).
        - Otherwise we fall back to a rectangle mask derived from roi_box.
        """
        h, w = int(frame_bgr_u8.shape[0]), int(frame_bgr_u8.shape[1])

        if force_crop_box is None:
            crop_box, _scale = crop_box_to_target_v3(
                roi_box,
                img_h=h,
                img_w=w,
                target_hw=(self.cfg.clip_size, self.cfg.clip_size),
                max_box_expansion_factor=self.cfg.max_box_expansion_factor,
                border_size=float(self.cfg.border_size),
            )
        else:
            crop_box = force_crop_box

        t, l, b, r = crop_box
        crop_img = frame_bgr_u8[t: b + 1, l: r + 1, :].clone()

        # Mask generation (seam-sensitive)
        crop_h = int(b - t + 1)
        crop_w = int(r - l + 1)

        # 1) Rectangle base mask (always)
        crop_mask_out = torch.zeros((crop_h, crop_w), device=frame_bgr_u8.device, dtype=torch.uint8)

        rt, rl, rb, rr = roi_box
        it = max(t, rt)
        il = max(l, rl)
        ib = min(b, rb)
        ir = min(r, rr)
        if ib >= it and ir >= il:
            crop_mask_out[it - t: ib - t + 1, il - l: ir - l + 1] = 255

        # 2) Optional seg mask: OR it in (never replace rectangle)
        if roi_mask is not None and self.cfg.use_seg_masks:
            try:
                def _mask_u8(m: torch.Tensor) -> torch.Tensor:
                    # Normalize to uint8 {0,255} without device-sync (no .item()).
                    if m.dtype == torch.bool:
                        return m.to(dtype=torch.uint8) * 255
                    if m.is_floating_point():
                        return torch.where(m > 0.5, 255, 0).to(dtype=torch.uint8)
                    return m.to(dtype=torch.uint8)

                seg_crop: Optional[torch.Tensor] = None
                if roi_mask.device == frame_bgr_u8.device:
                    if roi_mask.shape == (h, w):
                        seg_crop = roi_mask[t: b + 1, l: r + 1]
                    elif roi_mask.shape == (crop_h, crop_w):
                        seg_crop = roi_mask
                elif roi_mask.device.type == "cpu":
                    if roi_mask.shape == (h, w):
                        seg_crop = roi_mask[t: b + 1, l: r + 1].to(device=frame_bgr_u8.device)
                    elif roi_mask.shape == (crop_h, crop_w):
                        seg_crop = roi_mask.to(device=frame_bgr_u8.device)

                if seg_crop is not None:
                    crop_mask_out = torch.maximum(crop_mask_out, _mask_u8(seg_crop))

            except Exception:
                # Best-effort: don't crash the pipeline on mask oddities.
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
                if self._belongs_scene(s, box):
                    matched = s
                    break

            if matched is None:
                matched = self._new_scene(frame_num)
                self._scenes.append(matched)

            if matched.frame_end == frame_num:
                # Same-frame merge: union ROI and recompute crop from union.
                union_roi = _union_box(matched.roi_boxes[-1], box)

                h, w = int(frame_bgr_u8.shape[0]), int(frame_bgr_u8.shape[1])
                base_crop_box, _ = crop_box_to_target_v3(
                    union_roi,
                    img_h=h,
                    img_w=w,
                    target_hw=(self.cfg.clip_size, self.cfg.clip_size),
                    max_box_expansion_factor=self.cfg.max_box_expansion_factor,
                    border_size=float(self.cfg.border_size),
                )

                prev_crop_box = matched.crop_boxes[-1] if matched.crop_boxes else None
                stable_crop_box = self._stabilize_crop_box(
                    roi_box=union_roi,
                    base_crop_box=base_crop_box,
                    prev_crop_box=prev_crop_box,
                    img_h=h,
                    img_w=w,
                )

                crop_box, crop_img, cur_mask = self._compute_crop(
                    frame_bgr_u8,
                    union_roi,
                    mask,
                    force_crop_box=stable_crop_box,
                )

                # Re-crop from union ROI. IMPORTANT: pass current detection mask (if any),
                # then union it with the previous crop-mask by mapping old->new crop coords.
                prev_mask = matched.masks[-1] if matched.masks else None
                prev_crop_box = matched.crop_boxes[-1] if matched.crop_boxes else None

                if prev_mask is not None and prev_crop_box is not None:
                    nt, nl, nb, nr = crop_box
                    new_h = int(nb - nt + 1)
                    new_w = int(nr - nl + 1)
                    merged = torch.zeros((new_h, new_w), device=frame_bgr_u8.device, dtype=torch.uint8)

                    ot, ol, ob, or_ = prev_crop_box
                    it = max(nt, ot)
                    il = max(nl, ol)
                    ib = min(nb, ob)
                    ir = min(nr, or_)
                    if ib >= it and ir >= il:
                        h_int = int(ib - it + 1)
                        w_int = int(ir - il + 1)
                        oy0 = int(it - ot)
                        ox0 = int(il - ol)
                        ny0 = int(it - nt)
                        nx0 = int(il - nl)
                        merged[ny0: ny0 + h_int, nx0: nx0 + w_int] = torch.maximum(
                            merged[ny0: ny0 + h_int, nx0: nx0 + w_int],
                            prev_mask[oy0: oy0 + h_int, ox0: ox0 + w_int].to(dtype=torch.uint8),
                        )

                    crop_mask = torch.maximum(merged, cur_mask.to(dtype=torch.uint8))
                else:
                    crop_mask = cur_mask

                matched.merge_same_frame(
                    roi_box=union_roi,
                    crop_box=crop_box,
                    crop_img=crop_img,
                    crop_mask=crop_mask,
                )
            else:
                h, w = int(frame_bgr_u8.shape[0]), int(frame_bgr_u8.shape[1])
                base_crop_box, _ = crop_box_to_target_v3(
                    box,
                    img_h=h,
                    img_w=w,
                    target_hw=(self.cfg.clip_size, self.cfg.clip_size),
                    max_box_expansion_factor=self.cfg.max_box_expansion_factor,
                    border_size=float(self.cfg.border_size),
                )

                prev_crop_box = matched.crop_boxes[-1] if matched.crop_boxes else None
                stable_crop_box = self._stabilize_crop_box(
                    roi_box=box,
                    base_crop_box=base_crop_box,
                    prev_crop_box=prev_crop_box,
                    img_h=h,
                    img_w=w,
                )

                crop_box, crop_img, crop_mask = self._compute_crop(
                    frame_bgr_u8,
                    box,
                    mask,
                    force_crop_box=stable_crop_box,
                )
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

        overlay_boxes: List[Box] = []
        # Include scenes that complete on this frame (e.g. max_len), otherwise you see missing overlays.
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
