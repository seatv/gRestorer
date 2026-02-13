# SPDX-FileCopyrightText: gRestorer Authors
# SPDX-License-Identifier: AGPL-3.0

"""Scene / Clip primitives matching LADA 0.9.x semantics (inference pipeline).

This module is a focused, GPU-friendly port of the behavior described in the
"LadaAnalysis - Restoration Analysis.md" spec and the corresponding LADA 0.9.x
helpers.

Coordinate convention:
  - Boxes are tuples (t, l, b, r) == (y1, x1, y2, x2), inclusive.

Key behaviors mirrored from LADA:
  - Scene.belongs() uses a strict overlap check (NOT IoU).
  - Multiple detections belonging to the same scene *in the same frame* are
    UNIONed (box union; mask union if provided).
  - Clip creation normalizes crops to a fixed clip_size via per-clip scaling
    based on the maximum crop width/height across the scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math

import torch
import torch.nn.functional as F

Box = Tuple[int, int, int, int]  # (t, l, b, r) inclusive
Pad = Tuple[int, int, int, int]  # (pad_top, pad_bottom, pad_left, pad_right)


# -------------------------
# Box helpers (tlbr)
# -------------------------


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def xyxy_to_tlbr(xyxy: Tuple[float, float, float, float], h: int, w: int) -> Box:
    """Convert (x1,y1,x2,y2) -> (t,l,b,r) with clamping."""
    x1, y1, x2, y2 = xyxy
    l = _clamp_int(x1, 0, w - 1)
    t = _clamp_int(y1, 0, h - 1)
    r = _clamp_int(x2, 0, w - 1)
    b = _clamp_int(y2, 0, h - 1)
    if r < l:
        l, r = r, l
    if b < t:
        t, b = b, t
    return (t, l, b, r)


def tlbr_to_xyxy(b: Box) -> Tuple[int, int, int, int]:
    t, l, b_, r = b
    return (l, t, r, b_)


def _union_box(a: Box, b: Box) -> Box:
    at, al, ab, ar = a
    bt, bl, bb, br = b
    return (min(at, bt), min(al, bl), max(ab, bb), max(ar, br))


def _box_size(b: Box) -> Tuple[int, int]:
    t, l, b_, r = b
    return (b_ - t + 1, r - l + 1)


def _box_overlap(a: Box, b: Box) -> bool:
    """LADA-style overlap predicate used for Scene.belongs()."""
    at, al, ab, ar = a
    bt, bl, bb, br = b
    # Strict overlap (touching edges is NOT overlap).
    if ar <= bl or br <= al:
        return False
    if ab <= bt or bb <= at:
        return False
    return True


# -------------------------
# LADA crop_to_box_v3 (ported)
# -------------------------


def crop_box_to_target_v3(
    box: Box,
    img_h: int,
    img_w: int,
    target_hw: Tuple[int, int],
    *,
    max_box_expansion_factor: float = 1.0,
    border_size: float = 0.0,
) -> Tuple[Box, float]:
    """Exact port of LADA `crop_to_box_v3` box-expansion math.

    This returns the *cropped box coordinates* (t,l,b,r) within the original
    image and the computed scale_factor. It does NOT resize.

    Notes:
      - target_hw is (H,W) here, while LADA passes (W,H). For square clip_size
        (the default), this doesn't matter. We still implement the math
        faithfully by mapping appropriately.
    """
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    # LADA treats target_size as (target_width, target_height)
    target_width, target_height = target_w, target_h

    t, l, b, r = box
    width = int(r - l + 1)
    height = int(b - t + 1)

    # border expansion
    if border_size and border_size > 0.0:
        border_px = max(20, int(max(width, height) * float(border_size)))
    else:
        border_px = 0

    t = max(0, t - border_px)
    l = max(0, l - border_px)
    b = min(img_h - 1, b + border_px)
    r = min(img_w - 1, r + border_px)

    width = int(r - l + 1)
    height = int(b - t + 1)

    down_scale_factor = min(target_width / width, target_height / height)
    if down_scale_factor > 1.0:
        down_scale_factor = 1.0

    missing_width = int((target_width - (width * down_scale_factor)) / down_scale_factor)
    missing_height = int((target_height - (height * down_scale_factor)) / down_scale_factor)

    available_width_l = l
    available_width_r = (img_w - 1) - r
    available_height_t = t
    available_height_b = (img_h - 1) - b

    budget_width = int(max_box_expansion_factor * width)
    budget_height = int(max_box_expansion_factor * height)

    expand_width_lr = min(available_width_l, available_width_r, missing_width // 2, budget_width)
    expand_width_l = min(available_width_l - expand_width_lr, missing_width - expand_width_lr * 2, budget_width - expand_width_lr)
    expand_width_r = min(
        available_width_r - expand_width_lr,
        missing_width - expand_width_lr * 2 - expand_width_l,
        budget_width - expand_width_lr - expand_width_l,
    )

    expand_height_tb = min(available_height_t, available_height_b, missing_height // 2, budget_height)
    expand_height_t = min(available_height_t - expand_height_tb, missing_height - expand_height_tb * 2, budget_height - expand_height_tb)
    expand_height_b = min(
        available_height_b - expand_height_tb,
        missing_height - expand_height_tb * 2 - expand_height_t,
        budget_height - expand_height_tb - expand_height_t,
    )

    l2 = l - math.floor(expand_width_lr / 2) - expand_width_l
    r2 = r + math.ceil(expand_width_lr / 2) + expand_width_r
    t2 = t - math.floor(expand_height_tb / 2) - expand_height_t
    b2 = b + math.ceil(expand_height_tb / 2) + expand_height_b

    width2 = int(r2 - l2 + 1)
    height2 = int(b2 - t2 + 1)

    if down_scale_factor <= 1.0:
        scale_factor = float(down_scale_factor)
    else:
        # kept for parity with LADA, though down_scale_factor is clamped to <=1
        scale_factor = float(min(target_width / width2, target_height / height2))

    return (int(t2), int(l2), int(b2), int(r2)), scale_factor


# -------------------------
# Padding / resize (torch, HWC)
# -------------------------


def _torch_pad_reflect(x: torch.Tensor, pad: Pad) -> torch.Tensor:
    """Reflect pad that supports large pads by chunking."""
    pt, pb, pl, pr = pad
    if pt < 0 or pb < 0 or pl < 0 or pr < 0:
        raise ValueError(f"Negative pad: {pad}")

    # Work in NCHW for F.pad.
    x_nchw = x.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]

    def pad_once(y: torch.Tensor, p: Tuple[int, int, int, int]) -> torch.Tensor:
        return F.pad(y, p, mode="reflect")

    # Height
    while pt > 0 or pb > 0:
        _, _, hh, _ = x_nchw.shape
        max_step = max(1, hh - 1)
        step_t = min(pt, max_step)
        step_b = min(pb, max_step)
        x_nchw = pad_once(x_nchw, (0, 0, step_t, step_b))
        pt -= step_t
        pb -= step_b

    # Width
    while pl > 0 or pr > 0:
        _, _, _, ww = x_nchw.shape
        max_step = max(1, ww - 1)
        step_l = min(pl, max_step)
        step_r = min(pr, max_step)
        x_nchw = pad_once(x_nchw, (step_l, step_r, 0, 0))
        pl -= step_l
        pr -= step_r

    return x_nchw.squeeze(0).permute(1, 2, 0).contiguous()


def pad_image_hwc(
    x: torch.Tensor,
    target_hw: Tuple[int, int],
    *,
    pad_mode: str = "reflect",
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, Pad]:
    """Pad HWC tensor to target (H,W). Returns padded tensor and pad tuple."""
    th, tw = target_hw
    h, w = int(x.shape[0]), int(x.shape[1])
    if h > th or w > tw:
        raise ValueError(f"Cannot pad from {(h, w)} to {(th, tw)}; resize first")

    dh = th - h
    dw = tw - w
    pt = dh // 2
    pb = dh - pt
    pl = dw // 2
    pr = dw - pl
    pad = (pt, pb, pl, pr)

    if dh == 0 and dw == 0:
        return x, pad

    if pad_mode == "reflect":
        return _torch_pad_reflect(x, pad), pad

    if pad_mode in ("zero", "constant"):
        x_nchw = x.permute(2, 0, 1).unsqueeze(0)
        y = F.pad(x_nchw, (pl, pr, pt, pb), mode="constant", value=float(pad_value))
        y = y.squeeze(0).permute(1, 2, 0).contiguous()
        return y, pad

    raise ValueError(f"Unsupported pad_mode: {pad_mode}")


def resize_hwc(x: torch.Tensor, out_hw: Tuple[int, int], *, mode: str) -> torch.Tensor:
    """Resize HWC tensor via interpolate. mode: 'bilinear' or 'nearest'."""
    oh, ow = out_hw
    oh = max(1, int(oh))
    ow = max(1, int(ow))
    x_nchw = x.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
    if mode == "bilinear":
        y = F.interpolate(x_nchw, size=(oh, ow), mode="bilinear", align_corners=False)
    elif mode == "nearest":
        y = F.interpolate(x_nchw, size=(oh, ow), mode="nearest")
    else:
        raise ValueError(f"Unsupported resize mode: {mode}")
    return y.squeeze(0).permute(1, 2, 0).contiguous()


def resize_hw_mask(m: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize HW mask (uint8/float) with nearest."""
    oh, ow = out_hw
    oh = max(1, int(oh))
    ow = max(1, int(ow))
    if m.ndim != 2:
        raise ValueError(f"Expected HW mask, got {tuple(m.shape)}")
    m_f = m.to(torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    y = F.interpolate(m_f, size=(oh, ow), mode="nearest")
    y = y.squeeze(0).squeeze(0)
    if m.dtype == torch.uint8:
        y = y.clamp(0, 255).to(torch.uint8)
    return y.contiguous()


def pad_mask_hw(m: torch.Tensor, target_hw: Tuple[int, int]) -> Tuple[torch.Tensor, Pad]:
    """Pad HW mask to target with zeros."""
    th, tw = target_hw
    h, w = int(m.shape[0]), int(m.shape[1])
    dh = th - h
    dw = tw - w
    pt = dh // 2
    pb = dh - pt
    pl = dw // 2
    pr = dw - pl
    pad = (pt, pb, pl, pr)
    if dh == 0 and dw == 0:
        return m, pad
    m_nchw = m.unsqueeze(0).unsqueeze(0)
    y = F.pad(m_nchw, (pl, pr, pt, pb), mode="constant", value=0)
    y = y.squeeze(0).squeeze(0).contiguous()
    return y, pad


# -------------------------
# Scene + Clip
# -------------------------


@dataclass
class Scene:
    """Tracks one mosaic region across consecutive frames."""

    id: int
    frame_start: int
    frame_nums: List[int]
    roi_boxes: List[Box]
    crop_boxes: List[Box]
    crops: List[torch.Tensor]
    masks: List[Optional[torch.Tensor]]

    def __init__(self, *, id: int, start_frame: int) -> None:
        self.id = int(id)
        self.frame_start = int(start_frame)
        self.frame_nums = []
        self.roi_boxes = []
        self.crop_boxes = []
        self.crops = []
        self.masks = []

        self.ttl_after_end = 3      # allow this scene to persist for N frames
        self.end_frame = None       # last frame before it was considered done
        self.completed_reason = ""  # reason string for debug/logging


    def mark_completed(self, frame_num: int, reason: str):
        """Mark this scene as completed but keep it alive for a few frames."""
        self.end_frame = frame_num
        self.completed_reason = reason
        self.completed = True

    def is_expired(self, current_frame: int) -> bool:
        """Return True only when the linger period has passed."""
        if not self.completed or self.end_frame is None:
            return False
        return (current_frame - self.end_frame) > self.ttl_after_end


    @property
    def frame_end(self) -> int:
        return self.frame_nums[-1] if self.frame_nums else (self.frame_start - 1)

    def __len__(self) -> int:
        return len(self.frame_nums)

    def belongs(self, roi_box: Box) -> bool:
        if not self.roi_boxes:
            return False
        return _box_overlap(self.roi_boxes[-1], roi_box)

    def add_frame(
        self,
        *,
        frame_num: int,
        roi_box: Box,
        crop_box: Box,
        crop_img: torch.Tensor,
        crop_mask: Optional[torch.Tensor],
    ) -> None:
        if self.frame_nums and frame_num != self.frame_nums[-1] + 1:
            raise AssertionError(
                f"Scene frames must be consecutive: last={self.frame_nums[-1]} new={frame_num}"
            )
        self.frame_nums.append(int(frame_num))
        self.roi_boxes.append(roi_box)
        self.crop_boxes.append(crop_box)
        self.crops.append(crop_img)
        self.masks.append(crop_mask)

    # --- Compatibility aliases (older pipeline revisions) ---
    def append(
        self,
        frame_num: int,
        roi_box: Box,
        crop_box: Box,
        crop_img: torch.Tensor,
        crop_mask: Optional[torch.Tensor],
    ) -> None:
        """Alias for older callers that used Scene.append(...)."""
        self.add_frame(
            frame_num=frame_num,
            roi_box=roi_box,
            crop_box=crop_box,
            crop_img=crop_img,
            crop_mask=crop_mask,
        )

    def to_clip(self, *, clip_id: int, clip_size: int, pad_mode: str = "reflect") -> "Clip":
        """Build a Clip from this Scene.

        Kept as a Scene method because older pipeline revisions (and our
        SceneTracker) call s.to_clip(...).
        """
        return Clip(scene=self, clip_id=int(clip_id), clip_size=int(clip_size), pad_mode=str(pad_mode))

    def merge_same_frame(
        self,
        *,
        roi_box: Box,
        crop_box: Box,
        crop_img: torch.Tensor,
        crop_mask: Optional[torch.Tensor],
    ) -> None:
        if not self.frame_nums:
            raise AssertionError("Cannot merge into empty scene")
        self.roi_boxes[-1] = _union_box(self.roi_boxes[-1], roi_box)
        self.crop_boxes[-1] = crop_box
        self.crops[-1] = crop_img
        # NOTE (important): during same-frame merges, the *crop box usually changes*
        # (we recompute it from the union ROI). That means any previously stored
        # mask/crop for this frame may have a different spatial shape.
        #
        # LADA semantics are "union in full-frame coords, then re-crop".
        # Our tracker already re-crops from the UNION ROI and passes a mask in the
        # union-crop coordinate system.
        #
        # Therefore: if shapes mismatch, we must NOT try to max() them; we replace.
        if crop_mask is None:
            self.masks[-1] = None
        else:
            prev = self.masks[-1]
            if prev is None:
                self.masks[-1] = crop_mask
            else:
                if prev.shape == crop_mask.shape:
                    self.masks[-1] = torch.maximum(prev, crop_mask)
                else:
                    # Replace with the union-crop mask.
                    self.masks[-1] = crop_mask

    def max_crop_hw(self) -> Tuple[int, int]:
        max_h = 0
        max_w = 0
        for b in self.crop_boxes:
            h, w = _box_size(b)
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        return max_h, max_w


@dataclass
class Clip:
    """A normalized clip built from a Scene."""

    id: int
    frame_start: int
    frame_end: int
    frames: List[torch.Tensor]
    masks: List[torch.Tensor]
    boxes: List[Box]
    crop_shapes: List[Tuple[int, int]]
    pad_after_resizes: List[Pad]
    frame_nums: List[int]
    clip_size: int
    pad_mode: str

    def __init__(self, *, scene: Scene, clip_id: int, clip_size: int, pad_mode: str = "reflect") -> None:
        if len(scene) == 0:
            raise ValueError("Cannot build clip from empty scene")
        self.id = int(clip_id)
        self.frame_start = int(scene.frame_nums[0])
        self.frame_end = int(scene.frame_nums[-1])
        self.frame_nums = list(scene.frame_nums)
        self.clip_size = int(clip_size)
        self.pad_mode = str(pad_mode)
        self.boxes = list(scene.crop_boxes)
        self.crop_shapes = [(int(x.shape[0]), int(x.shape[1])) for x in scene.crops]

        max_h, max_w = scene.max_crop_hw()
        if max_h <= 0 or max_w <= 0:
            raise ValueError("Invalid max crop size")

        scale_h = self.clip_size / float(max_h)
        scale_w = self.clip_size / float(max_w)

        self.frames = []
        self.masks = []
        self.pad_after_resizes = []

        for i, crop_u8 in enumerate(scene.crops):
            ch, cw = self.crop_shapes[i]
            out_h = max(1, int(ch * scale_h))
            out_w = max(1, int(cw * scale_w))

            img_f = crop_u8.to(torch.float32) / 255.0
            img_rs = resize_hwc(img_f, (out_h, out_w), mode="bilinear")
            img_pd, pad = pad_image_hwc(
                img_rs, (self.clip_size, self.clip_size), pad_mode=self.pad_mode, pad_value=0.0
            )

            m = scene.masks[i]
            if m is None:
                m = torch.ones((ch, cw), dtype=torch.uint8, device=crop_u8.device) * 255
            m_rs = resize_hw_mask(m, (out_h, out_w))
            m_pd, _ = pad_mask_hw(m_rs, (self.clip_size, self.clip_size))

            self.frames.append(img_pd)
            self.masks.append(m_pd)
            self.pad_after_resizes.append(pad)

    def __len__(self) -> int:
        return len(self.frames)

    # --- Compatibility aliases ---
    @property
    def crop_boxes(self) -> List[Box]:
        return self.boxes

    def pop(self) -> Tuple[torch.Tensor, torch.Tensor, Box, Tuple[int, int], Pad]:
        """Pop the earliest clip element.

        LADA-faithful behavior: advance frame_start as frames are popped so a clip can be
        applied by matching `clip.frame_start == frame_num` and popping once per frame.
        """
        frame = self.frames.pop(0)
        mask = self.masks.pop(0)
        box = self.boxes.pop(0)
        crop_shape = self.crop_shapes.pop(0)
        pad = self.pad_after_resizes.pop(0)

        # Keep frame_nums/frame_start in sync with pops (LADA-style).
        if self.frame_nums:
            self.frame_nums.pop(0)

        if self.frame_nums:
            self.frame_start = int(self.frame_nums[0])
            self.frame_end = int(self.frame_nums[-1])
        else:
            # Mark empty: keep invariant frame_start > frame_end
            old_end = int(self.frame_end)
            self.frame_start = old_end + 1
            self.frame_end = old_end

        return frame, mask, box, crop_shape, pad


__all__ = [
    "Box",
    "Pad",
    "Scene",
    "Clip",
    "xyxy_to_tlbr",
    "tlbr_to_xyxy",
    "crop_box_to_target_v3",
]
