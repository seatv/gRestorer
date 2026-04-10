# SPDX-FileCopyrightText: gRestorer Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math

import torch
import torch.nn.functional as F

from gRestorer.detector.core import FaceMetadata

Box = Tuple[int, int, int, int]  # (t, l, b, r) inclusive
Pad = Tuple[int, int, int, int]  # (pad_top, pad_bottom, pad_left, pad_right)


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def xyxy_to_tlbr(xyxy: Tuple[float, float, float, float], h: int, w: int) -> Box:
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
    at, al, ab, ar = a
    bt, bl, bb, br = b
    if ar <= bl or br <= al:
        return False
    if ab <= bt or bb <= at:
        return False
    return True


def crop_box_to_target_v3(
    box: Box,
    img_h: int,
    img_w: int,
    target_hw: Tuple[int, int],
    *,
    max_box_expansion_factor: float = 1.0,
    border_size: float = 0.0,
) -> Tuple[Box, float]:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    target_width, target_height = target_w, target_h

    t, l, b, r = box
    width = int(r - l + 1)
    height = int(b - t + 1)

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
        scale_factor = float(min(target_width / width2, target_height / height2))

    return (int(t2), int(l2), int(b2), int(r2)), scale_factor


def _torch_pad_reflect(x: torch.Tensor, pad: Pad) -> torch.Tensor:
    pt, pb, pl, pr = pad
    if pt < 0 or pb < 0 or pl < 0 or pr < 0:
        raise ValueError(f"Negative pad: {pad}")

    x_nchw = x.permute(2, 0, 1).unsqueeze(0)

    def pad_once(y: torch.Tensor, p: Tuple[int, int, int, int]) -> torch.Tensor:
        return F.pad(y, p, mode="reflect")

    while pt > 0 or pb > 0:
        _, _, hh, _ = x_nchw.shape
        max_step = max(1, hh - 1)
        step_t = min(pt, max_step)
        step_b = min(pb, max_step)
        x_nchw = pad_once(x_nchw, (0, 0, step_t, step_b))
        pt -= step_t
        pb -= step_b

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
    oh, ow = out_hw
    oh = max(1, int(oh))
    ow = max(1, int(ow))
    x_nchw = x.permute(2, 0, 1).unsqueeze(0)
    if mode == "bilinear":
        y = F.interpolate(x_nchw, size=(oh, ow), mode="bilinear", align_corners=False)
    elif mode == "nearest":
        y = F.interpolate(x_nchw, size=(oh, ow), mode="nearest")
    else:
        raise ValueError(f"Unsupported resize mode: {mode}")
    return y.squeeze(0).permute(1, 2, 0).contiguous()


def resize_hw_mask(m: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    oh, ow = out_hw
    oh = max(1, int(oh))
    ow = max(1, int(ow))
    if m.ndim != 2:
        raise ValueError(f"Expected HW mask, got {tuple(m.shape)}")
    m_f = m.to(torch.float32).unsqueeze(0).unsqueeze(0)
    y = F.interpolate(m_f, size=(oh, ow), mode="nearest")
    y = y.squeeze(0).squeeze(0)
    if m.dtype == torch.uint8:
        y = y.clamp(0, 255).to(torch.uint8)
    return y.contiguous()


def pad_mask_hw(m: torch.Tensor, target_hw: Tuple[int, int]) -> Tuple[torch.Tensor, Pad]:
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


def face_meta_full_to_crop(face_meta: Optional[FaceMetadata], crop_box: Box) -> Optional[FaceMetadata]:
    if face_meta is None:
        return None
    t, l, _, _ = crop_box
    x1, y1, x2, y2 = face_meta.bbox_xyxy
    kps = face_meta.kps
    if kps is not None:
        kps = kps.clone().to(dtype=torch.float32)
        kps[:, 0] -= float(l)
        kps[:, 1] -= float(t)
    return FaceMetadata(
        bbox_xyxy=(float(x1 - l), float(y1 - t), float(x2 - l), float(y2 - t)),
        kps=kps,
        det_score=face_meta.det_score,
    )


def face_meta_crop_to_clip(
    face_meta: Optional[FaceMetadata],
    *,
    crop_shape: Tuple[int, int],
    out_hw: Tuple[int, int],
    pad: Pad,
) -> Optional[FaceMetadata]:
    if face_meta is None:
        return None

    ch, cw = int(crop_shape[0]), int(crop_shape[1])
    oh, ow = int(out_hw[0]), int(out_hw[1])
    pt, pb, pl, pr = [int(v) for v in pad]
    sx = float(ow) / float(max(1, cw))
    sy = float(oh) / float(max(1, ch))

    x1, y1, x2, y2 = face_meta.bbox_xyxy
    x1 = x1 * sx + pl
    x2 = x2 * sx + pl
    y1 = y1 * sy + pt
    y2 = y2 * sy + pt

    kps = face_meta.kps
    if kps is not None:
        kps = kps.clone().to(dtype=torch.float32)
        kps[:, 0] = kps[:, 0] * sx + pl
        kps[:, 1] = kps[:, 1] * sy + pt

    return FaceMetadata(
        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        kps=kps,
        det_score=face_meta.det_score,
    )


def _clip_coord_to_crop_coord(v: float, *, scale: float, pad_before: int, limit: int) -> float:
    if scale <= 0.0:
        return float(max(0, min(limit - 1 if limit > 0 else 0, int(v))))
    out = (float(v) - float(pad_before)) / float(scale)
    if limit > 0:
        hi = max(0.0, float(limit - 1))
        out = max(0.0, min(hi, out))
    return out


def face_meta_clip_to_crop(
    face_meta: Optional[FaceMetadata],
    *,
    crop_shape: Tuple[int, int],
    out_hw: Tuple[int, int],
    pad: Pad,
) -> Optional[FaceMetadata]:
    """Map face metadata from clip-space back to original crop-space.

    clip-space = resized+pad coordinates used inside Clip.frames
    crop-space = original ROI crop coordinates used by the worker input image
    """
    if face_meta is None:
        return None

    ch, cw = int(crop_shape[0]), int(crop_shape[1])
    oh, ow = int(out_hw[0]), int(out_hw[1])
    pt, pb, pl, pr = [int(v) for v in pad]
    sx = float(ow) / float(max(1, cw))
    sy = float(oh) / float(max(1, ch))

    x1, y1, x2, y2 = face_meta.bbox_xyxy
    x1 = _clip_coord_to_crop_coord(x1, scale=sx, pad_before=pl, limit=cw)
    x2 = _clip_coord_to_crop_coord(x2, scale=sx, pad_before=pl, limit=cw)
    y1 = _clip_coord_to_crop_coord(y1, scale=sy, pad_before=pt, limit=ch)
    y2 = _clip_coord_to_crop_coord(y2, scale=sy, pad_before=pt, limit=ch)

    kps = face_meta.kps
    if kps is not None:
        kps = kps.clone().to(dtype=torch.float32)
        kps[:, 0] = (kps[:, 0] - float(pl)) / float(max(sx, 1e-8))
        kps[:, 1] = (kps[:, 1] - float(pt)) / float(max(sy, 1e-8))
        if cw > 0:
            kps[:, 0].clamp_(0.0, float(max(0, cw - 1)))
        if ch > 0:
            kps[:, 1].clamp_(0.0, float(max(0, ch - 1)))

    return FaceMetadata(
        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        kps=kps,
        det_score=face_meta.det_score,
    )



@dataclass
class Scene:
    id: int
    frame_start: int
    frame_nums: List[int]
    roi_boxes: List[Box]
    crop_boxes: List[Box]
    crops: List[torch.Tensor]
    masks: List[Optional[torch.Tensor]]
    face_metas: List[Optional[FaceMetadata]]

    def __init__(self, *, id: int, start_frame: int) -> None:
        self.id = int(id)
        self.frame_start = int(start_frame)
        self.frame_nums = []
        self.roi_boxes = []
        self.crop_boxes = []
        self.crops = []
        self.masks = []
        self.face_metas = []

        self.ttl_after_end = 3
        self.end_frame = None
        self.completed_reason = ""

    def mark_completed(self, frame_num: int, reason: str):
        self.end_frame = frame_num
        self.completed_reason = reason
        self.completed = True

    def is_expired(self, current_frame: int) -> bool:
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
        face_meta: Optional[FaceMetadata] = None,
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
        self.face_metas.append(face_meta)

    def append(
        self,
        frame_num: int,
        roi_box: Box,
        crop_box: Box,
        crop_img: torch.Tensor,
        crop_mask: Optional[torch.Tensor],
        face_meta: Optional[FaceMetadata] = None,
    ) -> None:
        self.add_frame(
            frame_num=frame_num,
            roi_box=roi_box,
            crop_box=crop_box,
            crop_img=crop_img,
            crop_mask=crop_mask,
            face_meta=face_meta,
        )

    def to_clip(self, *, clip_id: int, clip_size: int, pad_mode: str = "reflect") -> "Clip":
        return Clip(scene=self, clip_id=int(clip_id), clip_size=int(clip_size), pad_mode=str(pad_mode))

    def merge_same_frame(
        self,
        *,
        roi_box: Box,
        crop_box: Box,
        crop_img: torch.Tensor,
        crop_mask: Optional[torch.Tensor],
        face_meta: Optional[FaceMetadata] = None,
    ) -> None:
        if not self.frame_nums:
            raise AssertionError("Cannot merge into empty scene")
        self.roi_boxes[-1] = _union_box(self.roi_boxes[-1], roi_box)
        self.crop_boxes[-1] = crop_box
        self.crops[-1] = crop_img
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
                    self.masks[-1] = crop_mask

        # For face-swap flows, same-frame scene merges are uncommon. When they do
        # happen we keep the metadata that corresponds to the union crop if present;
        # otherwise preserve the previous value.
        if face_meta is not None:
            self.face_metas[-1] = face_meta

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
    face_metas: List[Optional[FaceMetadata]]

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
        self.face_metas = []

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
            self.face_metas.append(
                face_meta_crop_to_clip(
                    scene.face_metas[i],
                    crop_shape=(ch, cw),
                    out_hw=(out_h, out_w),
                    pad=pad,
                )
            )

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def crop_boxes(self) -> List[Box]:
        return self.boxes

    def pop(self) -> Tuple[torch.Tensor, torch.Tensor, Box, Tuple[int, int], Pad]:
        frame = self.frames.pop(0)
        mask = self.masks.pop(0)
        box = self.boxes.pop(0)
        crop_shape = self.crop_shapes.pop(0)
        pad = self.pad_after_resizes.pop(0)
        if self.face_metas:
            self.face_metas.pop(0)

        if self.frame_nums:
            self.frame_nums.pop(0)

        if self.frame_nums:
            self.frame_start = int(self.frame_nums[0])
            self.frame_end = int(self.frame_nums[-1])
        else:
            old_end = int(self.frame_end)
            self.frame_start = old_end + 1
            self.frame_end = old_end

        return frame, mask, box, crop_shape, pad


__all__ = [
    "Box",
    "Pad",
    "Scene",
    "Clip",
    "FaceMetadata",
    "face_meta_full_to_crop",
    "face_meta_crop_to_clip",
    "face_meta_clip_to_crop",
    "xyxy_to_tlbr",
    "tlbr_to_xyxy",
    "crop_box_to_target_v3",
]
