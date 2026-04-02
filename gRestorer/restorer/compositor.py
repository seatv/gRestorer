from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from gRestorer.core.scene import Clip
from gRestorer.utils.mask_utils import create_blend_mask, create_support_blend_mask


def _unpad_hwc(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    """Remove padding (pt, pb, pl, pr) from an HWC tensor."""
    pt, pb, pl, pr = [int(v) for v in pad]
    h, w = int(x.shape[0]), int(x.shape[1])
    y0 = pt
    y1 = h - pb if pb > 0 else h
    x0 = pl
    x1 = w - pr if pr > 0 else w
    return x[y0:y1, x0:x1, :]


def _resize_hwc_float(x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """Bilinear resize for float32 HWC -> float32 HWC."""
    oh, ow = int(out_hw[0]), int(out_hw[1])
    y = x.permute(2, 0, 1).unsqueeze(0)
    y = F.interpolate(y, size=(oh, ow), mode="bilinear", align_corners=False)
    return y.squeeze(0).permute(1, 2, 0)


def _resize_hw_mask_u8(m: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """Nearest resize for uint8 HW -> uint8 HW."""
    oh, ow = int(out_hw[0]), int(out_hw[1])
    y = m.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    y = F.interpolate(y, size=(oh, ow), mode="nearest")
    y = y.squeeze(0).squeeze(0)
    return y.clamp(0.0, 255.0).to(dtype=torch.uint8)


def _feather_alpha(alpha_hw: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Optional extra feathering pass for experimentation."""
    r = int(radius)
    if r <= 0:
        return alpha_hw

    a = alpha_hw.unsqueeze(0).unsqueeze(0)
    k = 2 * r + 1
    a = F.avg_pool2d(a, kernel_size=k, stride=1, padding=r)
    a = a.squeeze(0).squeeze(0)

    if bool((alpha_hw > 0).any()):
        a = a * (alpha_hw > 0).to(dtype=a.dtype)
        amax = a.max()
        if float(amax) > 0.0:
            a = a / amax

    return a.clamp(0.0, 1.0)


def _alpha_support_box(alpha_hw: torch.Tensor, eps: float = 1e-6) -> Optional[Tuple[int, int, int, int]]:
    """Return (top, left, bottom, right) for the non-zero alpha support."""
    nz = alpha_hw > float(eps)
    if not bool(nz.any()):
        return None

    rows = torch.where(nz.any(dim=1))[0]
    cols = torch.where(nz.any(dim=0))[0]
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])



def _composite_clip_into_store(
    *,
    clip: Clip,
    restored_frames: List[torch.Tensor],
    store_bgr_u8: Dict[int, torch.Tensor],
    feather_radius: int = 0,
    quantize_before_resize: bool = False,
    resize_backend: str = "torch",
) -> None:
    """Paste restored clip results back into buffered full frames (in-place).

    Mainline compositor only.

    Changes versus the previous version:
      - alpha is built from the *actual resized mask support*, not from a fixed
        inner rectangle
      - compositing is limited to the effective alpha support box instead of the
        entire ROI rectangle
      - legacy blend-mask logic is kept as a fallback if the support-driven mask
        ends up empty
    """
    if len(restored_frames) != len(clip):
        raise ValueError(f"restored_frames length ({len(restored_frames)}) != clip length ({len(clip)})")

    clip_size = int(clip.clip_size)

    for i, frame_num in enumerate(clip.frame_nums):
        full = store_bgr_u8.get(int(frame_num))
        if full is None:
            continue

        crop_box = clip.crop_boxes[i]
        crop_h, crop_w = clip.crop_shapes[i]
        pad = clip.pad_after_resizes[i]

        frm = restored_frames[i]
        if frm.shape[0] != clip_size or frm.shape[1] != clip_size:
            raise ValueError(f"Restored frame must be {clip_size}x{clip_size}, got {tuple(frm.shape)}")

        # Unpad back to the resized crop.
        frm_u = _unpad_hwc(frm, pad)
        m_u = clip.masks[i]
        m_u = _unpad_hwc(m_u.unsqueeze(-1), pad).squeeze(-1)

        if quantize_before_resize:
            frm_u = frm_u.mul(255.0).round().clamp(0.0, 255.0).div(255.0)

        # Resize back to original crop size.
        patch = _resize_hwc_float(frm_u, (crop_h, crop_w))
        mask_rs = _resize_hw_mask_u8(m_u, (crop_h, crop_w))

        # New mainline path: build alpha from the real resized mask support.
        alpha = create_support_blend_mask(mask_rs)

        # Fallback to the legacy alpha if support somehow disappears.
        if not bool((alpha > 0).any()):
            alpha = create_blend_mask(mask_rs.to(dtype=torch.float32) / 255.0).clamp(0.0, 1.0)

        # Optional extra feathering knob for experiments.
        alpha = _feather_alpha(alpha, radius=feather_radius)

        support_box = _alpha_support_box(alpha)
        if support_box is None:
            continue

        mt, ml, mb, mr = support_box
        alpha_sub = alpha[mt:mb + 1, ml:mr + 1]
        patch_sub = patch[mt:mb + 1, ml:mr + 1, :]

        t, l, b, r = crop_box
        rt = t + mt
        rl = l + ml
        rb = t + mb
        rr = l + mr

        region_u8 = full[rt:rb + 1, rl:rr + 1, :]

        # Safety guard for rare off-by-one shape mismatches.
        if region_u8.shape[0] != patch_sub.shape[0] or region_u8.shape[1] != patch_sub.shape[1]:
            hh = min(int(region_u8.shape[0]), int(patch_sub.shape[0]))
            ww = min(int(region_u8.shape[1]), int(patch_sub.shape[1]))
            if hh <= 0 or ww <= 0:
                continue
            region_u8 = region_u8[:hh, :ww, :]
            patch_sub = patch_sub[:hh, :ww, :]
            alpha_sub = alpha_sub[:hh, :ww]

        a3 = alpha_sub.unsqueeze(-1)
        region_f = region_u8.to(dtype=torch.float32) / 255.0
        out_f = region_f * (1.0 - a3) + patch_sub * a3
        region_u8.copy_(out_f.mul(255.0).round().clamp(0.0, 255.0).to(dtype=torch.uint8))
