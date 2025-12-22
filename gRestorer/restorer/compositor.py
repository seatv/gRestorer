from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from gRestorer.core.scene import Clip


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
    """Cheap edge feathering: a few avg-pool passes."""
    r = int(radius)
    if r <= 0:
        return alpha_hw

    a = alpha_hw.unsqueeze(0).unsqueeze(0)
    k = 2 * r + 1
    a = F.avg_pool2d(a, kernel_size=k, stride=1, padding=r)
    return a.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def _composite_clip_into_store(
    *,
    clip: Clip,
    restored_frames: List[torch.Tensor],
    store_bgr_u8: Dict[int, torch.Tensor],
    feather_radius: int = 3,
) -> None:
    """Paste restored clip results back into buffered full frames (in-place)."""
    if len(restored_frames) != len(clip):
        raise ValueError(f"restored_frames length ({len(restored_frames)}) != clip length ({len(clip)})")

    clip_size = int(clip.clip_size)

    for i, frame_num in enumerate(clip.frame_nums):
        full = store_bgr_u8.get(int(frame_num))
        if full is None:
            # Shouldn't happen in the streaming design, but don't crash.
            continue

        crop_box = clip.crop_boxes[i]
        crop_h, crop_w = clip.crop_shapes[i]
        pad = clip.pad_after_resizes[i]

        # Restored frame is float HWC in [0,1] with clip_size.
        frm = restored_frames[i]
        if frm.shape[0] != clip_size or frm.shape[1] != clip_size:
            raise ValueError(f"Restored frame must be {clip_size}x{clip_size}, got {tuple(frm.shape)}")

        # Unpad back to the resized crop.
        frm_u = _unpad_hwc(frm, pad)
        m_u = clip.masks[i]
        m_u = _unpad_hwc(m_u.unsqueeze(-1), pad).squeeze(-1)

        # Resize to original crop size.
        patch = _resize_hwc_float(frm_u, (crop_h, crop_w))
        mask_rs = _resize_hw_mask_u8(m_u, (crop_h, crop_w))

        alpha = (mask_rs.to(dtype=torch.float32) / 255.0).clamp(0.0, 1.0)
        alpha = _feather_alpha(alpha, radius=feather_radius)
        a3 = alpha.unsqueeze(-1)

        t, l, b, r = crop_box
        region_u8 = full[t : b + 1, l : r + 1, :]

        # Blend in float on the (cropped) ROI, then write back to the uint8 buffer.
        region_f = region_u8.to(dtype=torch.float32) / 255.0
        out_f = region_f * (1.0 - a3) + patch * a3
        region_u8.copy_(out_f.mul(255.0).clamp(0.0, 255.0).to(dtype=torch.uint8))


