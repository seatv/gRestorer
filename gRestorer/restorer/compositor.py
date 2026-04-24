from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from gRestorer.core.scene import Clip
from gRestorer.utils.mask_utils import (
    create_blend_mask,
    create_support_blend_mask,
)
from gRestorer.restorer.mosaic_paste_debug import MosaicPasteDebug


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

def _blur_hwc_float(x: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Cheap box blur for float32 HWC image in [0,1]."""
    r = int(radius)
    if r <= 0:
        return x
    k = 2 * r + 1
    y = x.permute(2, 0, 1).unsqueeze(0)
    y = F.avg_pool2d(y, kernel_size=k, stride=1, padding=r)
    return y.squeeze(0).permute(1, 2, 0)


def _apply_support_inner_sharpen(
    roi_f: torch.Tensor,
    alpha_hw: torch.Tensor,
    *,
    amount: float = 0.18,
    blur_radius: int = 1,
    start_alpha: float = 0.35,
    full_alpha: float = 0.85,
) -> torch.Tensor:
    """
    Mild detail compensation inside support-mode paste.

    - no effect in the low-alpha outer ring
    - ramps up as alpha increases inward
    - keeps the support alpha unchanged
    """
    amt = float(amount)
    if amt <= 0.0:
        return roi_f

    alpha_hw = alpha_hw.clamp(0.0, 1.0)
    denom = max(1e-6, float(full_alpha) - float(start_alpha))
    zone = ((alpha_hw - float(start_alpha)) / denom).clamp(0.0, 1.0)

    if not bool((zone > 0).any()):
        return roi_f

    zone3 = zone.unsqueeze(-1)
    blurred = _blur_hwc_float(roi_f, radius=blur_radius)
    sharpened = (roi_f + amt * (roi_f - blurred)).clamp(0.0, 1.0)

    return (roi_f * (1.0 - zone3) + sharpened * zone3).clamp(0.0, 1.0)

def _normalize_blendmask_mode(blendmask_mode: str) -> str:
    mode = str(blendmask_mode or "legacy").strip().lower()
    aliases = {
        "facefusion": "support",
        "laplacian": "legacy",
        "conditioned": "legacy_conditioned",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"none", "legacy", "legacy_conditioned", "support"}:
        raise ValueError(f"Unsupported blendmask mode: {blendmask_mode!r}")
    return mode

def _build_alpha(mask_rs: torch.Tensor, *, blendmask_mode: str, feather_radius: int) -> torch.Tensor:
    mode = _normalize_blendmask_mode(blendmask_mode)
    mask_f = mask_rs.to(dtype=torch.float32) / 255.0

    if mode == "none":
        return (mask_rs > 0).to(dtype=torch.float32)

    if mode == "support":
        adaptive_feather = int(feather_radius) if int(feather_radius) > 0 else None
        return create_support_blend_mask(mask_f, feather_px=adaptive_feather).clamp(0.0, 1.0)

    # For now, keep legacy_conditioned on the legacy path until that experiment is reintroduced.
    alpha = create_blend_mask(mask_f).clamp(0.0, 1.0)
    return _feather_alpha(alpha, radius=feather_radius)

def _composite_clip_into_store(
    *,
    clip: Clip,
    restored_frames: List[torch.Tensor],
    store_bgr_u8: Dict[int, torch.Tensor],
    feather_radius: int = 0,
    quantize_before_resize: bool = False,
    resize_backend: str = "torch",
    blendmask_mode: str = "legacy",
    sharpen: bool = False,
    debug_ctx: MosaicPasteDebug | None = None,
    ) -> None:
    """Paste restored clip results back into buffered full frames (in-place)."""
    if len(restored_frames) != len(clip):
        raise ValueError(f"restored_frames length ({len(restored_frames)}) != clip length ({len(clip)})")

    clip_size = int(clip.clip_size)

    mode = _normalize_blendmask_mode(blendmask_mode)


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

        frm_u = _unpad_hwc(frm, pad)
        m_u = clip.masks[i]
        m_u = _unpad_hwc(m_u.unsqueeze(-1), pad).squeeze(-1)

        if quantize_before_resize:
            frm_u = frm_u.mul(255.0).round().clamp(0.0, 255.0).div(255.0)

        patch = _resize_hwc_float(frm_u, (crop_h, crop_w))
        mask_rs = _resize_hw_mask_u8(m_u, (crop_h, crop_w))
        alpha = _build_alpha(mask_rs, blendmask_mode=blendmask_mode, feather_radius=feather_radius)

        t, l, b, r = crop_box
        debug_enabled = bool(debug_ctx is not None and debug_ctx.should_dump(frame_num))
        orig_roi = full[t : b + 1, l : r + 1, :].clone() if debug_enabled else None
        legacy_alpha = create_blend_mask(mask_rs.to(dtype=torch.float32) / 255.0).clamp(0.0, 1.0) if debug_enabled else None

        nz = alpha > 0
        if bool(nz.any()):
            rows = torch.where(nz.any(dim=1))[0]
            cols = torch.where(nz.any(dim=0))[0]
            at, al, ab, ar = int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])
        else:
            at, al, ab, ar = 0, 0, crop_h - 1, crop_w - 1

        a_hw = alpha[at: ab + 1, al: ar + 1]
        a3 = a_hw.unsqueeze(-1)
        patch_roi = patch[at: ab + 1, al: ar + 1, :]

        region_u8 = full[t + at: t + ab + 1, l + al: l + ar + 1, :]
        region_f = region_u8.to(dtype=torch.float32) / 255.0
        out_f = region_f * (1.0 - a3) + patch_roi * a3

        if sharpen and mode == "support":
            out_f = _apply_support_inner_sharpen(out_f, a_hw, amount=0.18, blur_radius=1)
        elif sharpen and mode == "legacy":
            out_f = _apply_support_inner_sharpen(out_f, a_hw, amount=0.12, blur_radius=1)

        region_u8.copy_(out_f.mul(255.0).round().clamp(0.0, 255.0).to(dtype=torch.uint8))

        if debug_enabled and debug_ctx is not None and orig_roi is not None and legacy_alpha is not None:
            debug_ctx.dump(
                frame_num=int(frame_num),
                clip_id=int(getattr(clip, "id", -1)),
                crop_box=(int(t), int(l), int(b), int(r)),
                original_roi=orig_roi,
                restored_roi=patch,
                resized_mask=mask_rs,
                legacy_alpha=legacy_alpha,
                actual_alpha=alpha,
                final_roi=full[t : b + 1, l : r + 1, :].clone(),
            )


__all__ = ["_composite_clip_into_store"]
