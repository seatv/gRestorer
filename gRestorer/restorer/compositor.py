from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import torch
import torch.nn.functional as F

from gRestorer.core.scene import Clip
from gRestorer.utils.mask_utils import create_blend_mask, laplacian_pyramid_blend


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
    feather_radius: int = 0,
    quantize_before_resize: bool = False,
    resize_backend: str = "torch",
    blendmask_mode: str = "none"
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

        # Optional experiment knobs kept for pipeline compatibility.
        # For this debug drop-in we preserve the current generic numeric path,
        # but accept the extra arguments so pipeline.py does not explode.
        if quantize_before_resize:
            frm_u = frm_u.mul(255.0).round().clamp(0.0, 255.0).div(255.0)

        # Resize to original crop size.
        # image_utils backend is accepted for compatibility, but this debug drop-in
        # still uses the generic float bilinear path for the restored patch.
        patch = _resize_hwc_float(frm_u, (crop_h, crop_w))
        mask_rs = _resize_hw_mask_u8(m_u, (crop_h, crop_w))

        # LADA-style blend mask to reduce visible paste-back boundaries.
        # It creates a soft transition band near the crop edge while keeping the ROI interior strong.
        alpha = create_blend_mask(mask_rs.to(dtype=torch.float32) / 255.0).clamp(0.0, 1.0)
        # Optional extra feathering knob (usually keep 0 once blend mask is enabled).
        alpha = _feather_alpha(alpha, radius=feather_radius)
        a3 = alpha.unsqueeze(-1)

        t, l, b, r = crop_box
        region_u8 = full[t : b + 1, l : r + 1, :]

        # Blend in float on the (cropped) ROI, then write back to the uint8 buffer.
        region_f = region_u8.to(dtype=torch.float32) / 255.0
        out_f = region_f * (1.0 - a3) + patch * a3
        region_u8.copy_(out_f.mul(255.0).round().clamp(0.0, 255.0).to(dtype=torch.uint8))

        #region_u8.copy_(out_f.mul(255.0).clamp(0.0, 255.0).to(dtype=torch.uint8))



import torch.nn.functional as F

def laplacian_pyramid_blend(original: torch.Tensor, swapped: torch.Tensor, mask: torch.Tensor, levels: int = 3) -> torch.Tensor:
    """
    C++ Equivalent logic: Multi-resolution blending. 
    Prevents 'teeth-ghosting' by blending lighting at low frequencies 
    and texture at high frequencies.
    """
    # Convert HWC to BCHW for PyTorch ops
    o = original.permute(2, 0, 1).unsqueeze(0)
    s = swapped.permute(2, 0, 1).unsqueeze(0)
    m = mask.permute(2, 0, 1).unsqueeze(0)

    # 1. Build Gaussian Pyramids (Downsample stack)
    gauss_o, gauss_s, gauss_m = [o], [s], [m]
    for i in range(levels):
        o = F.interpolate(o, scale_factor=0.5, mode='bilinear', align_corners=False)
        s = F.interpolate(s, scale_factor=0.5, mode='bilinear', align_corners=False)
        m = F.interpolate(m, scale_factor=0.5, mode='bilinear', align_corners=False)
        gauss_o.append(o); gauss_s.append(s); gauss_m.append(m)

    # 2. Build Laplacian Pyramids (Detail maps)
    lap_o, lap_s = [], []
    for i in range(levels):
        size = (gauss_o[i].shape[2], gauss_o[i].shape[3])
        upsampled_o = F.interpolate(gauss_o[i+1], size=size, mode='bilinear', align_corners=False)
        upsampled_s = F.interpolate(gauss_s[i+1], size=size, mode='bilinear', align_corners=False)
        lap_o.append(gauss_o[i] - upsampled_o)
        lap_s.append(gauss_s[i] - upsampled_s)
    lap_o.append(gauss_o[levels])
    lap_s.append(gauss_s[levels])

    # 3. Blend & 4. Reconstruct (Collapse the pyramid)
    res = lap_s[levels] * gauss_m[levels] + lap_o[levels] * (1.0 - gauss_m[levels])
    for i in range(levels - 1, -1, -1):
        size = (lap_o[i].shape[2], lap_o[i].shape[3])
        res = F.interpolate(res, size=size, mode='bilinear', align_corners=False)
        res += lap_s[i] * gauss_m[i] + lap_o[i] * (1.0 - gauss_m[i])

    return res.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)

def _composite_clip_into_store_laplacian(
    *,
    clip: Clip,
    restored_frames: List[torch.Tensor],
    store_bgr_u8: Dict[int, torch.Tensor],
    feather_radius: int = 0,
    quantize_before_resize: bool = False,
    resize_backend: str = "torch",
    use_laplacian: bool = True  # New default for your tongue/tile issues
) -> None:
    for i, frame_num in enumerate(clip.frame_nums):
        full = store_bgr_u8[frame_num]
        frm_u = restored_frames[i]
        
        # Metadata recovery
        m_u = frm_u.get_metadata("mask")
        crop_box = frm_u.get_metadata("crop_box")
        pad = frm_u.get_metadata("pad")
        crop_h, crop_w = (crop_box[2] - crop_box[0] + 1), (crop_box[3] - crop_box[1] + 1)

        # Pre-processing
        frm_u = _unpad_hwc(frm_u, pad)
        patch = _resize_hwc_float(frm_u, (crop_h, crop_w))
        mask_rs = _resize_hw_mask_u8(m_u, (crop_h, crop_w))
        
        # ROI Pointers
        t, l, b, r = crop_box
        region_u8 = full[t : b + 1, l : r + 1, :]
        region_f = region_u8.to(dtype=torch.float32) / 255.0

        # MASK GENERATION
        # Convert u8 mask to float [0,1]
        alpha_mask = mask_rs.to(dtype=torch.float32).unsqueeze(-1) / 255.0
        
        # APPLY BLEND
        if use_laplacian:
            # Solve tongue/lighting issues by frequency-blending
            out_f = laplacian_pyramid_blend(region_f, patch, alpha_mask, levels=3)
        else:
            # Fallback to your original linear feathering
            alpha = create_blend_mask(alpha_mask.squeeze()).clamp(0.0, 1.0)
            alpha = _feather_alpha(alpha, radius=feather_radius).unsqueeze(-1)
            out_f = region_f * (1.0 - alpha) + patch * alpha

        # Write back to shared memory / buffer
        region_u8.copy_(out_f.mul(255.0).round().clamp(0.0, 255.0).to(dtype=torch.uint8))

def _composite_clip_into_store_laplacian(
    *,
    clip: Clip,
    restored_frames: List[torch.Tensor],
    store_bgr_u8: Dict[int, torch.Tensor],
    feather_radius: int = 0,
    quantize_before_resize: bool = False,
    resize_backend: str = "torch",
    blendmask_mode: str = "laplacian",
) -> None:
    """Alternate compositor that preserves the original path and only changes blending.

    This function intentionally mirrors _composite_clip_into_store() preprocessing:
      - same unpad path
      - same optional quantize-before-resize path
      - same resize path
      - same create_blend_mask() + feathering path

    The only semantic difference is the final blend operator:
      - original compositor: linear alpha blend
      - this compositor:    Laplacian pyramid blend
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

        frm_u = _unpad_hwc(frm, pad)
        m_u = clip.masks[i]
        m_u = _unpad_hwc(m_u.unsqueeze(-1), pad).squeeze(-1)

        if quantize_before_resize:
            frm_u = frm_u.mul(255.0).round().clamp(0.0, 255.0).div(255.0)

        patch = _resize_hwc_float(frm_u, (crop_h, crop_w))
        mask_rs = _resize_hw_mask_u8(m_u, (crop_h, crop_w))

        alpha = create_blend_mask(mask_rs.to(dtype=torch.float32) / 255.0).clamp(0.0, 1.0)
        alpha = _feather_alpha(alpha, radius=feather_radius)
        a3 = alpha.unsqueeze(-1)

        t, l, b, r = crop_box
        region_u8 = full[t : b + 1, l : r + 1, :]
        region_f = region_u8.to(dtype=torch.float32) / 255.0

        out_f = laplacian_pyramid_blend(region_f, patch, a3, levels=4)
        region_u8.copy_(out_f.mul(255.0).round().clamp(0.0, 255.0).to(dtype=torch.uint8))
