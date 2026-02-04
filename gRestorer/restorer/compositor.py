from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

from gRestorer.core.scene import Clip
from gRestorer.utils.mask_utils import create_blend_mask



def _dump_bgr_png(img_hwc_u8: torch.Tensor, frame_num: int, tag: str) -> None:
    dump_dir = os.environ.get("GR_DUMP_DIR", "").strip().strip('"').strip("'")
    if not dump_dir:
        return
    try:
        from PIL import Image
    except Exception:
        return

    Path(dump_dir).mkdir(parents=True, exist_ok=True)

    x = img_hwc_u8
    if x.dtype != torch.uint8:
        x = x.clamp(0, 255).to(dtype=torch.uint8)

    # Expect BGR; convert to RGB for saving
    rgb = x[..., [2, 1, 0]].contiguous().cpu().numpy()
    Image.fromarray(rgb).save(str(Path(dump_dir) / f"{tag}_{int(frame_num):06d}.png"))


def _dump_gray_png(gray_hw01: torch.Tensor, frame_num: int, tag: str) -> None:
    dump_dir = os.environ.get("GR_DUMP_DIR", "").strip().strip('"').strip("'")
    if not dump_dir:
        return
    try:
        from PIL import Image
    except Exception:
        return

    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    a = (gray_hw01.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8).cpu().numpy()
    Image.fromarray(a).save(str(Path(dump_dir) / f"{tag}_{int(frame_num):06d}.png"))


@dataclass(frozen=True)
class CompositeParams:
    """Parameters for compositing restored patches back into full frames."""
    feather_radius: int = 0


def _unpad_hwc(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    """Remove padding (pt, pb, pl, pr) from an HWC tensor."""
    pt, pb, pl, pr = [int(v) for v in pad]
    h, w = int(x.shape[0]), int(x.shape[1])
    y0 = pt
    y1 = h - pb if pb > 0 else h
    x0 = pl
    x1 = w - pr if pr > 0 else w
    return x[y0:y1, x0:x1, ...]


def _resize_hwc(x: torch.Tensor, out_hw: Tuple[int, int], *, mode: str) -> torch.Tensor:
    """Resize an HWC tensor using torch.interpolate."""
    oh, ow = int(out_hw[0]), int(out_hw[1])
    if x.ndim != 3:
        raise ValueError(f"_resize_hwc expects HWC, got shape={tuple(x.shape)}")
    y = x.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    y = F.interpolate(
        y,
        size=(oh, ow),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    return y.squeeze(0).permute(1, 2, 0)  # H,W,C


def _resize_hw(x: torch.Tensor, out_hw: Tuple[int, int], *, mode: str) -> torch.Tensor:
    """Resize an HW tensor using torch.interpolate."""
    oh, ow = int(out_hw[0]), int(out_hw[1])
    if x.ndim != 2:
        raise ValueError(f"_resize_hw expects HW, got shape={tuple(x.shape)}")
    y = x.unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    y = F.interpolate(
        y,
        size=(oh, ow),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    return y.squeeze(0).squeeze(0)


def _feather_alpha(alpha_hw: torch.Tensor, radius: int = 0) -> torch.Tensor:
    """Optional extra feathering on top of LADA blend mask."""
    r = int(radius)
    if r <= 0:
        return alpha_hw
    a = alpha_hw.unsqueeze(0).unsqueeze(0)
    k = 2 * r + 1
    a = F.avg_pool2d(a, kernel_size=k, stride=1, padding=r)
    return a.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def _dump_alpha_png(alpha_hw: torch.Tensor, frame_num: int, tag: str = "alpha") -> None:
    """Write alpha/masks (H,W float in [0,1]) as 8-bit PNG for debugging."""
    raw_dir = os.environ.get("GR_DUMP_ALPHA_DIR", r"D:\Results\alpha_dump")
    raw_dir = str(raw_dir).strip().strip('"').strip("'")
    if not raw_dir:
        return
    try:
        from PIL import Image
    except Exception:
        return

    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    a = (alpha_hw.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8).cpu().numpy()
    Image.fromarray(a).save(str(Path(raw_dir) / f"{tag}_{int(frame_num):06d}.png"))


def _as_float255_hwc(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a restored patch frame to float32 in [0,255] (HWC).
    Accepts:
      - uint8 [0..255]
      - float [0..1] (assumed) or [0..255] (rare)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected HWC, got {tuple(x.shape)}")
    if x.dtype == torch.uint8:
        return x.to(dtype=torch.float32)
    xf = x.to(dtype=torch.float32)
    if float(xf.max().item()) <= 1.5:
        xf = xf * 255.0
    return xf


def _composite_clip_into_store(
    *,
    clip: Clip,
    restored_frames: List[torch.Tensor],
    store_bgr_u8: Dict[int, torch.Tensor],
    feather_radius: int = 0,
    lada_parity: bool = True,
) -> None:
    """Paste restored clip results back into buffered full frames (in-place)."""
    if len(restored_frames) != len(clip):
        raise ValueError(f"restored_frames length ({len(restored_frames)}) != clip length ({len(clip)})")

    clip_size = int(clip.clip_size)

    for i, frame_num in enumerate(clip.frame_nums):
        full = store_bgr_u8.get(int(frame_num))
        if full is None:
            continue

        # Geometry from clip
        crop_box = clip.crop_boxes[i]
        pad = clip.pad_after_resizes[i]  # ✅ FIX: define pad here

        # Paste region geometry MUST come from crop_box (full-res target)
        t, l, b, r = [int(v) for v in crop_box]
        target_h = int(b - t + 1)
        target_w = int(r - l + 1)

        # Full-res region in the original frame
        region_u8 = full[t: b + 1, l: r + 1, :]
        region_255 = region_u8.to(dtype=torch.float32)

        # Restored clip frame: 256x256 (clip space) -> unpad -> resize to full-res ROI
        frm = restored_frames[i]
        frm_u = _unpad_hwc(frm, pad)  # now pad exists
        patch_255 = _as_float255_hwc(frm_u)
        patch_255 = _resize_hwc(patch_255, (target_h, target_w), mode="bilinear")

        # Mask: 256x256 -> unpad -> resize to full-res ROI
        m_u = clip.masks[i]
        if m_u.ndim != 2:
            m_u = m_u.squeeze()
        m_u = _unpad_hwc(m_u.unsqueeze(-1), pad).squeeze(-1)

        mask_rs = _resize_hw(m_u.to(dtype=torch.float32), (target_h, target_w), mode="nearest")
        mask01 = (mask_rs / 255.0).clamp(0.0, 1.0)

        alpha = create_blend_mask(mask01).clamp(0.0, 1.0)
        alpha = _feather_alpha(alpha, radius=feather_radius)
        a3 = alpha.unsqueeze(-1)

        dump_frame = os.environ.get("GR_DUMP_FRAME", "").strip().strip('"').strip("'")
        if dump_frame.isdigit() and int(dump_frame) == int(frame_num):
            # region before (BGR)
            _dump_bgr_png(region_u8, frame_num, "region_before")

            # patch (convert float [0..255] -> u8 BGR)
            _dump_bgr_png(patch_255.round().clamp(0, 255).to(torch.uint8), frame_num, "patch")

            # alpha
            _dump_gray_png(alpha, frame_num, "alpha")

        # LADA blend formulation
        if lada_parity:
            out_255 = (patch_255 - region_255) * a3 + region_255
        else:
            out_255 = region_255 * (1.0 - a3) + patch_255 * a3

        out_u8 = out_255.round().clamp(0.0, 255.0).to(dtype=torch.uint8)

        m3 = (alpha > 0).unsqueeze(-1)
        region_u8.copy_(torch.where(m3, out_u8, region_u8))

        if dump_frame.isdigit() and int(dump_frame) == int(frame_num):
            _dump_bgr_png(region_u8, frame_num, "region_after")

        dump_one = os.environ.get("GR_DUMP_ALPHA_FRAME", "").strip().strip('"').strip("'").strip()
        if dump_one.isdigit() and int(dump_one) == int(frame_num):
            _dump_alpha_png(alpha, frame_num, tag="alpha")
            _dump_alpha_png(mask01, frame_num, tag="mask01")


def composite_clip_into_store(
    *,
    clip: Clip,
    restored_frames: List[torch.Tensor],
    store_bgr_u8: Dict[int, torch.Tensor],
    restorer_dtype: Optional[torch.dtype] = None,
    lada_parity: bool = True,
    params: Optional[CompositeParams] = None,
) -> None:
    """
    Public API used by the CLI pipeline.

    - `restorer_dtype` is accepted for compatibility with the pipeline but blending is
      intentionally done in float32 for stability, then quantized to uint8.
    - `lada_parity=True` uses the LADA-friendly formulation in [0,255] space.
    """
    _ = restorer_dtype  # reserved for future fast-paths; keep signature stable
    pr = int(params.feather_radius) if params is not None else 0
    _composite_clip_into_store(
        clip=clip,
        restored_frames=restored_frames,
        store_bgr_u8=store_bgr_u8,
        feather_radius=pr,
        lada_parity=bool(lada_parity),
    )
