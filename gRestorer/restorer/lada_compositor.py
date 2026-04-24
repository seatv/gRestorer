from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from gRestorer.core.scene import Box, Pad
from gRestorer.utils import image_utils, mask_utils
from gRestorer.restorer.mosaic_paste_debug import MosaicPasteDebug


def _unpad_any(img: torch.Tensor, pad: Pad) -> torch.Tensor:
    return image_utils.unpad_image(img, pad)


def _resize_img_u8(img_u8: torch.Tensor, shape_hw: Tuple[int, int]) -> torch.Tensor:
    return image_utils.resize(img_u8, size=shape_hw, interpolation=cv2.INTER_LINEAR)


def _resize_mask_u8(mask_u8: torch.Tensor, shape_hw: Tuple[int, int]) -> torch.Tensor:
    if isinstance(mask_u8, torch.Tensor):
        if mask_u8.ndim == 2:
            mask_ch = mask_u8.unsqueeze(-1)
            out = image_utils.resize(mask_ch, size=shape_hw, interpolation=cv2.INTER_NEAREST)
            return out[:, :, 0]
        return image_utils.resize(mask_u8, size=shape_hw, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(mask_u8, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)


def _normalize_blendmask_mode(blendmask_mode: str) -> str:
    mode = str(blendmask_mode or "legacy").strip().lower()
    aliases = {"facefusion": "support", "laplacian": "legacy", "conditioned": "legacy_conditioned"}
    mode = aliases.get(mode, mode)
    if mode not in {"none", "legacy", "legacy_conditioned", "support"}:
        raise ValueError(f"Unsupported blendmask mode: {blendmask_mode!r}")
    return mode


def _build_alpha(mask_u8: torch.Tensor, *, blendmask_mode: str, feather_radius: int = 0) -> torch.Tensor:
    mode = _normalize_blendmask_mode(blendmask_mode)
    mask_f = mask_u8.to(dtype=torch.float32) / 255.0
    if mode == "none":
        return (mask_u8 > 0).to(dtype=torch.float32)
    if mode == "support":
        adaptive_feather = int(feather_radius) if int(feather_radius) > 0 else None
        return mask_utils.create_support_blend_mask(mask_f, feather_px=adaptive_feather).clamp(0.0, 1.0)
    if mode == "legacy_conditioned":
        return mask_utils.create_legacy_conditioned_blend_mask(mask_f).clamp(0.0, 1.0)
    return mask_utils.create_blend_mask(mask_f).clamp(0.0, 1.0)


def _blend_into_frame_lada(
    *,
    frame_bgr_u8: torch.Tensor,
    clip_img_u8: torch.Tensor,
    clip_mask_u8: torch.Tensor,
    orig_clip_box: Box,
    model_dtype: torch.dtype,
    blendmask_mode: str = "legacy",
    feather_radius: int = 0,
    debug_ctx: MosaicPasteDebug | None = None,
) -> None:
    """Direct port of the LADA blend path with selectable alpha construction."""
    t, l, b, r = map(int, orig_clip_box)
    frame_roi = frame_bgr_u8[t : b + 1, l : r + 1]
    blend_mask = _build_alpha(clip_mask_u8, blendmask_mode=blendmask_mode, feather_radius=feather_radius)

    if frame_bgr_u8.device.type != "cuda":
        frame_roi_np = frame_roi.detach().cpu().numpy()
        roi_np = frame_roi_np.astype(np.float32, copy=False)
        clip_np = clip_img_u8.detach().cpu().numpy().astype(np.float32, copy=False)
        bm = blend_mask.detach().cpu().numpy().astype(np.float32, copy=False)
        if bm.ndim == 2:
            bm = bm[:, :, None]
        temp = (clip_np - roi_np) * bm + roi_np
        frame_roi_np[:] = temp.astype(np.uint8)
        frame_roi[:] = torch.from_numpy(frame_roi_np)
        return

    target_dtype = model_dtype
    roi_f = frame_roi.to(dtype=target_dtype)
    temp = clip_img_u8.to(device=frame_roi.device, dtype=target_dtype)
    bm = blend_mask.to(device=frame_roi.device, dtype=target_dtype)
    if bm.ndim == 2:
        bm = bm.unsqueeze(-1)

    temp.sub_(roi_f)
    temp.mul_(bm)
    temp.add_(roi_f)
    temp.round_()
    temp.clamp_(0, 255)
    frame_roi[:] = temp


def composite_lada_clip_into_store(
    *,
    clip,
    restored_frames_u8: List[torch.Tensor],
    store_bgr_u8: Dict[int, torch.Tensor],
    model_dtype: torch.dtype,
    blendmask_mode: str = "legacy",
    feather_radius: int = 0,
    debug_ctx: MosaicPasteDebug | None = None,
) -> None:
    n = min(len(restored_frames_u8), len(clip.frame_nums))
    for i in range(n):
        frame_num = int(clip.frame_nums[i])
        frame = store_bgr_u8.get(frame_num)
        if frame is None:
            continue

        clip_img = restored_frames_u8[i]
        clip_mask = clip.masks[i]
        orig_box: Box = clip.boxes[i]
        orig_shape_hw = clip.crop_shapes[i]
        pad: Pad = clip.pad_after_resizes[i]

        clip_img = _unpad_any(clip_img, pad)
        clip_mask = _unpad_any(clip_mask, pad)
        clip_img = _resize_img_u8(clip_img, orig_shape_hw)
        clip_mask = _resize_mask_u8(clip_mask, orig_shape_hw)

        debug_enabled = bool(debug_ctx is not None and debug_ctx.should_dump(frame_num))
        t, l, b, r = map(int, orig_box)
        orig_roi = frame[t : b + 1, l : r + 1, :].clone() if debug_enabled else None
        legacy_alpha = mask_utils.create_blend_mask(clip_mask.to(dtype=torch.float32) / 255.0).clamp(0.0, 1.0) if debug_enabled else None
        actual_alpha = _build_alpha(clip_mask, blendmask_mode=blendmask_mode, feather_radius=feather_radius) if debug_enabled else None

        _blend_into_frame_lada(
            frame_bgr_u8=frame,
            clip_img_u8=clip_img,
            clip_mask_u8=clip_mask,
            orig_clip_box=orig_box,
            model_dtype=model_dtype,
            blendmask_mode=blendmask_mode,
            feather_radius=feather_radius,
        )

        if debug_enabled and debug_ctx is not None and orig_roi is not None and legacy_alpha is not None:
            debug_ctx.dump(
                frame_num=int(frame_num),
                clip_id=int(getattr(clip, "id", -1)),
                crop_box=(int(t), int(l), int(b), int(r)),
                original_roi=orig_roi,
                restored_roi=clip_img,
                resized_mask=clip_mask,
                legacy_alpha=legacy_alpha,
                actual_alpha=actual_alpha,
                final_roi=frame[t : b + 1, l : r + 1, :].clone(),
            )


__all__ = ["composite_lada_clip_into_store"]
