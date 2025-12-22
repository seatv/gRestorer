from __future__ import annotations

from typing import Sequence, Tuple

import torch

from gRestorer.utils.config_util import Config


def _sync_device(device: torch.device) -> None:
    """Sync torch work before NVENC reads GPU buffers (CUDA/XPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        # only present on torch builds with XPU support
        try:
            torch.xpu.synchronize(device)  # type: ignore[attr-defined]
        except Exception:
            pass

def wrap_surface_as_tensor(surface) -> torch.Tensor:
    """Wrap a PyNvVideoCodec surface (DLPack) as a torch tensor without copies."""
    return torch.from_dlpack(surface)


def rgb_to_bgr_u8_inplace(dst_bgr: torch.Tensor, src_rgb: torch.Tensor) -> None:
    """Convert RGB->BGR into dst (uint8 HWC), from src RGB HWC or RGBP CHW."""
    if dst_bgr.dtype != torch.uint8 or dst_bgr.ndim != 3 or dst_bgr.shape[-1] != 3:
        raise ValueError(f"dst_bgr must be uint8 [H,W,3], got {dst_bgr.dtype} {tuple(dst_bgr.shape)}")

    if src_rgb.ndim != 3:
        raise ValueError(f"src_rgb must be 3D, got {tuple(src_rgb.shape)}")

    # RGBP planar CHW
    if src_rgb.shape[0] == 3 and src_rgb.shape[-1] != 3:
        r = src_rgb[0]
        g = src_rgb[1]
        b = src_rgb[2]
        dst_bgr[..., 0].copy_(b)
        dst_bgr[..., 1].copy_(g)
        dst_bgr[..., 2].copy_(r)
        return

    # Packed HWC
    if src_rgb.shape[-1] == 3:
        dst_bgr[..., 0].copy_(src_rgb[..., 2])
        dst_bgr[..., 1].copy_(src_rgb[..., 1])
        dst_bgr[..., 2].copy_(src_rgb[..., 0])
        return

    raise ValueError(f"Unsupported src_rgb shape={tuple(src_rgb.shape)}")


def pack_bgr_u8_to_bgra_u8_inplace(dst_bgra: torch.Tensor, src_bgr: torch.Tensor) -> None:
    """Pack BGR uint8 [H,W,3] -> BGRA uint8 [H,W,4] (alpha=255)."""
    if dst_bgra.dtype != torch.uint8 or dst_bgra.ndim != 3 or dst_bgra.shape[-1] != 4:
        raise ValueError(f"dst_bgra must be uint8 [H,W,4], got {dst_bgra.dtype} {tuple(dst_bgra.shape)}")
    if src_bgr.dtype != torch.uint8 or src_bgr.ndim != 3 or src_bgr.shape[-1] != 3:
        raise ValueError(f"src_bgr must be uint8 [H,W,3], got {src_bgr.dtype} {tuple(src_bgr.shape)}")

    dst_bgra[..., 0:3].copy_(src_bgr)
    dst_bgra[..., 3].fill_(255)


def _cfg_get(cfg: Config, *keys: str, default=None):
    try:
        return cfg.get(*keys, default=default)
    except TypeError:
        # In case a raw dict sneaks in.
        d = cfg  # type: ignore
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k)
            if d is None:
                return default
        return d


def _cfg_first(cfg: Config, paths: Sequence[Tuple[str, ...]], default=None):
    for p in paths:
        v = _cfg_get(cfg, *p, default=None)
        if v is not None:
            return v
    return default


def _parse_color_bgr(v) -> Optional[Tuple[int, int, int]]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) == 3:
        b, g, r = v
        return (int(b), int(g), int(r))
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",")]
        if len(parts) == 3:
            b, g, r = [int(x) for x in parts]
            return (b, g, r)
    raise ValueError(f"Invalid color value: {v!r} (expected [b,g,r] or 'b,g,r')")


