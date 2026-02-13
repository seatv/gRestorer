# gRestorer/cli/pipeline_utils.py
from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

Box = Tuple[int, int, int, int]  # (t,l,b,r) inclusive


def now_ms() -> int:
    return int(time.time() * 1000)


@contextlib.contextmanager
def timing(name: str, enabled: bool = True) -> Iterator[None]:
    if not enabled:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[Timing] {name}: {dt:.2f} ms")


def sync_device(device: torch.device) -> None:
    """Sync torch work before NVENC/NVDEC reads GPU buffers (CUDA/XPU)."""
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize(device=device)  # type: ignore[attr-defined]
    except Exception:
        # Best-effort; do not crash.
        pass


# Backwards-compatible alias (older code referenced this name)
_sync_device = sync_device


def cfg_first(cfg, paths: Sequence[Sequence[str]], default=None):
    """
    Return the first non-None config value among multiple key-paths.
    Example:
        cfg_first(cfg, [("decoder","gpu_id"), ("gpu_id",)], default=0)
    """
    for keys in paths:
        try:
            v = cfg.get(*keys, default=None)
        except Exception:
            v = None
        if v is not None:
            return v
    return default


def cfg_path(cfg, keys: Sequence[str], default: str = "") -> str:
    """Read a string path from config and normalize/expand it."""
    v = cfg.get(*keys, default=default)
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    s = os.path.expandvars(os.path.expanduser(s))
    return s


def wrap_surface_as_tensor(surface) -> torch.Tensor:
    """
    PyNvVideoCodec surfaces support DLPack; torch.from_dlpack gives a tensor view.
    The returned tensor is usually uint8 and on GPU.
    """
    return torch.from_dlpack(surface)


def rgbp_chw_to_rgb_hwc_u8(x: torch.Tensor) -> torch.Tensor:
    """
    Decoder output is typically RGBP CHW uint8 on GPU.
    Return RGB HWC uint8 contiguous.
    """
    if x.ndim == 3 and x.shape[0] == 3:
        y = x.permute(1, 2, 0)
    elif x.ndim == 3 and x.shape[-1] == 3:
        y = x
    else:
        raise ValueError(f"Unexpected frame tensor shape: {tuple(x.shape)} (expected CHW or HWC RGB)")
    if y.dtype != torch.uint8:
        y = y.to(torch.uint8)
    return y.contiguous()


def rgb_hwc_to_bgr_hwc_u8(rgb: torch.Tensor) -> torch.Tensor:
    """
    RGB HWC -> BGR HWC. Returns contiguous uint8.
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {tuple(rgb.shape)}")
    if rgb.dtype != torch.uint8:
        rgb = rgb.to(torch.uint8)
    return rgb.flip(-1).contiguous()


def bgr_u8_to_bgra_u8(bgr: torch.Tensor) -> torch.Tensor:
    """
    Encoder expects BGRA uint8 HWC (ARGB format, little-endian).
    """
    if bgr.ndim != 3 or bgr.shape[-1] != 3:
        raise ValueError(f"Expected HWC BGR, got {tuple(bgr.shape)}")
    if bgr.dtype != torch.uint8:
        bgr = bgr.to(torch.uint8)
    h, w, _ = bgr.shape
    out = torch.empty((h, w, 4), device=bgr.device, dtype=torch.uint8)
    out[..., :3].copy_(bgr)
    out[..., 3].fill_(255)
    return out


def clip_box_to_bounds(box: Box, w: int, h: int) -> Box:
    t, l, b, r = box
    t = max(0, min(int(t), h - 1))
    b = max(0, min(int(b), h - 1))
    l = max(0, min(int(l), w - 1))
    r = max(0, min(int(r), w - 1))
    if b < t:
        t, b = b, t
    if r < l:
        l, r = r, l
    return (t, l, b, r)


def seam_split_boxes(
    boxes: Sequence[Box],
    seam_x: int,
    full_w: int,
    full_h: int,
    masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
) -> Tuple[List[Box], Optional[List[Optional[torch.Tensor]]]]:
    """
    Ensure no box crosses the SBS seam. If a box spans the seam, split into up to two.
    Masks (if provided) are *not* precisely split; for seam-crossing boxes we drop the mask (None)
    so downstream uses rectangle masks safely.
    """
    out_boxes: List[Box] = []
    out_masks: Optional[List[Optional[torch.Tensor]]] = [] if masks is not None else None

    for i, box in enumerate(boxes):
        t, l, b, r = clip_box_to_bounds(box, full_w, full_h)
        m = masks[i] if masks is not None else None

        crosses = (l < seam_x) and (r >= seam_x)
        if not crosses:
            out_boxes.append((t, l, b, r))
            if out_masks is not None:
                out_masks.append(m)
            continue

        # Left part
        if l <= seam_x - 1:
            out_boxes.append((t, l, b, seam_x - 1))
            if out_masks is not None:
                out_masks.append(None)

        # Right part
        if r >= seam_x:
            out_boxes.append((t, seam_x, b, r))
            if out_masks is not None:
                out_masks.append(None)

    return out_boxes, out_masks


def split_frame_lr(rgb_hwc: torch.Tensor, layout: str = "lr") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a full RGB frame into left/right halves.
    layout:
      - "lr": left half is left eye, right half is right eye
      - "rl": swapped (right eye on left half, left eye on right half)
    """
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[-1] != 3:
        raise ValueError("split_frame_lr expects RGB HWC")
    h, w, _ = rgb_hwc.shape
    half = w // 2
    left = rgb_hwc[:, :half, :]
    right = rgb_hwc[:, half : half * 2, :]
    if layout == "lr":
        return left, right
    if layout == "rl":
        return right, left
    raise ValueError(f"Unknown sbs layout: {layout!r}")


def unsplit_boxes_layout(
    boxes_left: Sequence[Box],
    boxes_right: Sequence[Box],
    half_w: int,
    layout: str = "lr",
) -> List[Box]:
    """
    Merge per-half detections back into full-frame coordinates, honoring layout.
    For layout="lr": left boxes stay, right boxes shift +half_w.
    For layout="rl": (because we swapped frames before detection) we invert: first list maps to right half.
    """
    out: List[Box] = []
    if layout == "lr":
        out.extend(boxes_left)
        out.extend([(t, l + half_w, b, r + half_w) for (t, l, b, r) in boxes_right])
        return out
    if layout == "rl":
        out.extend([(t, l + half_w, b, r + half_w) for (t, l, b, r) in boxes_left])
        out.extend(boxes_right)
        return out
    raise ValueError(f"Unknown sbs layout: {layout!r}")


def unsplit_masks_layout(
    masks_left: Optional[Sequence[Optional[torch.Tensor]]],
    masks_right: Optional[Sequence[Optional[torch.Tensor]]],
    full_w: int,
    half_w: int,
    layout: str = "lr",
) -> Optional[List[Optional[torch.Tensor]]]:
    """
    Pad half-width masks back to full width.
    Each mask is HW (or 1xHW); output is full-width HW mask.
    """
    if masks_left is None and masks_right is None:
        return None

    def pad_mask(m: Optional[torch.Tensor], offset_x: int) -> Optional[torch.Tensor]:
        if m is None:
            return None
        if m.ndim == 2:
            hw = m
        elif m.ndim == 3 and m.shape[0] == 1:
            hw = m[0]
        else:
            hw = m
        h = int(hw.shape[-2])
        out = torch.zeros((h, full_w), device=hw.device, dtype=hw.dtype)
        out[:, offset_x : offset_x + half_w].copy_(hw[..., :half_w])
        return out

    out: List[Optional[torch.Tensor]] = []

    if layout == "lr":
        for m in masks_left or []:
            out.append(pad_mask(m, 0))
        for m in masks_right or []:
            out.append(pad_mask(m, half_w))
        return out

    if layout == "rl":
        for m in masks_left or []:
            out.append(pad_mask(m, half_w))
        for m in masks_right or []:
            out.append(pad_mask(m, 0))
        return out

    raise ValueError(f"Unknown sbs layout: {layout!r}")


@dataclass
class FrameStore:
    """In-memory store of full frames that may be modified later by clip compositing."""
    frames_bgr_u8: Dict[int, torch.Tensor]

    def __init__(self) -> None:
        self.frames_bgr_u8 = {}

    def put(self, frame_num: int, frame_bgr_u8: torch.Tensor) -> None:
        self.frames_bgr_u8[int(frame_num)] = frame_bgr_u8

    def pop(self, frame_num: int) -> torch.Tensor:
        return self.frames_bgr_u8.pop(int(frame_num))

    def keys_sorted(self) -> List[int]:
        return sorted(self.frames_bgr_u8.keys())

    def __len__(self) -> int:
        return len(self.frames_bgr_u8)


def drain_store_to_encoder(
    *,
    store: FrameStore,
    safe_before: int,
    encoder,
    device: torch.device,
    sync_before_encode: bool = True,
) -> int:
    """
    Encode and remove all frames with frame_num < safe_before.
    Returns number of frames encoded.
    """
    keys = store.keys_sorted()
    drain_keys = [k for k in keys if k < safe_before]
    if not drain_keys:
        return 0

    # IMPORTANT: sync once per drain, not per-frame.
    # This reduces CPU-side stalls and improves overlap between GPU compute and encode.
    if sync_before_encode:
        sync_device(device)

    count = 0
    for k in drain_keys:
        frm_bgr = store.pop(k)
        encoder.encode_frame(bgr_u8_to_bgra_u8(frm_bgr))
        count += 1
    return count



def nv12_to_rgb_hwc_u8(
    nv12: torch.Tensor,
    *,
    width: int,
    height: int,
    matrix: str = "auto",
    full_range: bool = False,
) -> torch.Tensor:
    """Convert an NV12 frame to RGB HWC uint8.

    nv12 is expected to be uint8 shaped [H*3/2, W] (or flattenable to that).
    Conversion runs on nv12.device (CPU/CUDA/XPU).

    Notes:
      - Default assumes limited-range YUV (typical video). Set full_range=True if needed.
      - matrix="auto" picks bt709 for HD-ish frames, else bt601.
    """
    h = int(height)
    w = int(width)
    if nv12.dtype != torch.uint8:
        raise TypeError(f"nv12 must be uint8, got {nv12.dtype}")

    if nv12.ndim == 1:
        nv12 = nv12.view(h * 3 // 2, w)
    elif nv12.ndim != 2:
        raise ValueError(f"nv12 must be 1D or 2D, got shape={tuple(nv12.shape)}")

    y = nv12[:h, :].to(torch.float32)
    uv = nv12[h:, :].contiguous().view(h // 2, w // 2, 2).to(torch.float32)
    u = uv[..., 0]
    v = uv[..., 1]

    # Nearest upsample to full res
    u = u.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    v = v.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

    if matrix == "auto":
        matrix = "bt709" if (w >= 1280 or h >= 720) else "bt601"

    if full_range:
        c = y
    else:
        # limited-range luma: [16..235] -> scale by 1.164...
        c = (y - 16.0) * 1.164383

    d = u - 128.0
    e = v - 128.0

    if matrix == "bt709":
        r = c + 1.792741 * e
        g = c - 0.213249 * d - 0.532909 * e
        b = c + 2.112402 * d
    else:
        # bt601
        r = c + 1.402000 * e
        g = c - 0.344136 * d - 0.714136 * e
        b = c + 1.772000 * d

    rgb = torch.stack([r, g, b], dim=-1)
    return rgb.round().clamp(0, 255).to(torch.uint8)
