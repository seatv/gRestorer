from __future__ import annotations

import os
from pathlib import Path
import torch.nn.functional as F

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict

import torch

from gRestorer.core.scene import Box, Clip, Scene, crop_box_to_target_v3


@dataclass
class TrackerConfig:
    clip_size: int = 256
    max_clip_length: int = 30
    pad_mode: str = "reflect"
    border_size: float = 0.06
    max_box_expansion_factor: float = 1.0
    debug: bool = False

    # If True and the detector provides per-pixel masks, we'll use them for
    # clip masks (more LADA-faithful compositing). If False, clip masks will
    # be simple rectangle-box masks.
    use_seg_masks: bool = True

    # NOTE: LADA unions multiple detections for the same scene in the same frame.
    # This can occasionally create a single larger ROI if two boxes briefly overlap.
    # TODO: consider adding a *debug-only* metric for "weak overlap" merges so we can
    #       inspect detector jitter. We should not change LADA semantics by default.


@dataclass
class TrackerStepResult:
    overlay_boxes: List[Box]
    new_clips: List[Clip]
    active_scenes: int
    t_track: float
    t_clip_build: float


def _union_box(a: Box, b: Box) -> Box:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )

import math
import torch
import torch.nn.functional as F

def _env_int(name: str, default: int) -> int:
    s = os.environ.get(name, "")
    s = s.strip().strip('"').strip("'").strip()
    if not s:
        return default
    try:
        return int(s)
    except Exception:
        return default

def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def _dominant_period(sig: torch.Tensor, min_px: int, max_px: int) -> int:
    """
    sig: 1D float tensor on CPU. Returns dominant period (lag) in [min_px, max_px], or 0 if unknown.
    """
    sig = sig.float()
    sig = sig - sig.mean()
    L = int(sig.numel())
    if L < (max_px * 2):
        return 0

    n = _next_pow2(L * 2)
    Fsig = torch.fft.rfft(sig, n=n)
    ac = torch.fft.irfft(Fsig * torch.conj(Fsig), n=n).real
    ac = ac[: max_px + 1]

    k0 = int(min_px)
    k1 = int(min(max_px, ac.numel() - 1))
    if k1 <= k0:
        return 0

    # best peak
    rel = ac[k0 : k1 + 1]
    k = int(torch.argmax(rel).item() + k0)

    # small “fundamental” heuristic: prefer k/2 if nearly as good
    if k % 2 == 0 and (k // 2) >= k0:
        if ac[k // 2] > 0.88 * ac[k]:
            k = k // 2

    return k

def _estimate_tile_px_from_crop_cpu(bgr_u8_cpu: torch.Tensor, mask_u8_cpu: torch.Tensor,
                                   min_px: int, max_px: int) -> int:
    """
    bgr_u8_cpu: (H,W,3) uint8 on CPU
    mask_u8_cpu: (H,W) uint8 on CPU, 255 inside mosaic support
    """
    # Luma (rough Rec.601)
    b = bgr_u8_cpu[..., 0].float()
    g = bgr_u8_cpu[..., 1].float()
    r = bgr_u8_cpu[..., 2].float()
    y = 0.114 * b + 0.587 * g + 0.299 * r

    m = (mask_u8_cpu > 0).float()

    # Edge energy along X/Y, weighted by mask overlap
    dx = (y[:, 1:] - y[:, :-1]).abs() * (m[:, 1:] * m[:, :-1])
    dy = (y[1:, :] - y[:-1, :]).abs() * (m[1:, :] * m[:-1, :])

    gx = dx.sum(dim=0)  # (W-1,)
    gy = dy.sum(dim=1)  # (H-1,)

    # light smoothing to stabilize peaks
    if gx.numel() >= 9:
        gx = F.avg_pool1d(gx.view(1, 1, -1), kernel_size=9, stride=1, padding=4).view(-1)
    if gy.numel() >= 9:
        gy = F.avg_pool1d(gy.view(1, 1, -1), kernel_size=9, stride=1, padding=4).view(-1)

    px_x = _dominant_period(gx, min_px=min_px, max_px=max_px)
    px_y = _dominant_period(gy, min_px=min_px, max_px=max_px)

    cand = [p for p in (px_x, px_y) if p > 0]
    if not cand:
        return 0
    cand.sort()
    return int(cand[len(cand)//2])  # median


class SceneTracker:
    """Track per-frame detections into LADA-style Scenes, then emit Clips."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self._scenes: List[Scene] = []
        self._scene_counter: int = 0
        self._clip_counter: int = 0
        # Auto-estimated mosaic tile size (pixels), cached once per run.
        self._tile_px_cached: Optional[int] = None


    def reset(self) -> None:
        self._scenes.clear()
        self._scene_counter = 0
        self._clip_counter = 0

    @property
    def scenes_active(self) -> int:
        return len(self._scenes)

    # Back-compat alias used by the CLI pipeline.
    @property
    def active_scenes(self) -> int:
        return self.scenes_active

    def min_active_start(self) -> Optional[int]:
        """Earliest start-frame among active scenes (None if no active scenes)."""
        if not self._scenes:
            return None
        return min(s.frame_start for s in self._scenes)

    def clips_emitted(self) -> int:
        return self._clip_counter

    def _new_scene(self, frame_num: int) -> Scene:
        s = Scene(id=self._scene_counter, start_frame=frame_num)
        self._scene_counter += 1
        return s

    def _compute_crop(
            self,
            frame_num: int,
            frame_bgr_u8: torch.Tensor,
            roi_box: Box,
            roi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Box, torch.Tensor, torch.Tensor]:
        """Compute LADA crop_to_box_v3 crop box and slice crop from the frame.

        We create a per-crop mask on the frame device.

        - If roi_mask is provided (per-pixel segmentation) and use_seg_masks=True, we use it
          as the clip mask (cropped to crop_box). If roi_mask is on CPU at full resolution,
          we slice *only the crop region* on CPU and transfer just that to the frame device
          (avoids copying full HxW masks each frame).
        - Otherwise we fall back to a rectangle mask derived from roi_box.
        """
        h, w = int(frame_bgr_u8.shape[0]), int(frame_bgr_u8.shape[1])
        crop_box, _scale = crop_box_to_target_v3(
            roi_box,
            img_h=h,
            img_w=w,
            target_hw=(self.cfg.clip_size, self.cfg.clip_size),
            max_box_expansion_factor=self.cfg.max_box_expansion_factor,
            border_size=float(self.cfg.border_size),
        )
        t, l, b, r = crop_box
        crop_img = frame_bgr_u8[t: b + 1, l: r + 1, :].clone()

        # Mask generation (seam-sensitive)
        # Mask generation (seam-sensitive)
        crop_h = int(b - t + 1)
        crop_w = int(r - l + 1)

        # Rectangle mask: used as fallback and as a limiter for seg masks
        rect_mask = torch.zeros((crop_h, crop_w), device=frame_bgr_u8.device, dtype=torch.uint8)

        rt, rl, rb, rr = roi_box
        it = max(t, rt)
        il = max(l, rl)
        ib = min(b, rb)
        ir = min(r, rr)
        if ib >= it and ir >= il:
            rect_mask[it - t: ib - t + 1, il - l: ir - l + 1] = 255

        crop_mask_out = rect_mask  # default fallback

        # If we have a seg mask, USE IT (limited to rect), don't OR with rect.
        if roi_mask is not None and self.cfg.use_seg_masks:
            try:
                def _mask_u8(m: torch.Tensor) -> torch.Tensor:
                    if m.dtype == torch.bool:
                        return m.to(dtype=torch.uint8) * 255
                    if m.is_floating_point():
                        return torch.where(m > 0.5, 255, 0).to(dtype=torch.uint8)
                    return m.to(dtype=torch.uint8)

                seg_crop: Optional[torch.Tensor] = None
                if roi_mask.device == frame_bgr_u8.device:
                    if roi_mask.shape == (h, w):
                        seg_crop = roi_mask[t: b + 1, l: r + 1]
                    elif roi_mask.shape == (crop_h, crop_w):
                        seg_crop = roi_mask
                elif roi_mask.device.type == "cpu":
                    if roi_mask.shape == (h, w):
                        seg_crop = roi_mask[t: b + 1, l: r + 1].to(device=frame_bgr_u8.device)
                    elif roi_mask.shape == (crop_h, crop_w):
                        seg_crop = roi_mask.to(device=frame_bgr_u8.device)

                if seg_crop is not None:
                    seg_u8 = _mask_u8(seg_crop)

                    # Auto-estimate once (cached) if GR_AUTO_TILE=1, else fallback to GR_MOSAIC_TILE_PX
                    pad_px = self._get_tile_pad_px(crop_img, seg_u8)
                    pad_px = max(0, int(pad_px))

                    # Dilate segmentation so tiny misses (like 1 block row) get covered
                    seg_u8 = _dilate_u8_mask(seg_u8, pad_px)

                    # Also dilate the ROI rectangle so we don't clip off the newly-covered strip
                    rect_support = _dilate_u8_mask(rect_mask, pad_px)

                    # Final: seg (grown) but bounded to expanded ROI support
                    crop_mask_out = torch.where(rect_support > 0, seg_u8, torch.zeros_like(seg_u8))

            except Exception:
                # best-effort fallback
                crop_mask_out = rect_mask

        return crop_box, crop_img, crop_mask_out

    def _get_tile_pad_px(self, crop_bgr_u8: torch.Tensor, base_mask_u8: torch.Tensor) -> int:
        """
        Returns pad in pixels. If GR_AUTO_TILE=1, estimate once (cached).
        Falls back to GR_MOSAIC_TILE_PX (or 40 default).
        """
        fallback = _env_int("GR_MOSAIC_TILE_PX", 40)
        if _env_int("GR_AUTO_TILE", 0) != 1:
            return fallback

        if self._tile_px_cached is not None:
            return int(self._tile_px_cached)

        # Only estimate when mask area is meaningful
        mask01 = (base_mask_u8 > 0)
        area = int(mask01.sum().item())
        if area < _env_int("GR_AUTO_TILE_MIN_AREA", 2048):
            return fallback

        # Downsample to reduce work + transfer (estimate only needs rough periodicity)
        max_side = _env_int("GR_AUTO_TILE_MAX_SIDE", 512)
        h, w = int(crop_bgr_u8.shape[0]), int(crop_bgr_u8.shape[1])
        scale = min(1.0, float(max_side) / float(max(h, w)))
        if scale < 1.0:
            oh = max(8, int(h * scale))
            ow = max(8, int(w * scale))
            # resize image (bilinear) and mask (nearest)
            img = crop_bgr_u8.permute(2, 0, 1).unsqueeze(0).float()  # 1x3xHxW
            img = F.interpolate(img, size=(oh, ow), mode="bilinear", align_corners=False)
            img_u8 = img.round().clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).contiguous()

            m = base_mask_u8.unsqueeze(0).unsqueeze(0).float()
            m = F.interpolate(m, size=(oh, ow), mode="nearest")
            m_u8 = (m[0, 0] > 0).to(torch.uint8) * 255
        else:
            img_u8 = crop_bgr_u8
            m_u8 = base_mask_u8

        # move to CPU once (cached result)
        img_cpu = img_u8.detach().cpu()
        m_cpu = m_u8.detach().cpu()

        min_px = _env_int("GR_AUTO_TILE_MIN_PX", 6)
        max_px = _env_int("GR_AUTO_TILE_MAX_PX", 96)

        est = _estimate_tile_px_from_crop_cpu(img_cpu, m_cpu, min_px=min_px, max_px=max_px)
        if est <= 0:
            return fallback

        # cache + log once
        self._tile_px_cached = int(est)
        print(f"[Tracker] Auto tile size estimated: {self._tile_px_cached}px (cached)")
        return int(self._tile_px_cached)

    def step_frame(
            self,
            frame_num: int,
            frame_bgr_u8: torch.Tensor,
            roi_boxes: Sequence[Box],
            roi_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> TrackerStepResult:
        """Ingest one frame's detections, update scenes, and flush completed scenes."""
        if roi_masks is not None and len(roi_masks) != len(roi_boxes):
            raise ValueError("roi_masks length must match roi_boxes length")

        t0 = time.perf_counter()

        # Update scenes with detections.
        for i, box in enumerate(roi_boxes):
            mask = None
            if self.cfg.use_seg_masks and roi_masks is not None:
                mask = roi_masks[i]

            matched: Optional[Scene] = None
            for s in self._scenes:
                if s.belongs(box):
                    matched = s
                    break

            if matched is None:
                matched = self._new_scene(frame_num)
                self._scenes.append(matched)

            if matched.frame_end == frame_num:
                # Same-frame merge: union ROI and recompute crop from union.
                union_roi = _union_box(matched.roi_boxes[-1], box)

                # Re-crop from union ROI. IMPORTANT: pass current detection mask (if any),
                # then union it with the previous crop-mask by mapping old->new crop coords.
                crop_box, crop_img, cur_mask = self._compute_crop(frame_num, frame_bgr_u8, union_roi, mask)

                prev_mask = matched.masks[-1] if matched.masks else None
                prev_crop_box = matched.crop_boxes[-1] if matched.crop_boxes else None

                if prev_mask is not None and prev_crop_box is not None:
                    nt, nl, nb, nr = crop_box
                    new_h = int(nb - nt + 1)
                    new_w = int(nr - nl + 1)
                    merged = torch.zeros((new_h, new_w), device=frame_bgr_u8.device, dtype=torch.uint8)

                    ot, ol, ob, or_ = prev_crop_box
                    it = max(nt, ot)
                    il = max(nl, ol)
                    ib = min(nb, ob)
                    ir = min(nr, or_)
                    if ib >= it and ir >= il:
                        h_int = int(ib - it + 1)
                        w_int = int(ir - il + 1)
                        oy0 = int(it - ot)
                        ox0 = int(il - ol)
                        ny0 = int(it - nt)
                        nx0 = int(il - nl)
                        merged[ny0 : ny0 + h_int, nx0 : nx0 + w_int] = torch.maximum(
                            merged[ny0 : ny0 + h_int, nx0 : nx0 + w_int],
                            prev_mask[oy0 : oy0 + h_int, ox0 : ox0 + w_int].to(dtype=torch.uint8),
                        )

                    crop_mask = torch.maximum(merged, cur_mask.to(dtype=torch.uint8))
                else:
                    crop_mask = cur_mask

                matched.merge_same_frame(
                    roi_box=union_roi,
                    crop_box=crop_box,
                    crop_img=crop_img,
                    crop_mask=crop_mask,
                )
                dump_one = os.environ.get("GR_DUMP_FULLMASK_FRAME", "").strip().strip('"').strip("'").strip()
                if dump_one.isdigit() and int(dump_one) == int(frame_num):
                    _dump_full_frame_mask_overlay(frame_bgr_u8, crop_box, cur_mask, frame_num, suffix=f"roi{i}_merge")

            else:
                crop_box, crop_img, crop_mask = self._compute_crop(frame_num, frame_bgr_u8, box, mask)
                dump_one = os.environ.get("GR_DUMP_FULLMASK_FRAME", "").strip().strip('"').strip("'").strip()
                if dump_one.isdigit() and int(dump_one) == int(frame_num):
                    _dump_full_frame_mask_overlay(frame_bgr_u8, crop_box, crop_mask, frame_num, suffix=f"roi{i}")

                matched.add_frame(
                    frame_num=frame_num,
                    roi_box=box,
                    crop_box=crop_box,
                    crop_img=crop_img,
                    crop_mask=crop_mask,
                )

        # Any scenes not updated in this frame are completed (gap).
        completed_gap: List[Scene] = [s for s in self._scenes if s.frame_end < frame_num]

        # And any scenes that reached max length are completed.
        completed_maxlen: List[Scene] = [s for s in self._scenes if len(s) >= self.cfg.max_clip_length]

        # LADA-faithful rule: when a scene completes, also complete any scenes that started earlier.
        # This guarantees deterministic clip ordering and prevents early-started scenes from
        # blocking drain forever when later scenes end first.
        completed_scenes: List[Scene] = []
        reason_by_scene_id: Dict[int, str] = {}
        for s in completed_gap:
            reason_by_scene_id[s.id] = "gap"
        for s in completed_maxlen:
            reason_by_scene_id.setdefault(s.id, "max_len")

        for current_scene in list(self._scenes):
            is_done = (current_scene.frame_end < frame_num) or (len(current_scene) >= self.cfg.max_clip_length)
            if not is_done:
                continue

            if current_scene not in completed_scenes:
                completed_scenes.append(current_scene)

            for other_scene in self._scenes:
                if other_scene is current_scene:
                    continue
                if other_scene.frame_start < current_scene.frame_start and other_scene not in completed_scenes:
                    completed_scenes.append(other_scene)
                    reason_by_scene_id.setdefault(other_scene.id, "cascade")

        # LADA: complete in ascending start-frame order.
        completed_unique: List[Scene] = sorted(completed_scenes, key=lambda s: s.frame_start)

        # Remove completed from active list.
        if completed_unique:
            completed_ids = {s.id for s in completed_unique}
            self._scenes = [s for s in self._scenes if s.id not in completed_ids]

        t1 = time.perf_counter()

        # reason_by_scene_id is built alongside the LADA-style completion selection above.

        new_clips: List[Clip] = []
        t_clip_build = 0.0
        for s in completed_unique:
            tb0 = time.perf_counter()
            new_clips.append(
                s.to_clip(
                    clip_id=self._clip_counter,
                    clip_size=self.cfg.clip_size,
                    pad_mode=self.cfg.pad_mode,
                )
            )

            if self.cfg.debug:
                why = reason_by_scene_id.get(s.id, "?")
                roi_xyxy = s.roi_boxes[-1] if s.roi_boxes else (0, 0, 0, 0)
                print(
                    f"[Clip] clip_id={self._clip_counter:5d} scene_id={s.id:4d} why={why:10s} "
                    f"frames={s.frame_start:5d}-{s.frame_end:5d} len={len(s):3d} roi_xyxy={roi_xyxy}"
                )
            self._clip_counter += 1
            tb1 = time.perf_counter()
            t_clip_build += (tb1 - tb0)

        t2 = time.perf_counter()

        overlay_boxes: List[Box] = []
        # For debug/visualization we want the boxes corresponding to *this frame's* overlays.
        # Crucially, this must include scenes that *complete on this frame* (e.g. max_len),
        # otherwise you see missing overlays at f=29,59,... even though detections exist.
        #
        # NOTE: At this point, any scenes that completed have already been removed from
        # self._scenes, so we must also include completed_unique.
        for s in self._scenes:
            if s.frame_end == frame_num and s.roi_boxes:
                overlay_boxes.append(s.roi_boxes[-1])
        for s in completed_unique:
            if s.frame_end == frame_num and s.roi_boxes:
                overlay_boxes.append(s.roi_boxes[-1])

        return TrackerStepResult(
            overlay_boxes=overlay_boxes,
            new_clips=new_clips,
            active_scenes=len(self._scenes),
            t_track=(t1 - t0),
            t_clip_build=t_clip_build,
        )

    def flush_eof(self, *_: object) -> List[Clip]:
        """Flush all remaining scenes at end-of-file."""
        clips: List[Clip] = []
        for s in self._scenes:
            if len(s) == 0:
                continue
            clips.append(
                s.to_clip(
                    clip_id=self._clip_counter,
                    clip_size=self.cfg.clip_size,
                    pad_mode=self.cfg.pad_mode,
                )
            )
            self._clip_counter += 1

        self._scenes.clear()
        return clips

    # --- Compatibility helpers (older pipeline revisions) ---
    def ingest_frame(self, frame_num: int, frame_bgr_u8: torch.Tensor, roi_boxes: Sequence[Box]) -> List[Box]:
        """Compat: older pipelines expect an ingest_frame() that returns overlay boxes."""
        res = self.step_frame(frame_num, frame_bgr_u8, roi_boxes)
        return res.overlay_boxes

    def flush_completed(self, current_frame: int) -> List[Clip]:
        """Compat: older pipelines called flush_completed() explicitly.

        In the current implementation, completion is handled inside step_frame().
        This is therefore a no-op and returns an empty list.
        """
        return []


__all__ = [
    "TrackerConfig",
    "TrackerStepResult",
    "SceneTracker",
]


def _dilate_u8_mask(seg_u8: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return seg_u8
    m = (seg_u8 > 0).to(torch.float32)[None, None, :, :]
    m = F.max_pool2d(m, kernel_size=2 * pad + 1, stride=1, padding=pad)
    return (m[0, 0] > 0).to(torch.uint8) * 255


def _dump_full_frame_mask_overlay(
    frame_bgr_u8: torch.Tensor,
    crop_box: Box,
    crop_mask_u8: torch.Tensor,
    frame_num: int,
    suffix: str = "",
) -> None:
    raw_dir = os.environ.get("GR_DUMP_FULLMASK_DIR", r"D:\Results\mask_overlay_dump")
    raw_dir = raw_dir.strip().strip('"').strip("'").strip()
    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suf = f"_{suffix}" if suffix else ""
    mask_path = out_dir / f"dbg_f{frame_num:06d}_fullmask{suf}.png"
    ov_path   = out_dir / f"dbg_f{frame_num:06d}_overlay{suf}.png"

    h, w = int(frame_bgr_u8.shape[0]), int(frame_bgr_u8.shape[1])
    t, l, b, r = crop_box

    # Full-frame mask (same device as frame)
    full_mask = torch.zeros((h, w), dtype=torch.uint8, device=frame_bgr_u8.device)
    full_mask[t : b + 1, l : r + 1] = crop_mask_u8.to(dtype=torch.uint8)

    # Overlay (tint green where mask>0). frame is BGR uint8.
    frm_f = frame_bgr_u8.to(torch.float32)
    m3 = (full_mask > 0).unsqueeze(-1)

    # green in BGR
    color = torch.tensor([0.0, 255.0, 0.0], device=frame_bgr_u8.device).view(1, 1, 3)
    a = 0.5
    ov_f = torch.where(m3, frm_f * (1.0 - a) + color * a, frm_f)
    ov_u8 = ov_f.round().clamp(0, 255).to(torch.uint8)

    # Move to CPU for writing
    if full_mask.device.type != "cpu":
        full_mask_cpu = full_mask.detach().cpu()
        ov_cpu = ov_u8.detach().cpu()
    else:
        full_mask_cpu = full_mask.detach()
        ov_cpu = ov_u8.detach()

    # Convert BGR->RGB for file viewing
    ov_rgb = ov_cpu[:, :, [2, 1, 0]].numpy()

    try:
        import imageio.v2 as imageio
        imageio.imwrite(str(mask_path), full_mask_cpu.numpy())
        imageio.imwrite(str(ov_path), ov_rgb)
    except Exception:
        from PIL import Image
        Image.fromarray(full_mask_cpu.numpy(), mode="L").save(str(mask_path))
        Image.fromarray(ov_rgb, mode="RGB").save(str(ov_path))


