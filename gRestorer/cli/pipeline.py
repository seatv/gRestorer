# gRestorer/cli/pipeline.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch

from gRestorer.video import Decoder, Encoder
from gRestorer.restorer import NoneRestorer, PseudoRestorer
from gRestorer.detector.core import Detector, Detection

from gRestorer.core.scene import Box, tlbr_to_xyxy, xyxy_to_tlbr
from gRestorer.core.scene_tracker import SceneTracker, TrackerConfig


# ---------------------------
# Timing / sync helpers
# ---------------------------

def _now() -> float:
    return time.perf_counter()


def _sync_device(device: torch.device) -> None:
    """Synchronize GPU for honest timings (no-op on CPU)."""
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize(device)
    except Exception:
        pass


@dataclass
class StageTimes:
    frames: int = 0
    clips_created: int = 0

    t_total: float = 0.0
    t_decode_batch: float = 0.0
    t_wrap_dlpack: float = 0.0
    t_to_bgr: float = 0.0
    t_to_float: float = 0.0
    t_detect: float = 0.0
    t_track: float = 0.0
    t_clip_build: float = 0.0
    t_restore: float = 0.0
    t_pack: float = 0.0
    t_encode: float = 0.0
    t_flush: float = 0.0

    def _ms_per_frame(self, seconds: float) -> float:
        return (seconds * 1000.0 / self.frames) if self.frames else 0.0

    def pretty(self) -> str:
        return "\n".join(
            [
                f"[Timing] frames:       {self.frames}",
                f"[Timing] clips:        {self.clips_created}",
                f"[Timing] total:        {self.t_total:.3f}s ({self._ms_per_frame(self.t_total):.3f} ms/f)",
                f"[Timing] decode_batch: {self.t_decode_batch:.3f}s ({self._ms_per_frame(self.t_decode_batch):.3f} ms/f)",
                f"[Timing] wrap_dlpack:  {self.t_wrap_dlpack:.3f}s ({self._ms_per_frame(self.t_wrap_dlpack):.3f} ms/f)",
                f"[Timing] rgb->bgr:     {self.t_to_bgr:.3f}s ({self._ms_per_frame(self.t_to_bgr):.3f} ms/f)",
                f"[Timing] to_float:     {self.t_to_float:.3f}s ({self._ms_per_frame(self.t_to_float):.3f} ms/f)",
                f"[Timing] detect:       {self.t_detect:.3f}s ({self._ms_per_frame(self.t_detect):.3f} ms/f)",
                f"[Timing] track:        {self.t_track:.3f}s ({self._ms_per_frame(self.t_track):.3f} ms/f)",
                f"[Timing] clip_build:   {self.t_clip_build:.3f}s ({self._ms_per_frame(self.t_clip_build):.3f} ms/f)",
                f"[Timing] restore:      {self.t_restore:.3f}s ({self._ms_per_frame(self.t_restore):.3f} ms/f)",
                f"[Timing] pack:         {self.t_pack:.3f}s ({self._ms_per_frame(self.t_pack):.3f} ms/f)",
                f"[Timing] encode:       {self.t_encode:.3f}s ({self._ms_per_frame(self.t_encode):.3f} ms/f)",
                f"[Timing] flush:        {self.t_flush:.3f}s ({self._ms_per_frame(self.t_flush):.3f} ms/f)",
            ]
        )


# ---------------------------
# Config helpers
# ---------------------------

def _get(cfg: Any, name: str, default: Any = None) -> Any:
    """Best-effort config getter (dict / Config / attribute)."""
    if cfg is None:
        return default

    if isinstance(cfg, dict):
        v = cfg.get(name, default)
        return default if v is None else v

    data = getattr(cfg, "data", None)
    if isinstance(data, dict) and name in data:
        v = data.get(name, default)
        return default if v is None else v

    fn = getattr(cfg, "get", None)
    if callable(fn):
        try:
            v = fn(name, default)
            return default if v is None else v
        except Exception:
            pass

    v = getattr(cfg, name, default)
    return default if v is None else v


def _get_first(cfg: Any, names: Sequence[str], default: Any = None) -> Any:
    for n in names:
        v = _get(cfg, n, None)
        if v is not None:
            return v
    return default


def _parse_bgr(s: Any, allow_none: bool = False) -> Optional[Tuple[int, int, int]]:
    if s is None:
        return None if allow_none else (0, 255, 0)
    if isinstance(s, (list, tuple)) and len(s) == 3:
        return (int(s[0]), int(s[1]), int(s[2]))
    if isinstance(s, str):
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
    return None if allow_none else (0, 255, 0)


# ---------------------------
# Color / packing helpers
# ---------------------------

def rgb_to_bgr_u8_inplace(dst_bgr_u8: torch.Tensor, src_rgb: torch.Tensor) -> None:
    """dst: [H,W,3] uint8, src: [H,W,3] uint8 OR [3,H,W] uint8 (RGBP)."""
    if dst_bgr_u8.ndim != 3 or dst_bgr_u8.shape[-1] != 3 or dst_bgr_u8.dtype != torch.uint8:
        raise ValueError(f"Expected dst uint8 [H,W,3], got {dst_bgr_u8.dtype} {tuple(dst_bgr_u8.shape)}")

    # Packed RGB
    if src_rgb.ndim == 3 and src_rgb.shape[-1] == 3:
        dst_bgr_u8[..., 0].copy_(src_rgb[..., 2])  # B
        dst_bgr_u8[..., 1].copy_(src_rgb[..., 1])  # G
        dst_bgr_u8[..., 2].copy_(src_rgb[..., 0])  # R
        return

    # Planar RGBP (3,H,W)
    if src_rgb.ndim == 3 and src_rgb.shape[0] == 3:
        dst_bgr_u8[..., 0].copy_(src_rgb[2])  # B
        dst_bgr_u8[..., 1].copy_(src_rgb[1])  # G
        dst_bgr_u8[..., 2].copy_(src_rgb[0])  # R
        return

    raise ValueError(f"Expected src [H,W,3] or [3,H,W], got {tuple(src_rgb.shape)}")


def bgr_u8_to_bgr_f32(bgr_u8: torch.Tensor) -> torch.Tensor:
    if bgr_u8.dtype != torch.uint8:
        raise ValueError(f"Expected uint8, got {bgr_u8.dtype}")
    return bgr_u8.to(torch.float32) / 255.0


def pack_bgr_to_bgra_inplace(dst_bgra: torch.Tensor, bgr_u8: torch.Tensor, alpha: int = 255) -> None:
    """dst_bgra: [H,W,4] uint8 (BGRA), bgr_u8: [H,W,3] uint8."""
    if bgr_u8.ndim != 3 or bgr_u8.shape[-1] != 3 or bgr_u8.dtype != torch.uint8:
        raise ValueError(f"Expected bgr_u8 uint8 [H,W,3], got {bgr_u8.dtype} {tuple(bgr_u8.shape)}")
    if dst_bgra.ndim != 3 or dst_bgra.shape[-1] != 4 or dst_bgra.dtype != torch.uint8:
        raise ValueError(f"Expected dst_bgra uint8 [H,W,4], got {dst_bgra.dtype} {tuple(dst_bgra.shape)}")

    dst_bgra[..., 0].copy_(bgr_u8[..., 0])  # B
    dst_bgra[..., 1].copy_(bgr_u8[..., 1])  # G
    dst_bgra[..., 2].copy_(bgr_u8[..., 2])  # R
    dst_bgra[..., 3].fill_(int(alpha))      # A


# ---------------------------
# Pipeline
# ---------------------------

class Pipeline:
    """Decode (GPU) -> Convert -> Detect -> Track/Clip -> Overlay/Restore -> Encode (GPU)."""

    def __init__(self, cfg: Any):
        self.cfg = cfg

        self.input_path: str = str(_get_first(cfg, ["input_path", "input"], ""))
        self.output_path: str = str(_get_first(cfg, ["output_path", "output"], ""))

        self.gpu_id: int = int(_get(cfg, "gpu_id", 0))
        self.batch_size: int = int(_get(cfg, "batch_size", 8))
        self.max_frames: Optional[int] = _get(cfg, "max_frames", None)
        self.debug: bool = bool(_get(cfg, "debug", False))

        # Optional honest timings (sync after each stage)
        self.profile_sync: bool = bool(_get(cfg, "profile_sync", False))

        # Debug cadence (optional): prints per-frame line every N frames.
        # 0 disables per-frame spam. Per-batch summaries still print when --debug.
        self.debug_every: int = int(_get(cfg, "debug_every", int(os.getenv("GRESTORER_DEBUG_EVERY", "0"))))

        # Restorer
        self.restorer_name: str = str(_get(cfg, "restorer", "none"))

        # Detector config (accept a few common key variants)
        self.det_model_path: Optional[str] = _get_first(cfg, ["det_model_path", "det_model", "det_model_file"], None)
        self.det_conf: float = float(_get(cfg, "det_conf", _get(cfg, "conf_thres", 0.25)))
        self.det_iou: float = float(_get(cfg, "det_iou", _get(cfg, "iou_thres", 0.45)))
        self.det_imgsz: int = int(_get(cfg, "det_imgsz", 640))
        self.det_fp16: bool = bool(_get(cfg, "det_fp16", False))
        self.det_classes: Optional[Sequence[int]] = _get(cfg, "det_classes", None)

        # Scene/Clip tracker (LADA-ish)
        self.max_clip_length: int = int(_get(cfg, "max_clip_length", 30))
        self.clip_size: int = int(_get(cfg, "clip_size", 256))
        self.pad_mode: str = str(_get(cfg, "pad_mode", "reflect"))
        self.border_size: float = float(_get(cfg, "border_size", 0.06))
        self.max_box_expansion_factor: float = float(_get(cfg, "max_box_expansion_factor", 1.0))

        # Viz overrides
        self.box_color: Any = _get(cfg, "box_color", None)
        self.box_thickness: int = int(_get(cfg, "box_thickness", 2))
        self.fill_color: Any = _get(cfg, "fill_color", None)
        self.fill_opacity: float = float(_get(cfg, "fill_opacity", 0.5))

        # Encode overrides
        self.codec: str = str(_get(cfg, "codec", "hevc"))
        self.preset: str = str(_get(cfg, "preset", "P7"))
        self.profile: str = str(_get(cfg, "profile", "main"))
        self.qp: int = int(_get(cfg, "qp", 23))
        self.alpha: int = int(_get(cfg, "alpha", 255))

        # Device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu_id}")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = torch.device(f"xpu:{self.gpu_id}")
        else:
            self.device = torch.device("cpu")

    def _maybe_sync(self) -> None:
        if self.profile_sync:
            _sync_device(self.device)

    def run(self) -> StageTimes:
        t = StageTimes()
        t0 = _now()

        decoder: Optional[Decoder] = None
        encoder: Optional[Encoder] = None
        detector: Optional[Detector] = None
        restorer: Any = None
        tracker: Optional[SceneTracker] = None

        # Reused batch buffers (allocated after we know H/W)
        bgr_u8_bufs: List[torch.Tensor] = []
        bgra_u8_bufs: List[torch.Tensor] = []

        try:
            decoder = Decoder(self.input_path, batch_size=self.batch_size, gpu_id=self.gpu_id)
            width = decoder.metadata.width
            height = decoder.metadata.height
            fps = decoder.metadata.fps

            encoder = Encoder(
                self.output_path,
                width=width,
                height=height,
                fps=fps,
                codec=self.codec,
                preset=self.preset,
                profile=self.profile,
                qp=self.qp,
                gpu_id=self.gpu_id,
            )

            # Restorer init
            if self.restorer_name == "none":
                restorer = NoneRestorer(width=width, height=height)
            elif self.restorer_name == "pseudo":
                restorer = PseudoRestorer(
                    width=width,
                    height=height,
                    box_color=_parse_bgr(self.box_color) or (0, 255, 0),
                    box_thickness=self.box_thickness,
                    fill_color=_parse_bgr(self.fill_color, allow_none=True),
                    fill_opacity=float(self.fill_opacity),
                )
            else:
                raise ValueError(f"Unknown restorer: {self.restorer_name}")

            requires_detection = bool(getattr(restorer, "requires_detection", False)) or (self.restorer_name != "none")

            # Detector + tracker init only if needed
            if requires_detection:
                if not self.det_model_path:
                    raise ValueError("This restorer requires detection but det_model_path/det_model was not provided")

                detector = Detector(
                    model_path=self.det_model_path,
                    device=str(self.device),
                    imgsz=self.det_imgsz,
                    conf_thres=self.det_conf,
                    iou_thres=self.det_iou,
                    classes=self.det_classes,
                    fp16=self.det_fp16,
                )

                tracker = SceneTracker(
                    TrackerConfig(
                        clip_size=self.clip_size,
                        max_clip_length=self.max_clip_length,
                        pad_mode=self.pad_mode,
                        border_size=self.border_size,
                        max_box_expansion_factor=self.max_box_expansion_factor,
                    )
                )

            frames_done = 0

            while True:
                if self.max_frames is not None and frames_done >= int(self.max_frames):
                    break

                # Per-batch debug counters
                batch_total_boxes = 0
                batch_frames_with_boxes = 0
                batch_total_overlay = 0
                batch_new_clips = 0

                tb = _now()
                frames = decoder.read_batch()
                self._maybe_sync()
                t.t_decode_batch += (_now() - tb)

                if not frames:
                    break

                # Wrap dlpack -> torch
                tb = _now()
                rgb_list: List[torch.Tensor] = [torch.from_dlpack(f) for f in frames]
                self._maybe_sync()
                t.t_wrap_dlpack += (_now() - tb)

                # Allocate buffers for this batch size
                if len(bgr_u8_bufs) != len(rgb_list):
                    bgr_u8_bufs = [torch.empty((height, width, 3), dtype=torch.uint8, device=self.device) for _ in rgb_list]
                    bgra_u8_bufs = [torch.empty((height, width, 4), dtype=torch.uint8, device=self.device) for _ in rgb_list]

                # RGB -> BGR (u8)
                tb = _now()
                for i, rgb in enumerate(rgb_list):
                    rgb_to_bgr_u8_inplace(bgr_u8_bufs[i], rgb)
                self._maybe_sync()
                t.t_to_bgr += (_now() - tb)

                # To float for detector/restorer
                tb = _now()
                bgr_f32_list = [bgr_u8_to_bgr_f32(x) for x in bgr_u8_bufs]
                self._maybe_sync()
                t.t_to_float += (_now() - tb)

                # Detect (optional)
                detections: Optional[List[Detection]] = None
                if detector is not None:
                    tb = _now()
                    detections = detector.detect_batch(bgr_f32_list)
                    self._maybe_sync()
                    t.t_detect += (_now() - tb)

                # Track scenes + build clips + overlay detections
                dets_for_restorer: Optional[Sequence[Detection]] = detections

                if tracker is not None:
                    assert detections is not None
                    overlay_dets: List[Detection] = []
                    frame_base = frames_done

                    for i, det in enumerate(detections):
                        frame_num = frame_base + i

                        # Convert detector boxes (xyxy float) -> tlbr int boxes.
                        roi_boxes: List[Box] = []
                        n_det = 0
                        if det.boxes is not None and det.boxes.numel() > 0:
                            n_det = int(det.boxes.shape[0])
                            if self.debug:
                                batch_total_boxes += n_det
                                batch_frames_with_boxes += 1
                            # Boxes are in original coords already.
                            for (x1, y1, x2, y2) in det.boxes.tolist():
                                roi_boxes.append(
                                    xyxy_to_tlbr(
                                        (float(x1), float(y1), float(x2), float(y2)),
                                        height,
                                        width,
                                    )
                                )

                        tb2 = _now()
                        overlay_roi_boxes = tracker.ingest_frame(frame_num, bgr_u8_bufs[i], roi_boxes)
                        self._maybe_sync()
                        t.t_track += (_now() - tb2)

                        tb3 = _now()
                        new_clips = tracker.flush_completed(frame_num, eof=False)
                        self._maybe_sync()
                        t.t_clip_build += (_now() - tb3)
                        t.clips_created += len(new_clips)

                        if self.debug:
                            batch_new_clips += len(new_clips)

                        n_overlay = len(overlay_roi_boxes) if overlay_roi_boxes else 0
                        if self.debug:
                            batch_total_overlay += n_overlay

                        # Overlay uses the *tracked* ROI boxes (unioned per scene), converted back to xyxy.
                        if overlay_roi_boxes:
                            xyxy_boxes = [tlbr_to_xyxy(b) for b in overlay_roi_boxes]
                            boxes_t = torch.as_tensor(xyxy_boxes, device=self.device, dtype=torch.float32)
                        else:
                            boxes_t = torch.empty((0, 4), device=self.device, dtype=torch.float32)

                        overlay_dets.append(Detection(boxes=boxes_t, scores=None, classes=None, masks=None))

                        # Optional per-frame debug line
                        if self.debug and self.debug_every > 0 and (frame_num % self.debug_every == 0):
                            print(
                                f"[Dbg] f={frame_num:6d} det={n_det:2d} overlay={n_overlay:2d} "
                                f"active_scenes={tracker.active_scenes:2d} new_clips={len(new_clips):2d}"
                            )

                    dets_for_restorer = overlay_dets

                # Restore / overlay
                tb = _now()
                out_f32_list = restorer.process_batch(bgr_f32_list, dets_for_restorer)
                self._maybe_sync()
                t.t_restore += (_now() - tb)

                # Pack for encoder (BGRA u8)
                tb = _now()
                for i, out_f32 in enumerate(out_f32_list):
                    out_u8 = (out_f32.clamp(0, 1) * 255.0).to(torch.uint8)
                    pack_bgr_to_bgra_inplace(bgra_u8_bufs[i], out_u8, alpha=self.alpha)
                self._maybe_sync()
                t.t_pack += (_now() - tb)

                # Encode
                tb = _now()
                encoder.encode_frames(bgra_u8_bufs)
                self._maybe_sync()
                t.t_encode += (_now() - tb)

                # Bookkeeping
                frames_done += len(frames)
                t.frames = frames_done

                # Per-batch debug summary (low spam)
                if self.debug and tracker is not None:
                    f0 = frame_base
                    f1 = frame_base + len(frames) - 1
                    print(
                        f"[Dbg] batch {f0}-{f1} det_frames={batch_frames_with_boxes}/{len(frames)} "
                        f"boxes={batch_total_boxes} overlay_sum={batch_total_overlay} "
                        f"new_clips={batch_new_clips} active_scenes={tracker.active_scenes}"
                    )

            # EOF flush for tracker (like LADA calls with next frame_num)
            if tracker is not None:
                tb = _now()
                new_clips = tracker.flush_eof(frames_done)
                self._maybe_sync()
                t.t_clip_build += (_now() - tb)
                t.clips_created += len(new_clips)
                if self.debug:
                    print(f"[Dbg] EOF flush: new_clips={len(new_clips)} active_scenes={tracker.active_scenes}")

            tb = _now()
            encoder.flush()
            self._maybe_sync()
            t.t_flush += (_now() - tb)

        finally:
            try:
                if encoder is not None:
                    encoder.close()
            except Exception:
                pass
            try:
                if decoder is not None:
                    decoder.close()
            except Exception:
                pass

        t.t_total = _now() - t0
        if self.debug:
            print(t.pretty())
        return t
