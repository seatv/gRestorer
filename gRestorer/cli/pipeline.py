# gRestorer/cli/pipeline.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch

from gRestorer.video import Decoder, Encoder
from gRestorer.detector.core import Detector as MosaicDetector, Detection
from gRestorer.restorer import NoneRestorer, PseudoRestorer


# ---------------------------
# Timing / sync helpers
# ---------------------------

def _now() -> float:
    return time.perf_counter()


def _sync_device(device: torch.device) -> None:
    """
    Make GPU timings honest by waiting for device work to finish.
    (No-op on CPU.)
    """
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize(device)
    except Exception:
        # Don't let timing helpers crash the pipeline.
        pass


@dataclass
class StageTimes:
    frames: int = 0
    batches: int = 0

    t_total: float = 0.0
    t_decode_batch: float = 0.0
    t_wrap_dlpack: float = 0.0
    t_to_bgr: float = 0.0
    t_to_float: float = 0.0
    t_detect: float = 0.0
    t_restore: float = 0.0
    t_pack: float = 0.0
    t_encode: float = 0.0
    t_flush: float = 0.0

    def fps(self) -> float:
        return (self.frames / self.t_total) if self.t_total > 0 else 0.0

    def _ms_per_frame(self, t: float) -> float:
        return (t * 1000.0 / self.frames) if self.frames else 0.0

    def summary(self) -> str:
        return "\n".join(
            [
                f"[Timing] frames={self.frames} batches={self.batches} total={self.t_total:.3f}s ({self.fps():.2f} fps)",
                f"[Timing] decode_batch: {self.t_decode_batch:.3f}s ({self._ms_per_frame(self.t_decode_batch):.3f} ms/f)",
                f"[Timing] wrap_dlpack:  {self.t_wrap_dlpack:.3f}s ({self._ms_per_frame(self.t_wrap_dlpack):.3f} ms/f)",
                f"[Timing] rgb->bgr:     {self.t_to_bgr:.3f}s ({self._ms_per_frame(self.t_to_bgr):.3f} ms/f)",
                f"[Timing] to_float:     {self.t_to_float:.3f}s ({self._ms_per_frame(self.t_to_float):.3f} ms/f)",
                f"[Timing] detect:       {self.t_detect:.3f}s ({self._ms_per_frame(self.t_detect):.3f} ms/f)",
                f"[Timing] restore:      {self.t_restore:.3f}s ({self._ms_per_frame(self.t_restore):.3f} ms/f)",
                f"[Timing] pack:         {self.t_pack:.3f}s ({self._ms_per_frame(self.t_pack):.3f} ms/f)",
                f"[Timing] encode:       {self.t_encode:.3f}s ({self._ms_per_frame(self.t_encode):.3f} ms/f)",
                f"[Timing] flush:        {self.t_flush:.3f}s ({self._ms_per_frame(self.t_flush):.3f} ms/f)",
            ]
        )


# ---------------------------
# Config helpers (MODULE SCOPE!)
# ---------------------------
def _get(cfg: Any, name: str, default: Any = None) -> Any:
    """
    Best-effort config getter. IMPORTANT: if the key exists but value is None,
    we treat it as "not set" and return default.
    """
    if cfg is None:
        return default

    # dict
    if isinstance(cfg, dict):
        v = cfg.get(name, default)
        return default if v is None else v

    # Config object with .data dict
    data = getattr(cfg, "data", None)
    if isinstance(data, dict) and name in data:
        v = data.get(name, default)
        return default if v is None else v

    # .get(name, default) style
    fn = getattr(cfg, "get", None)
    if callable(fn):
        try:
            v = fn(name, default)
            return default if v is None else v
        except Exception:
            pass

    # attribute fallback
    v = getattr(cfg, name, default)
    return default if v is None else v



def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """
    Nested getter: _cfg_get(cfg, "visualization", "fill_color", default=None)

    Works with:
      - dict nesting
      - utils.config.Config (.data dict nesting)
      - attribute nesting (rare, but supported)
    """
    if cfg is None:
        return default

    cur: Any
    if isinstance(cfg, dict):
        cur = cfg
    else:
        data = getattr(cfg, "data", None)
        cur = data if isinstance(data, dict) else cfg

    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
        if cur is None:
            return default
    return cur


def _parse_bgr(v: Any, allow_none: bool = True) -> Optional[Tuple[int, int, int]]:
    """
    Parse a BGR color from:
      - "B,G,R" string
      - (b,g,r) list/tuple
      - None
    """
    if v is None:
        return None if allow_none else (0, 0, 0)

    if isinstance(v, (list, tuple)) and len(v) == 3:
        return (int(v[0]), int(v[1]), int(v[2]))

    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Expected 'B,G,R', got: {v!r}")
        return (int(parts[0]), int(parts[1]), int(parts[2]))

    raise TypeError(f"Unsupported color type: {type(v)} ({v!r})")


# ---------------------------
# Tensor helpers
# ---------------------------

def _to_u8(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.uint8:
        return x
    if x.dtype.is_floating_point:
        return (x * 255.0).round().clamp(0, 255).to(torch.uint8)
    raise TypeError(f"Unsupported src dtype: {x.dtype}")


def _frame_hw(frame: torch.Tensor) -> Tuple[int, int]:
    """
    Supports:
      - packed: [H,W,3]
      - planar: [3,H,W]
    """
    if frame.ndim != 3:
        raise ValueError(f"Expected 3D frame, got {tuple(frame.shape)}")

    if frame.shape[-1] == 3:   # HWC
        return int(frame.shape[0]), int(frame.shape[1])
    if frame.shape[0] == 3:    # CHW
        return int(frame.shape[1]), int(frame.shape[2])

    raise ValueError(f"Unsupported frame layout: {tuple(frame.shape)}")


def rgb_to_bgr_u8_inplace(dst_bgr: torch.Tensor, src_rgb: torch.Tensor) -> None:
    """
    Convert decoder output RGB -> BGR uint8 into dst_bgr.

    src_rgb layouts supported:
      - packed RGB:  [H,W,3]
      - planar RGBP: [3,H,W]
    src_rgb dtype supported:
      - uint8
      - float in [0,1]

    dst_bgr:
      - packed BGR uint8 [H,W,3]
    """
    if dst_bgr.ndim != 3 or dst_bgr.shape[-1] != 3 or dst_bgr.dtype != torch.uint8:
        raise ValueError(f"dst_bgr must be uint8 [H,W,3], got {dst_bgr.dtype} {tuple(dst_bgr.shape)}")

    if src_rgb.ndim != 3:
        raise ValueError(f"Expected src_rgb 3D, got {tuple(src_rgb.shape)}")

    src_u8 = _to_u8(src_rgb)

    # packed RGB
    if src_u8.shape[-1] == 3:
        if tuple(dst_bgr.shape[:2]) != tuple(src_u8.shape[:2]):
            raise ValueError(f"dst_bgr shape {tuple(dst_bgr.shape)} != src {tuple(src_u8.shape)}")
        dst_bgr[..., 0].copy_(src_u8[..., 2])  # B
        dst_bgr[..., 1].copy_(src_u8[..., 1])  # G
        dst_bgr[..., 2].copy_(src_u8[..., 0])  # R
        return

    # planar RGBP
    if src_u8.shape[0] == 3:
        H, W = int(src_u8.shape[1]), int(src_u8.shape[2])
        if tuple(dst_bgr.shape[:2]) != (H, W):
            raise ValueError(f"dst_bgr shape {tuple(dst_bgr.shape)} != src planar {(3,H,W)}")
        dst_bgr[..., 0].copy_(src_u8[2])  # B
        dst_bgr[..., 1].copy_(src_u8[1])  # G
        dst_bgr[..., 2].copy_(src_u8[0])  # R
        return

    raise ValueError(f"Expected src_rgb [H,W,3] or [3,H,W], got {tuple(src_rgb.shape)}")


def bgr_u8_to_bgr_f32(bgr_u8: torch.Tensor) -> torch.Tensor:
    if bgr_u8.dtype != torch.uint8:
        raise ValueError(f"Expected uint8, got {bgr_u8.dtype}")
    return bgr_u8.to(torch.float32) / 255.0


def pack_bgr_to_bgra_inplace(dst_bgra: torch.Tensor, bgr_u8: torch.Tensor, alpha: int = 255) -> None:
    """
    dst_bgra: [H,W,4] uint8, BGRA byte order in memory
    bgr_u8:   [H,W,3] uint8, BGR
    """
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
    """
    Stage-0 pipeline:
      Decode (GPU) -> RGB/RGBP -> BGR -> Detect (optional) -> Restore (none/pseudo) -> Encode (GPU)
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg

        self.input_path: str = str(_get(cfg, "input_path"))
        self.output_path: str = str(_get(cfg, "output_path"))

        self.gpu_id: int = int(_get(cfg, "gpu_id", 0))
        self.batch_size: int = int(_get(cfg, "batch_size", 8))

        self.max_frames: Optional[int] = _get(cfg, "max_frames", None)
        if self.max_frames is None:
            env_max = os.environ.get("GRESTORER_MAX_FRAMES")
            if env_max:
                try:
                    self.max_frames = int(env_max)
                except Exception:
                    self.max_frames = None

        # Restorer choice
        self.restorer_name: str = str(_get(cfg, "restorer", "none")).lower()

        # Detector config
        det_model = _get(cfg, "det_model_path", None)
        self.det_model_path: Optional[str] = str(det_model) if det_model else None
        self.det_conf: float = float(_get(cfg, "det_conf", 0.25))
        self.det_iou: float = float(_get(cfg, "det_iou", 0.45))
        self.det_imgsz: int = int(_get(cfg, "det_imgsz", 640))
        self.det_fp16: bool = bool(_get(cfg, "det_fp16", True))

        # Visualization options (prefer visualization.*; fall back to top-level)
        self.box_color = _cfg_get(cfg, "visualization", "box_color", default=_get(cfg, "box_color", "0,255,0"))
        self.box_thickness = int(_cfg_get(cfg, "visualization", "box_thickness", default=_get(cfg, "box_thickness", 2)))
        self.fill_color = _cfg_get(cfg, "visualization", "fill_color", default=_get(cfg, "fill_color", None))
        self.fill_opacity = float(_cfg_get(cfg, "visualization", "fill_opacity", default=_get(cfg, "fill_opacity", 0.5)))

        # Encode config
        self.alpha: int = int(_get(cfg, "alpha", 255))
        self.codec: str = str(_get(cfg, "codec", "hevc"))
        self.preset: str = str(_get(cfg, "preset", "P7"))
        self.profile: str = str(_get(cfg, "profile", "main"))
        self.qp: int = int(_get(cfg, "qp", 23))

        self.debug: bool = bool(_get(cfg, "debug", False))

        # Torch device for buffers
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu_id}")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = torch.device(f"xpu:{self.gpu_id}")
        else:
            self.device = torch.device("cpu")

        if self.debug:
            print(f"[Pipeline] device={self.device} batch_size={self.batch_size} restorer={self.restorer_name}")

    def run(self) -> StageTimes:
        t = StageTimes()
        t0 = _now()

        decoder: Optional[Decoder] = None
        encoder: Optional[Encoder] = None
        detector: Optional[MosaicDetector] = None
        restorer: Any = None

        # Reused batch buffers (allocated after we know H/W)
        bgr_u8_bufs: List[torch.Tensor] = []
        bgra_u8_bufs: List[torch.Tensor] = []

        try:
            decoder = Decoder(self.input_path, batch_size=self.batch_size, gpu_id=self.gpu_id)

            width = int(getattr(decoder.metadata, "width", 0))
            height = int(getattr(decoder.metadata, "height", 0))
            fps = float(getattr(decoder.metadata, "fps", 30.0))

            if width <= 0 or height <= 0:
                raise RuntimeError(f"Decoder metadata invalid: {width}x{height}")

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

            # detector only if pseudo
            if self.restorer_name == "pseudo":
                if not self.det_model_path:
                    raise ValueError("Pseudo restorer requires --det-model / det_model_path")
                detector = MosaicDetector(
                    model_path=self.det_model_path,
                    device=str(self.device),
                    imgsz=self.det_imgsz,
                    conf_thres=self.det_conf,
                    iou_thres=self.det_iou,
                    classes=None,
                    fp16=self.det_fp16,
                )

            # restorer init
            if self.restorer_name == "none":
                restorer = NoneRestorer(width=width, height=height, gpu_id=self.gpu_id)
            elif self.restorer_name == "pseudo":
                restorer = PseudoRestorer(
                    width=width,
                    height=height,
                    box_color=_parse_bgr(self.box_color) or (0, 255, 0),
                    box_thickness=self.box_thickness,
                    fill_color=_parse_bgr(self.fill_color, allow_none=True),
                    fill_opacity=self.fill_opacity,
                    gpu_id=self.gpu_id,
                )
            else:
                raise ValueError(f"Unknown restorer: {self.restorer_name}")

            frames_done = 0

            while True:
                if self.max_frames is not None and frames_done >= self.max_frames:
                    break

                tb = _now()
                frames = decoder.read_batch()
                _sync_device(self.device)
                t.t_decode_batch += (_now() - tb)

                if not frames:
                    break

                t.batches += 1

                # dlpack -> torch
                tb = _now()
                rgb_tensors: List[torch.Tensor] = []
                for fr in frames:
                    rgb_tensors.append(torch.from_dlpack(fr))
                _sync_device(self.device)
                t.t_wrap_dlpack += (_now() - tb)

                # allocate buffers if needed
                H, W = _frame_hw(rgb_tensors[0])
                if not bgr_u8_bufs or bgr_u8_bufs[0].shape[0] != H or bgr_u8_bufs[0].shape[1] != W:
                    bgr_u8_bufs = [torch.empty((H, W, 3), dtype=torch.uint8, device=self.device) for _ in range(self.batch_size)]
                    bgra_u8_bufs = [torch.empty((H, W, 4), dtype=torch.uint8, device=self.device) for _ in range(self.batch_size)]

                # RGB/RGBP -> BGR u8
                tb = _now()
                bgr_u8_list: List[torch.Tensor] = []
                for i, rgb in enumerate(rgb_tensors):
                    dst = bgr_u8_bufs[i]
                    rgb_to_bgr_u8_inplace(dst, rgb)
                    bgr_u8_list.append(dst)
                _sync_device(self.device)
                t.t_to_bgr += (_now() - tb)

                # BGR u8 -> float [0,1]
                tb = _now()
                bgr_f32_list = [bgr_u8_to_bgr_f32(x) for x in bgr_u8_list]
                _sync_device(self.device)
                t.t_to_float += (_now() - tb)

                # detect (optional)
                detections: Optional[List[Detection]] = None
                if detector is not None:
                    tb = _now()
                    detections = detector.detect_batch(bgr_f32_list)
                    t.t_detect += (_now() - tb)

                # restore
                tb = _now()
                if self.restorer_name == "none":
                    out_f32_list = restorer.process_batch(bgr_f32_list)
                else:
                    assert detections is not None
                    out_f32_list = restorer.process_batch(bgr_f32_list, detections)
                _sync_device(self.device)
                t.t_restore += (_now() - tb)

                # pack for encoder (BGRA u8)
                tb = _now()
                out_bgra_list: List[torch.Tensor] = []
                for i, out_f32 in enumerate(out_f32_list):
                    bgr_u8 = _to_u8(out_f32)
                    dst_bgra = bgra_u8_bufs[i]
                    pack_bgr_to_bgra_inplace(dst_bgra, bgr_u8, alpha=self.alpha)
                    out_bgra_list.append(dst_bgra)
                _sync_device(self.device)
                t.t_pack += (_now() - tb)

                # encode
                tb = _now()
                for frm in out_bgra_list:
                    encoder.encode_frame(frm)
                _sync_device(self.device)
                t.t_encode += (_now() - tb)

                frames_done += len(frames)
                t.frames += len(frames)

            # flush encoder
            tb = _now()
            if encoder is not None:
                encoder.flush()
            _sync_device(self.device)
            t.t_flush += (_now() - tb)

        finally:
            # Close encoder file handle
            try:
                if encoder is not None:
                    encoder.close()
            except Exception:
                pass

            # Decoder wrapper has no close(); rely on GC
            decoder = None

        t.t_total = _now() - t0
        print(t.summary())
        return t
