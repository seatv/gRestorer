# gRestorer/cli/pipeline.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch

from gRestorer.video import Decoder, Encoder
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

    def _ms_per_frame(self, seconds: float) -> float:
        return (seconds * 1000.0 / self.frames) if self.frames else 0.0

    def pretty(self) -> str:
        return "\n".join(
            [
                f"[Timing] frames:       {self.frames}",
                f"[Timing] total:        {self.t_total:.3f}s ({self._ms_per_frame(self.t_total):.3f} ms/f)",
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


def _parse_bgr(s: Any, allow_none: bool = False) -> Optional[Tuple[int, int, int]]:
    if s is None:
        return None if allow_none else (0, 255, 0)
    if isinstance(s, (list, tuple)) and len(s) == 3:
        return (int(s[0]), int(s[1]), int(s[2]))
    if isinstance(s, str):
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
    if allow_none:
        return None
    return (0, 255, 0)


# ---------------------------
# Color / packing helpers
# ---------------------------

def rgb_to_bgr_u8_inplace(dst_bgr_u8: torch.Tensor, src_rgb: torch.Tensor) -> None:
    """
    dst_bgr_u8: [H,W,3] uint8
    src_rgb:    [H,W,3] uint8 OR [3,H,W] uint8 (RGBP)
    """
    if dst_bgr_u8.ndim != 3 or dst_bgr_u8.shape[-1] != 3 or dst_bgr_u8.dtype != torch.uint8:
        raise ValueError(f"Expected dst_bgr_u8 uint8 [H,W,3], got {dst_bgr_u8.dtype} {tuple(dst_bgr_u8.shape)}")

    if src_rgb.ndim == 3 and src_rgb.shape[-1] == 3:
        # packed HWC
        dst_bgr_u8[..., 0].copy_(src_rgb[..., 2])  # B
        dst_bgr_u8[..., 1].copy_(src_rgb[..., 1])  # G
        dst_bgr_u8[..., 2].copy_(src_rgb[..., 0])  # R
        return

    if src_rgb.ndim == 3 and src_rgb.shape[0] == 3:
        # planar CHW (RGBP)
        dst_bgr_u8[..., 0].copy_(src_rgb[2])  # B
        dst_bgr_u8[..., 1].copy_(src_rgb[1])  # G
        dst_bgr_u8[..., 2].copy_(src_rgb[0])  # R
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
      Decode (GPU) -> RGBP -> BGR -> Detect (optional) -> Restore (none/pseudo) -> Encode (GPU)
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg

        self.input_path: str = str(_get(cfg, "input_path"))
        self.output_path: str = str(_get(cfg, "output_path"))

        self.gpu_id: int = int(_get(cfg, "gpu_id", 0))
        self.batch_size: int = int(_get(cfg, "batch_size", 8))

        self.max_frames: Optional[int] = _get(cfg, "max_frames", None)

        self.debug: bool = bool(_get(cfg, "debug", False))

        # Timing control: sync slows throughput but makes stage timings honest.
        self.profile_sync: bool = bool(_get(cfg, "profile_sync", False))

        # Restorer
        self.restorer_name: str = str(_get(cfg, "restorer", "none"))

        # Detector config
        self.det_model_path: Optional[str] = _get(cfg, "det_model_path", None)
        self.det_conf: float = float(_get(cfg, "det_conf", 0.25))
        self.det_iou: float = float(_get(cfg, "det_iou", 0.45))
        self.det_imgsz: int = int(_get(cfg, "det_imgsz", 640))
        self.det_fp16: bool = bool(_get(cfg, "det_fp16", False))

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
        detector: Any = None
        restorer: Any = None

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

            # detector init ONLY if needed
            requires_detection = (self.restorer_name != "none")
            if requires_detection:
                if not self.det_model_path:
                    raise ValueError("restorer requires detection but --det-model was not provided")

                try:
                    from gRestorer.detector.core import Detector as MosaicDetector
                except Exception as e:
                    raise RuntimeError(
                        "Detector initialization failed. Ensure ultralytics is installed and the model path is valid.")(
                        e)

                detector = MosaicDetector(
                    model_path=self.det_model_path,
                    imgsz=self.det_imgsz,
                    conf=self.det_conf,
                    iou=self.det_iou,
                    fp16=self.det_fp16,
                )

            # restorer init
            if self.restorer_name == "none":
                restorer = NoneRestorer(width=width, height=height)
            elif self.restorer_name == "pseudo":
                restorer = PseudoRestorer(
                    width=width,
                    height=height,
                    box_color=_parse_bgr(self.box_color) or (0, 255, 0),
                    box_thickness=self.box_thickness,
                    fill_color=_parse_bgr(self.fill_color, allow_none=True),
                    fill_opacity=self.fill_opacity,
                )
            else:
                raise ValueError(f"Unknown restorer: {self.restorer_name}")

            frames_done = 0

            while True:
                if self.max_frames is not None and frames_done >= self.max_frames:
                    break

                tb = _now()
                frames = decoder.read_batch()
                self._maybe_sync()
                t.t_decode_batch += (_now() - tb)

                if not frames:
                    break

                # wrap dlpack -> torch
                tb = _now()
                rgb_list: List[torch.Tensor] = []
                for f in frames:
                    rgb_list.append(torch.from_dlpack(f))
                self._maybe_sync()
                t.t_wrap_dlpack += (_now() - tb)

                # allocate buffers for this batch size
                if len(bgr_u8_bufs) != len(rgb_list):
                    bgr_u8_bufs = [torch.empty((height, width, 3), dtype=torch.uint8, device=self.device) for _ in rgb_list]
                    bgra_u8_bufs = [torch.empty((height, width, 4), dtype=torch.uint8, device=self.device) for _ in rgb_list]

                # rgb->bgr (u8)
                tb = _now()
                for i, rgb in enumerate(rgb_list):
                    rgb_to_bgr_u8_inplace(bgr_u8_bufs[i], rgb)
                self._maybe_sync()
                t.t_to_bgr += (_now() - tb)

                # to float for models/restorers
                tb = _now()
                bgr_f32_list = [bgr_u8_to_bgr_f32(x) for x in bgr_u8_bufs]
                self._maybe_sync()
                t.t_to_float += (_now() - tb)

                # detect (optional)
                detections: Optional[Sequence[Any]] = None
                if detector is not None:
                    tb = _now()
                    detections = detector.detect_batch(bgr_f32_list)
                    self._maybe_sync()
                    t.t_detect += (_now() - tb)

                # restore
                tb = _now()
                if self.restorer_name == "none":
                    out_f32_list = restorer.process_batch(bgr_f32_list)
                else:
                    assert detections is not None
                    out_f32_list = restorer.process_batch(bgr_f32_list, detections)
                self._maybe_sync()
                t.t_restore += (_now() - tb)

                # pack for encoder (BGRA u8)
                tb = _now()
                for i, out_f32 in enumerate(out_f32_list):
                    out_u8 = (out_f32.clamp(0, 1) * 255.0).to(torch.uint8)
                    pack_bgr_to_bgra_inplace(bgra_u8_bufs[i], out_u8, alpha=self.alpha)
                self._maybe_sync()
                t.t_pack += (_now() - tb)

                # encode
                tb = _now()
                encoder.encode_frames(bgra_u8_bufs)
                self._maybe_sync()
                t.t_encode += (_now() - tb)

                frames_done += len(frames)
                t.frames = frames_done

            tb = _now()
            encoder.flush()
            self._maybe_sync()
            t.t_flush += (_now() - tb)

        finally:
# Always close encoder file handle (flush is done above, but try anyway on error)
            try:
                if encoder is not None:
                    try:
                        encoder.flush()
                    except Exception:
                        pass
                    encoder.close()
            except Exception:
                pass

            t.t_total = (_now() - t0)
            if self.debug:
                print(t.pretty())

        return t