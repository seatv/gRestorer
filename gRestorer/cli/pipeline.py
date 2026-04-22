# gRestorer/cli/pipeline.py
from __future__ import annotations

import datetime as _dt
import json
import os
import queue as _queue
import threading as _threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from gRestorer.core.scene_tracker import SceneTracker, TrackerConfig
from gRestorer.detector.core import Detection, Detector as YoloDetector, FaceMetadata
from gRestorer.detector.face_detector import FaceDetector
from gRestorer.restorer.basicvsrpp_clip_restorer import BasicVSRPPClipRestorer
from gRestorer.restorer.compositor import _composite_clip_into_store, _composite_clip_into_store_laplacian
from gRestorer.restorer.pseudo_clip_restorer import PseudoClipRestorer

from gRestorer.restorer.inswapper_clip_restorer import InSwapperClipRestorer
from gRestorer.restorer.simswap_clip_restorer import SimSwapClipRestorer
from gRestorer.restorer.hyperswap_clip_restorer import HyperSwapClipRestorer

from gRestorer.core.lada_clip import LadaClip
from gRestorer.restorer.lada_basicvsrpp_clip_restorer import LadaBasicVSRPPClipRestorer
from gRestorer.restorer.lada_compositor import composite_lada_clip_into_store

from gRestorer.utils.config_util import Config
from gRestorer.video.decoder import Decoder
from gRestorer.video.encoder import Encoder

from .pipeline_utils import (
    Box,
    FrameStore,
    bgr_u8_to_bgra_u8,
    clip_box_to_bounds,
    cfg_first,
    cfg_path,
    compute_pts_fps,
    drain_store_to_encoder,
    nv12_to_rgb_hwc_u8,
    rgb_hwc_to_bgr_hwc_u8,
    rgbp_chw_to_rgb_hwc_u8,
    seam_split_boxes,
    split_frame_lr,
    sync_device,
    unsplit_boxes_layout,
    unsplit_masks_layout,
    wrap_surface_as_tensor,
    write_timecodes_v2,
)


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in ("", "0", "false", "no")


def _cfg_bool(cfg: Config, key_paths: List[Tuple[str, ...]], *, env_name: Optional[str] = None, default: bool = False) -> bool:
    value = cfg_first(cfg, key_paths, default=None)
    if value is not None:
        return bool(value)
    if env_name:
        return _truthy_env(env_name, "1" if default else "0")
    return bool(default)


def _cfg_int(cfg: Config, key_paths: List[Tuple[str, ...]], *, env_name: Optional[str] = None, default: int = 0) -> int:
    value = cfg_first(cfg, key_paths, default=None)
    if value is not None:
        return int(value)
    if env_name:
        return int(os.getenv(env_name, str(default)))
    return int(default)


def _cfg_float(cfg: Config, key_paths: List[Tuple[str, ...]], *, env_name: Optional[str] = None, default: float = 0.0) -> float:
    value = cfg_first(cfg, key_paths, default=None)
    if value is not None:
        return float(value)
    if env_name:
        return float(os.getenv(env_name, str(default)))
    return float(default)


def _cfg_str(cfg: Config, key_paths: List[Tuple[str, ...]], *, env_name: Optional[str] = None, default: str = "") -> str:
    value = cfg_first(cfg, key_paths, default=None)
    if value is not None:
        return str(value)
    if env_name:
        return str(os.getenv(env_name, default))
    return str(default)


@dataclass
class DetStats:
    frames_total: int = 0
    frames_with_det: int = 0
    total_boxes: int = 0
    total_roi_area_px: float = 0.0
    frame_area_px: int = 0

    def add(self, boxes, w: int, h: int) -> None:
        self.frames_total += 1
        if self.frame_area_px == 0:
            self.frame_area_px = int(w) * int(h)
        if not boxes:
            return

        self.frames_with_det += 1
        self.total_boxes += len(boxes)

        a = 0.0
        for (t, l, b, r) in boxes:
            ww = max(0, int(r) - int(l) + 1)
            hh = max(0, int(b) - int(t) + 1)
            a += float(ww * hh)
        self.total_roi_area_px += a

    def summary(self):
        avg_area = (self.total_roi_area_px / max(1, self.frames_with_det))
        pct = (avg_area / max(1, self.frame_area_px)) * 100.0
        return self.frames_with_det, self.frames_total, self.total_boxes, avg_area, pct


@dataclass
class PipelineMetrics:
    processed_frames: int = 0
    early_passthrough_frames: int = 0

    t_decode: float = 0.0
    t_det: float = 0.0
    t_track: float = 0.0
    t_restore: float = 0.0
    t_encode: float = 0.0
    t_mux: float = 0.0

    t_queue_wait: float = 0.0
    t_prepare: float = 0.0
    t_upload: float = 0.0
    t_csc: float = 0.0

    wall_start: _dt.datetime | None = None
    wall_end: _dt.datetime | None = None

    det_stats: DetStats = field(default_factory=DetStats)
    backpressure_waits: int = 0

    def sum_parts(self) -> float:
        return self.t_decode + self.t_det + self.t_track + self.t_restore + self.t_encode


def _detect_face_swap_backend(model_path: str, configured_backend: str = "auto") -> str:
    b = str(configured_backend or "auto").strip().lower()
    if b in ("inswapper", "simswap", "hyperswap"):
        return b

    name = Path(str(model_path or "")).name.lower()
    if "hyperswap" in name:
        return "hyperswap"
    if "simswap" in name:
        return "simswap"
    return "inswapper"



def _pick_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
        return torch.device(f"xpu:{gpu_id}")
    return torch.device("cpu")


def _tensor_boxes_to_list_xyxy(
    boxes_xyxy: Optional[torch.Tensor],
    *,
    w: Optional[int] = None,
    h: Optional[int] = None,
) -> List[Box]:
    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        return []

    w_f = float(w) if w is not None else None
    h_f = float(h) if h is not None else None

    out: List[Box] = []
    for row in boxes_xyxy.tolist():
        x1, y1, x2, y2 = row

        if w_f is not None:
            x1 = max(0.0, min(float(x1), w_f))
            x2 = max(0.0, min(float(x2), w_f))
        if h_f is not None:
            y1 = max(0.0, min(float(y1), h_f))
            y2 = max(0.0, min(float(y2), h_f))

        l = int(x1)
        t = int(y1)
        r = int(x2)
        b = int(y2)
        out.append((t, l, b, r))
    return out


def _extract_masks_list(det: Detection) -> Optional[List[Optional[torch.Tensor]]]:
    m = det.masks
    if m is None:
        return None
    if isinstance(m, torch.Tensor):
        if m.ndim == 3:
            return [m[i] for i in range(m.shape[0])]
        if m.ndim == 2:
            return [m]
    return None


def _extract_face_metas_list(det: Detection) -> Optional[List[Optional[FaceMetadata]]]:
    fm = getattr(det, "face_metas", None)
    if fm is None:
        return None
    return list(fm)


def _compute_default_store_max(width: int, height: int, max_clip_length: int) -> int:
    frame_bytes = width * height * 3
    if frame_bytes <= 0:
        return 300
    vram_budget = 1.5 * 1024 * 1024 * 1024
    budget_frames = int(vram_budget / frame_bytes)
    min_frames = max(max_clip_length * 2 + 32, 64)
    return max(min_frames, min(budget_frames, 600))


def _compute_emergency_store_max(
    width: int,
    height: int,
    max_clip_length: int,
    current_max_frames: int,
    device: torch.device,
) -> int:
    base = int(current_max_frames)
    if base <= 0:
        return base
    target = max(base, int(max_clip_length) * 4 + 64)
    abs_cap = 600
    frame_bytes = int(width) * int(height) * 3
    if frame_bytes <= 0:
        return max(base, min(target, abs_cap))
    cap_by_mem = abs_cap
    if device.type == "cuda":
        try:
            total = int(torch.cuda.get_device_properties(device).total_memory)
            budget = int(total * 0.55)
            cap_by_mem = max(base, int(budget // frame_bytes))
        except Exception:
            cap_by_mem = abs_cap
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        try:
            getp = getattr(torch.xpu, "get_device_properties", None)
            if getp is not None:
                idx = getattr(device, "index", 0) or 0
                props = getp(int(idx))
                total = int(getattr(props, "total_memory", 0))
                if total > 0:
                    budget = int(total * 0.55)
                    cap_by_mem = max(base, int(budget // frame_bytes))
        except Exception:
            cap_by_mem = abs_cap
    ceiling = max(base, min(abs_cap, cap_by_mem if cap_by_mem > 0 else abs_cap))
    return max(base, min(target, ceiling))

def _device_mem_snapshot(device: torch.device) -> dict:
    snap = {"device": str(device), "type": getattr(device, "type", "unknown")}
    try:
        if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
            idx = device.index if device.index is not None else torch.cuda.current_device()
            snap.update(
                {
                    "allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024.0 * 1024.0), 2),
                    "reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024.0 * 1024.0), 2),
                    "max_allocated_mb": round(torch.cuda.max_memory_allocated(idx) / (1024.0 * 1024.0), 2),
                    "max_reserved_mb": round(torch.cuda.max_memory_reserved(idx) / (1024.0 * 1024.0), 2),
                }
            )
        elif getattr(device, "type", None) == "xpu" and hasattr(torch, "xpu"):
            mem_getter = getattr(torch.xpu, "memory_allocated", None)
            if callable(mem_getter):
                idx = device.index if device.index is not None else 0
                snap["allocated_mb"] = round(float(mem_getter(idx)) / (1024.0 * 1024.0), 2)
    except Exception as e:
        snap["mem_error"] = repr(e)
    return snap


def _store_snapshot(store: FrameStore) -> dict:
    frames = getattr(store, "frames_bgr_u8", None)
    if not isinstance(frames, dict):
        return {"max_frames": int(getattr(store, "max_frames", -1)), "len": None}

    keys = sorted(int(k) for k in frames.keys())
    out = {
        "max_frames": int(getattr(store, "max_frames", -1)),
        "len": int(len(keys)),
        "oldest": (keys[0] if keys else None),
        "newest": (keys[-1] if keys else None),
        "head": keys[:4],
        "tail": keys[-4:] if len(keys) > 4 else keys,
    }
    try:
        out["est_vram_mb"] = round(float(store.vram_mb()), 2)
    except Exception:
        pass
    return out


def _tracker_snapshot(tracker) -> dict:
    if tracker is None:
        return {"enabled": False}

    scene_count = None
    for name in ("active_scenes", "_active_scenes", "scenes", "_scenes"):
        value = getattr(tracker, name, None)
        if value is None:
            continue
        try:
            scene_count = len(value)
            break
        except Exception:
            pass

    min_active_start = None
    try:
        fn = getattr(tracker, "min_active_start", None)
        if callable(fn):
            min_active_start = fn()
    except Exception:
        pass

    return {
        "enabled": True,
        "scene_count": scene_count,
        "min_active_start": (None if min_active_start is None else int(min_active_start)),
    }


@dataclass
class Pipeline:
    cfg: Config

    def __post_init__(self) -> None:
        self.input_path = str(self.cfg.get("input"))
        self.output_path = str(self.cfg.get("output"))
        self.max_frames: Optional[int] = self.cfg.get("max_frames", default=None)
        self.process: str = str(self.cfg.get("process", default="mosaic") or "mosaic").lower()
        if self.process not in ("mosaic", "face"):
            raise ValueError(f"Invalid process: {self.process!r}. Expected 'mosaic' or 'face'.")
        self.debug: bool = bool(self.cfg.get("debug_enabled", default=False))
        self.profile_sync: bool = bool(self.cfg.get("profile_sync", default=False))
        self.batch_size: int = int(cfg_first(self.cfg, [("runtime", "batch_size"), ("batch_size",)], default=8))
        self.dec_gpu_id: int = int(cfg_first(self.cfg, [("decoder", "gpu_id"), ("runtime", "gpu_id")], default=0))
        self.enc_gpu_id: int = int(cfg_first(self.cfg, [("encoder", "gpu_id"), ("runtime", "gpu_id")], default=self.dec_gpu_id))
        self.device: torch.device = _pick_device(self.dec_gpu_id)
        self.mode: str = str(self.cfg.get("mode", default="real")).lower()
        _restorer_default = "face_swap" if self.process == "face" else "basicvsrpp"
        self.restorer_name: str = str(self.cfg.get("restorer", default=_restorer_default) or _restorer_default).lower()
        if self.restorer_name == "face_swap" and self.process != "face":
            raise ValueError("restorer=face_swap requires process=face. Unless explicitly specified, gRestorer operates on mosaic restoration.")

        self.dec_output_format: str = str(self.cfg.get("decoder", "output_format", default="RGBP")).upper()
        self.dec_ffmpeg_input_args: str = str(self.cfg.get("decoder", "ffmpeg_input_args", default="") or "")

        if self.process == "face":
            self.det_type: str = str(
                cfg_first(
                    self.cfg,
                    [("face_detection", "det_type"), ("detection", "det_type")],
                    default="face",
                )
                or "face"
            ).lower()
            det_paths = [("face_detection", "model_path"), ("detection", "model_path")]
            imgsz_paths = [("face_detection", "imgsz"), ("detection", "imgsz")]
            conf_paths = [("face_detection", "conf_threshold"), ("detection", "conf_threshold")]
            iou_paths = [("face_detection", "iou_threshold"), ("detection", "iou_threshold")]
            fp16_paths = [("face_detection", "fp16"), ("detection", "fp16")]
        else:
            self.det_type: str = str(
                cfg_first(
                    self.cfg,
                    [("mosaic_detection", "det_type"), ("detection", "det_type")],
                    default="yolo",
                )
                or "yolo"
            ).lower()
            det_paths = [("mosaic_detection", "model_path"), ("detection", "model_path")]
            imgsz_paths = [("mosaic_detection", "imgsz"), ("detection", "imgsz")]
            conf_paths = [("mosaic_detection", "conf_threshold"), ("detection", "conf_threshold")]
            iou_paths = [("mosaic_detection", "iou_threshold"), ("detection", "iou_threshold")]
            fp16_paths = [("mosaic_detection", "fp16"), ("detection", "fp16")]

        self.det_model: str = str(cfg_first(self.cfg, det_paths, default="") or "")
        self.det_imgsz: int = int(cfg_first(self.cfg, imgsz_paths, default=640))
        self.det_conf: float = float(cfg_first(self.cfg, conf_paths, default=0.30))
        self.det_iou: float = float(cfg_first(self.cfg, iou_paths, default=0.70))
        self.det_fp16: bool = bool(cfg_first(self.cfg, fp16_paths, default=True))
        self.face_det_suppress_onnx_warnings: bool = _cfg_bool(
            self.cfg,
            [("face_detection", "suppress_onnx_warnings")],
            default=False,
        )

        self.roi_dilate: int = int(self.cfg.get("roi_dilate", default=0))
        self.use_seg_masks: bool = bool(self.cfg.get("use_seg_masks", default=True))

        self.sbs_enabled: bool = bool(self.cfg.get("sbs_enabled", default=False))
        self.sbs_layout: str = str(self.cfg.get("sbs_layout", default="lr")).lower()
        self.sbs_det_split: bool = bool(self.cfg.get("sbs_det_split", default=False))

        use_face_restoration_cfg = self.process == "face"
        if use_face_restoration_cfg:
            rest_paths = [("face_restoration", "rest_model_path"), ("restoration", "rest_model_path")]
            fp16_rest_paths = [("face_restoration", "fp16"), ("restoration", "fp16")]
            max_clip_paths = [("face_restoration", "max_clip_length"), ("restoration", "max_clip_length")]
            clip_size_paths = [("face_restoration", "clip_size"), ("restoration", "clip_size")]
            border_paths = [("face_restoration", "border_ratio"), ("restoration", "border_ratio")]
            pad_paths = [("face_restoration", "pad_mode"), ("restoration", "pad_mode")]
        else:
            rest_paths = [("mosaic_restoration", "rest_model_path"), ("restoration", "rest_model_path")]
            fp16_rest_paths = [("mosaic_restoration", "fp16"), ("restoration", "fp16")]
            max_clip_paths = [("mosaic_restoration", "max_clip_length"), ("restoration", "max_clip_length")]
            clip_size_paths = [("mosaic_restoration", "clip_size"), ("restoration", "clip_size")]
            border_paths = [("mosaic_restoration", "border_ratio"), ("restoration", "border_ratio")]
            pad_paths = [("mosaic_restoration", "pad_mode"), ("restoration", "pad_mode")]

        self.rest_model: str = str(cfg_first(self.cfg, rest_paths, default="") or "")
        self.rest_fp16: bool = bool(cfg_first(self.cfg, fp16_rest_paths, default=True))
        self.source_face_path: str = str(cfg_first(self.cfg, [("face_restoration", "source_face_path"), ("restoration", "source_face_path")], default="") or "")
        self.swap_model_path: str = str(cfg_first(self.cfg, [("face_restoration", "swap_model_path"), ("restoration", "swap_model_path")], default="") or "")
        self.swap_input_size: int = int(cfg_first(self.cfg, [("face_restoration", "swap_input_size"), ("restoration", "swap_input_size")], default=128))
        self.swap_provider: str = str(cfg_first(self.cfg, [("face_restoration", "provider"), ("restoration", "swap_provider")], default="auto") or "auto").lower()
        self.swap_backend: str = str(cfg_first(self.cfg, [("face_restoration", "swap_backend"), ("restoration", "swap_backend")], default="auto") or "auto").lower()

        self.face_comp_mask_mode: str = str(
            cfg_first(
                self.cfg,
                [
                    ("face_restoration", "compositor", "mask_mode"),
                    ("face_restoration", "compositor_mask_mode"),
                ],
                default="geom_backend_intersection",
            )
            or "geom_backend_intersection"
        ).lower()

        self.face_comp_geom_expand: float = float(
            cfg_first(
                self.cfg,
                [
                    ("face_restoration", "compositor", "geom_expand"),
                    ("face_restoration", "compositor_geom_expand"),
                ],
                default=1.05,
            )
        )

        self.face_comp_mask_erode: int = int(
            cfg_first(
                self.cfg,
                [
                    ("face_restoration", "compositor", "mask_erode"),
                    ("face_restoration", "compositor_mask_erode"),
                ],
                default=0,
            )
        )

        self.face_comp_mask_dilate: int = int(
            cfg_first(
                self.cfg,
                [
                    ("face_restoration", "compositor", "mask_dilate"),
                    ("face_restoration", "compositor_mask_dilate"),
                ],
                default=2,
            )
        )

        self.face_comp_mask_blur: int = int(
            cfg_first(
                self.cfg,
                [
                    ("face_restoration", "compositor", "mask_blur"),
                    ("face_restoration", "compositor_mask_blur"),
                ],
                default=5,
            )
        )

        self.face_comp_blend_mode: str = str(
            cfg_first(
                self.cfg,
                [
                    ("face_restoration", "compositor", "blend_mode"),
                    ("face_restoration", "compositor_blend_mode"),
                ],
                default="alpha",
            )
            or "alpha"
        ).lower()

        self.face_comp_color_transfer: str = str(
            cfg_first(
                self.cfg,
                [
                    ("face_restoration", "compositor", "color_transfer"),
                    ("face_restoration", "compositor_color_transfer"),
                ],
                default="none",
            )
            or "none"
        ).lower()

        self.face_comp_face_scale: float = float(
            cfg_first(
                self.cfg,
                [
                    ("face_restoration", "compositor", "face_scale"),
                    ("face_restoration", "compositor_face_scale"),
                ],
                default=0.0,
            )
        )

        self.face_comp_debug: bool = bool(
            cfg_first(
                self.cfg,
                [
                    ("face_restoration", "compositor", "debug"),
                    ("face_restoration", "compositor_debug"),
                ],
                default=False,
            )
        )

        self.face_enhancer_enabled: bool = _cfg_bool(self.cfg, [("enhancement", "enabled")], default=False)
        self.face_enhancer_model_path: str = str(cfg_first(self.cfg, [("enhancement", "model_path"), ("restoration", "face_enhancer_model_path")], default="") or "")
        self.face_enhancer_provider: str = str(cfg_first(self.cfg, [("enhancement", "provider"), ("face_restoration", "provider"), ("restoration", "swap_provider")], default=self.swap_provider) or self.swap_provider).lower()
        self.face_enhancer_blend: int = int(cfg_first(self.cfg, [("enhancement", "blend"), ("restoration", "face_enhancer_blend")], default=80))

        self.face_occluder_enabled: bool = _cfg_bool(self.cfg, [("occlusion", "enabled")], default=False)
        self.face_occluder_model_path: str = str(cfg_first(self.cfg, [("occlusion", "model_path")], default="") or "")
        self.face_occluder_provider: str = str(cfg_first(self.cfg, [("occlusion", "provider"), ("face_restoration", "provider"), ("restoration", "swap_provider")], default=self.swap_provider) or self.swap_provider).lower()
        self.face_occluder_threshold: float = float(cfg_first(self.cfg, [("occlusion", "threshold")], default=0.5))
        self.face_occluder_blur: int = int(cfg_first(self.cfg, [("occlusion", "blur")], default=5))
        self.face_occluder_blend: int = int(cfg_first(self.cfg, [("occlusion", "blend")], default=100))
        self.face_occluder_invert: bool = bool(cfg_first(self.cfg, [("occlusion", "invert")], default=False))

        self.landmark_refiner_enabled: bool = _cfg_bool(
            self.cfg,
            [("face_restoration", "landmark_refiner_enabled")],
            default=False,
        )
        self.landmark_refiner_model: str = str(
            cfg_first(self.cfg, [("face_restoration", "landmark_model")], default="2dfan4") or "2dfan4"
        )
        self.landmark_refiner_model_path: str = str(
            cfg_first(self.cfg, [("face_restoration", "landmark_model_path")], default="") or ""
        )
        self.landmark_refiner_provider: str = str(
            cfg_first(
                self.cfg,
                [("face_restoration", "landmark_provider"), ("face_restoration", "provider"), ("restoration", "swap_provider")],
                default=self.swap_provider,
            )
            or self.swap_provider
        ).lower()
        self.landmark_refiner_score: float = float(
            cfg_first(self.cfg, [("face_restoration", "landmark_refiner_score"), ("face_restoration", "landmark_score")], default=0.5)
        )

        self.rest_max_clip_length: int = int(cfg_first(self.cfg, max_clip_paths, default=30))
        self.rest_clip_size: int = int(cfg_first(self.cfg, clip_size_paths, default=256))
        self.rest_border_ratio: float = float(cfg_first(self.cfg, border_paths, default=0.06))
        self.rest_pad_mode: str = str(cfg_first(self.cfg, pad_paths, default="reflect"))

        self.feather_radius: int = int(cfg_first(self.cfg, [("mosaic_restoration", "feather_radius"), ("restoration", "feather_radius")], default=0))
        self.rest_blendmask: str = str(cfg_first(self.cfg, [("mosaic_restoration", "blendmask"), ("restoration", "blendmask")], default="none") or "none").lower()
        if self.rest_blendmask not in ("none", "facefusion", "laplacian"):
            raise ValueError(f"Invalid restoration.blendmask: {self.rest_blendmask!r}")

        self.rest_compositor_quantize_before_resize: bool = bool(
            cfg_first(self.cfg, [("mosaic_restoration", "compositor_quantize_before_resize"), ("restoration", "compositor_quantize_before_resize")], default=False)
        )
        self.rest_compositor_resize_backend: str = str(
            cfg_first(self.cfg, [("mosaic_restoration", "compositor_resize_backend"), ("restoration", "compositor_resize_backend")], default="torch") or "torch"
        ).lower()

        self.analysis_use_synth_rois: bool = bool(
            cfg_first(
                self.cfg,
                [("mosaic_restoration", "analysis_use_synth_rois"), ("restoration", "analysis_use_synth_rois")],
                default=False,
            )
        )
        _raw_synth_rois = self.cfg.get("synth_mosaic", "rois", default=[]) or []
        self.analysis_synth_rois: List[Tuple[int, int, int, int]] = []
        try:
            for _roi in _raw_synth_rois:
                if isinstance(_roi, (list, tuple)) and len(_roi) == 4:
                    t, l, b, r = [int(v) for v in _roi]
                    self.analysis_synth_rois.append((t, l, b, r))
        except Exception:
            self.analysis_synth_rois = []

        self.store_max_frames: int = int(cfg_first(self.cfg, [("runtime", "store_max_frames"), ("store_max_frames",)], default=0))

        self.enc_codec: str = str(self.cfg.get("encoder", "codec", default="hevc")).lower()
        self.enc_preset: str = str(self.cfg.get("encoder", "preset", default="P6"))
        self.enc_profile: str = str(self.cfg.get("encoder", "profile", default="main"))
        self.enc_qp: int = int(self.cfg.get("encoder", "qp", default=20))
        self.enc_sync_before_encode: bool = bool(self.cfg.get("encoder", "sync_before_encode", default=True))

        self.enc_mode: str = str(self.cfg.get("encoder", "mode", default="hq") or "hq").lower()
        self.enc_options_str: str = str(self.cfg.get("encoder", "options", default="") or "")
        self.enc_opt_dict = self.cfg.get("encoder", "opt", default={}) or {}
        if not isinstance(self.enc_opt_dict, dict):
            self.enc_opt_dict = {}
        self.enc_allow_unknown: bool = bool(self.cfg.get("encoder", "allow_unknown", default=False))

        self.mux_audio: str = str(self.cfg.get("encoder", "mux_audio", default="auto") or "auto").lower()
        self.mux_keep_subs: bool = bool(self.cfg.get("encoder", "mux_keep_subs", default=False))
        self.mux_extra_args: str = str(self.cfg.get("encoder", "mux_extra_args", default="") or "")
        self.mp4_faststart: bool = bool(self.cfg.get("encoder", "mp4_faststart", default=True))

        self.fs_trace_enabled: bool = _cfg_bool(
            self.cfg,
            [("debug", "face_swap", "trace"), ("debug", "face_swap", "trace_enabled")],
            env_name="GR_FS_TRACE",
            default=False,
        )
        self.fs_trace_dir: Path = Path(
            _cfg_str(
                self.cfg,
                [("debug", "face_swap", "trace_dir")],
                env_name="GR_FS_TRACE_DIR",
                default="fs_debug",
            )
        )
        self.fs_trace_detector_path: Path = self.fs_trace_dir / "detector_rois.jsonl"
        if self.fs_trace_enabled:
            self.fs_trace_dir.mkdir(parents=True, exist_ok=True)

        self.fs_debug_enabled: bool = _cfg_bool(
            self.cfg,
            [("debug", "face_swap", "enabled"), ("debug", "face_swap", "debug_images")],
            env_name="GR_FS_DEBUG",
            default=False,
        )
        self.fs_debug_dir: str = _cfg_str(
            self.cfg,
            [("debug", "face_swap", "debug_dir")],
            env_name="GR_FS_DEBUG_DIR",
            default="fs_debug",
        )
        self.fs_debug_start: int = _cfg_int(
            self.cfg,
            [("debug", "face_swap", "debug_start")],
            env_name="GR_FS_DEBUG_START",
            default=-1,
        )
        self.fs_debug_end: int = _cfg_int(
            self.cfg,
            [("debug", "face_swap", "debug_end")],
            env_name="GR_FS_DEBUG_END",
            default=-1,
        )
        self.fs_material_mad: float = _cfg_float(
            self.cfg,
            [("debug", "face_swap", "material_mad")],
            env_name="GR_FS_MATERIAL_MAD",
            default=1.0,
        )

    def _build_detector(self):
        if self.mode == "none":
            return None
        if not self.det_model:
            raise FileNotFoundError("Detector model path is empty (check config.json or --det-model)")

        print(f"[Detector] type={self.det_type}")

        common = dict(
            model_path=self.det_model,
            device=self.device,
            imgsz=self.det_imgsz,
            conf_thres=self.det_conf,
            iou_thres=self.det_iou,
            fp16=self.det_fp16,
        )

        if self.det_type == "yolo":
            return YoloDetector(**common)
        elif self.det_type in ("lada-yolo", "lada_yolo"):
            from gRestorer.detector.lada_yolo import LadaYoloDetector
            return LadaYoloDetector(**common)
        elif self.det_type == "face":
            return FaceDetector(**common, suppress_onnx_warnings=self.face_det_suppress_onnx_warnings)
        else:
            raise ValueError(f"Unknown detector_type: {self.det_type}")

    def _build_restorer(self):
        if self.mode == "none":
            return None

        if self.mode == "pseudo" or self.restorer_name == "pseudo":
            fill = self.cfg.get("visualization", "fill_color", default=[255, 0, 255])
            op = float(self.cfg.get("visualization", "fill_opacity", default=0.70))
            r, g, b = [int(x) for x in fill]
            return PseudoClipRestorer(device=self.device, fill_color_bgr=(b, g, r), fill_opacity=op)

        if self.restorer_name in ("none", "noop"):
            return None

        if self.restorer_name == "face_swap":
            if not self.source_face_path:
                raise FileNotFoundError("Source face path is empty (check config.json or --source-face)")
            if not self.swap_model_path:
                raise FileNotFoundError("Swap model path is empty (check config.json or --swap-model)")

            swap_backend = _detect_face_swap_backend(self.swap_model_path, self.swap_backend)
            print(f"[FaceSwap] backend={swap_backend}")

            if swap_backend == "simswap":
                restorer_cls = SimSwapClipRestorer
            elif swap_backend == "hyperswap":
                restorer_cls = HyperSwapClipRestorer
            else:
                restorer_cls = InSwapperClipRestorer

            return restorer_cls(
                device=self.device,
                source_face_path=self.source_face_path,
                swap_model_path=self.swap_model_path,
                swap_input_size=self.swap_input_size,
                provider=self.swap_provider,
                face_comp_mask_mode=self.face_comp_mask_mode,
                face_comp_geom_expand=self.face_comp_geom_expand,
                face_comp_mask_erode=self.face_comp_mask_erode,
                face_comp_mask_dilate=self.face_comp_mask_dilate,
                face_comp_mask_blur=self.face_comp_mask_blur,
                face_comp_blend_mode=self.face_comp_blend_mode,
                face_comp_color_transfer=self.face_comp_color_transfer,
                face_comp_face_scale=self.face_comp_face_scale,
                face_comp_debug=self.face_comp_debug,
                face_enhancer_enabled=self.face_enhancer_enabled,
                face_enhancer_model_path=self.face_enhancer_model_path,
                face_enhancer_provider=self.face_enhancer_provider,
                face_enhancer_blend=self.face_enhancer_blend,
                face_occluder_enabled=self.face_occluder_enabled,
                face_occluder_model_path=self.face_occluder_model_path,
                face_occluder_provider=self.face_occluder_provider,
                face_occluder_threshold=self.face_occluder_threshold,
                face_occluder_blur=self.face_occluder_blur,
                face_occluder_blend=self.face_occluder_blend,
                face_occluder_invert=self.face_occluder_invert,
                landmark_refiner_enabled=self.landmark_refiner_enabled,
                landmark_model=self.landmark_refiner_model,
                landmark_model_path=self.landmark_refiner_model_path,
                landmark_provider=self.landmark_refiner_provider,
                landmark_score=self.landmark_refiner_score,
                debug_enabled=self.fs_debug_enabled,
                debug_dir=self.fs_debug_dir,
                debug_start=self.fs_debug_start,
                debug_end=self.fs_debug_end,
                material_change_mad_threshold=self.fs_material_mad,
            )

        if not self.rest_model:
            raise FileNotFoundError("Restoration model path is empty (check config.json or --rest-model)")

        if self.restorer_name in ("lada", "lada-basicvsrpp", "lada_basicvsrpp"):
            return LadaBasicVSRPPClipRestorer(
                model_path=self.rest_model,
                device=self.device,
                fp16=self.rest_fp16,
                max_frames=32,
            )

        return BasicVSRPPClipRestorer(
            device=self.device,
            checkpoint_path=self.rest_model,
            fp16=self.rest_fp16,
            config=None,
        )


    def _trace_detector_frame(
        self,
        *,
        frame_num: int,
        boxes: List[Box],
        face_metas: Optional[List[Optional[FaceMetadata]]],
    ) -> None:
        if not self.fs_trace_enabled:
            return
        payload = {
            "frame": int(frame_num),
            "roi_count": int(len(boxes)),
            "boxes_tlbr": [[int(t), int(l), int(b), int(r)] for (t, l, b, r) in boxes],
            "face_meta_count": int(sum(1 for fm in (face_metas or []) if fm is not None)),
            "face_meta": [],
        }
        for fm in (face_metas or []):
            if fm is None:
                payload["face_meta"].append(None)
                continue
            payload["face_meta"].append({
                "bbox_xyxy": [float(v) for v in fm.bbox_xyxy],
                "det_score": None if fm.det_score is None else float(fm.det_score),
                "has_kps": bool(fm.kps is not None),
            })
        with open(self.fs_trace_detector_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _print_face_swap_stats(self, restorer) -> None:
        if restorer is None:
            return
        get_lines = getattr(restorer, "get_stats_lines", None)
        if callable(get_lines):
            for line in get_lines():
                print(line)

    def _annotate_peer_crop_shapes(self, clips) -> None:
        if not clips:
            return
        frame_to_shapes: dict[int, list[tuple[int, int]]] = {}
        for clip in clips:
            try:
                for fn, shape in zip(clip.frame_nums, clip.crop_shapes):
                    ch, cw = int(shape[0]), int(shape[1])
                    frame_to_shapes.setdefault(int(fn), []).append((ch, cw))
            except Exception:
                continue
        for clip in clips:
            try:
                peer_map: dict[int, list[tuple[int, int]]] = {}
                for fn in clip.frame_nums:
                    peer_map[int(fn)] = list(frame_to_shapes.get(int(fn), []))
                setattr(clip, "peer_crop_shapes_by_frame", peer_map)
            except Exception:
                continue

    def run(self) -> None:
        inp = Path(self.input_path)
        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.fs_trace_enabled:
            try:
                self.fs_trace_detector_path.unlink(missing_ok=True)
            except Exception:
                pass

        metrics = PipelineMetrics()
        metrics.wall_start = _dt.datetime.now()
        t0_all = time.perf_counter()

        decoder = Decoder(
            input_path=str(inp),
            gpu_id=self.dec_gpu_id,
            batch_size=self.batch_size,
            output_format=self.dec_output_format,
            ffmpeg_input_args=self.dec_ffmpeg_input_args,
            trim_negative_pts=False,
        )

        w = int(decoder.metadata.width)
        h = int(decoder.metadata.height)
        fps = float(decoder.metadata.fps or 0.0) or 0.0
        total_frames = int(decoder.metadata.num_frames or 0)

        if self.analysis_use_synth_rois:
            print(f"[Analysis] Fixed synth ROIs enabled: {len(self.analysis_synth_rois)} boxes/frame")

        if self.sbs_enabled:
            if w < 2 or w % 2 != 0:
                print(f"[SBS] Warning: width {w} is not even; splitting by floor(w/2).")
            if self.sbs_layout not in ("lr", "rl"):
                raise ValueError(f"Invalid --sbs-layout: {self.sbs_layout!r}")

        encoder = Encoder(
            output_path=str(out),
            width=w,
            height=h,
            fps=fps,
            codec=self.enc_codec,
            preset=self.enc_preset,
            profile=self.enc_profile,
            qp=self.enc_qp,
            gpu_id=self.enc_gpu_id,
            input_path=str(inp),
            mode=self.enc_mode,
            nvenc_options_str=self.enc_options_str,
            nvenc_options=self.enc_opt_dict,
            nvenc_allow_unknown=self.enc_allow_unknown,
            mux_audio=self.mux_audio,
            mux_keep_subs=self.mux_keep_subs,
            mux_extra_args=self.mux_extra_args,
            mp4_faststart=self.mp4_faststart,
            max_frames=self.max_frames,
        )

        detector = None if self.analysis_use_synth_rois else self._build_detector()
        restorer = self._build_restorer()
        use_lada_restoration = isinstance(restorer, LadaBasicVSRPPClipRestorer)

        tracker = None
        if self.mode != "none":
            tracker_cfg = TrackerConfig(
                clip_size=self.rest_clip_size,
                max_clip_length=self.rest_max_clip_length,
                pad_mode=self.rest_pad_mode,
                border_size=self.rest_border_ratio,
                debug=self.debug,
                use_seg_masks=self.use_seg_masks,
            )

            if use_lada_restoration:
                tracker = SceneTracker(cfg=tracker_cfg, clip_cls=LadaClip, seg_mask_only=True)
                print("[Tracker] LADA clip+mask semantics enabled")
            else:
                tracker = SceneTracker(cfg=tracker_cfg)

        if self.store_max_frames == 0:
            computed_max = _compute_default_store_max(w, h, self.rest_max_clip_length)
            store = FrameStore(max_frames=computed_max)
        elif self.store_max_frames < 0:
            store = FrameStore(max_frames=0)
        else:
            store = FrameStore(max_frames=self.store_max_frames)

        if store.max_frames > 0:
            est_mb = (store.max_frames * w * h * 3) / (1024.0 * 1024.0)
            print(f"[FrameStore] max_frames={store.max_frames} (~{est_mb:.0f} MB VRAM budget)")
        else:
            print("[FrameStore] max_frames=unlimited")

        def _composite_selected(clip, restored):
            if use_lada_restoration:
                composite_lada_clip_into_store(
                    clip=clip,
                    restored_frames_u8=restored,
                    store_bgr_u8=store.frames_bgr_u8,
                    model_dtype=restorer.model_dtype,
                )
                return

            if self.rest_blendmask == "laplacian":
                _composite_clip_into_store_laplacian(
                    clip=clip,
                    restored_frames=restored,
                    store_bgr_u8=store.frames_bgr_u8,
                    feather_radius=int(self.feather_radius),
                    quantize_before_resize=bool(self.rest_compositor_quantize_before_resize),
                    resize_backend=str(self.rest_compositor_resize_backend),
                )
            else:
                _composite_clip_into_store(
                    clip=clip,
                    restored_frames=restored,
                    store_bgr_u8=store.frames_bgr_u8,
                    feather_radius=int(self.feather_radius),
                    quantize_before_resize=bool(self.rest_compositor_quantize_before_resize),
                    resize_backend=str(self.rest_compositor_resize_backend),
                    blendmask_mode=self.rest_blendmask,
                )


        pts_log: List[Tuple[int, Optional[int]]] = []
        pbar_total = (self.max_frames if self.max_frames is not None else (total_frames if total_frames > 0 else None))
        pbar = tqdm(total=pbar_total, disable=self.debug)
        frame_num = 0

        store_debug = _truthy_env("GRESTORER_DEBUG_STORE", "0")
        store_debug_every = max(1, int(os.getenv("GRESTORER_DEBUG_STORE_EVERY", "100")))
        store_debug_verbose = _truthy_env("GRESTORER_DEBUG_STORE_VERBOSE", "0")

        def _emit_store_debug(
            tag: str,
            *,
            reason: str = "",
            safe_before: Optional[int] = None,
            extra: Optional[dict] = None,
        ) -> None:
            if not store_debug:
                return
            payload = {
                "tag": tag,
                "reason": reason,
                "frame_num": int(frame_num),
                "safe_before": (None if safe_before is None else int(safe_before)),
                "processed_frames": int(metrics.processed_frames),
                "backpressure_waits": int(metrics.backpressure_waits),
                "store": _store_snapshot(store),
                "tracker": _tracker_snapshot(tracker),
                "device_mem": _device_mem_snapshot(self.device),
            }
            if extra:
                payload.update(extra)
            print("[StoreDebug] " + json.dumps(payload, sort_keys=True, default=str))

        def _drain_store(reason: str, safe_before: int) -> float:
            t0_drain = time.perf_counter()
            before = _store_snapshot(store)

            if store_debug and store_debug_verbose:
                _emit_store_debug(
                    "pre_drain",
                    reason=reason,
                    safe_before=safe_before,
                    extra={"store_before": before},
                )

            try:
                drain_store_to_encoder(
                    store=store,
                    safe_before=int(safe_before),
                    encoder=encoder,
                    device=self.device,
                    sync_before_encode=self.enc_sync_before_encode,
                    pts_log=pts_log,
                )
            except Exception as e:
                after_exc = _store_snapshot(store)
                _emit_store_debug(
                    "drain_exception",
                    reason=reason,
                    safe_before=safe_before,
                    extra={
                        "error": repr(e),
                        "store_before": before,
                        "store_after": after_exc,
                    },
                )
                raise

            elapsed = time.perf_counter() - t0_drain
            after = _store_snapshot(store)
            drained = None
            if before.get("len") is not None and after.get("len") is not None:
                drained = int(before["len"]) - int(after["len"])

            if store_debug and (store_debug_verbose or drained not in (None, 0)):
                _emit_store_debug(
                    "post_drain",
                    reason=reason,
                    safe_before=safe_before,
                    extra={
                        "elapsed_ms": round(elapsed * 1000.0, 2),
                        "drained": drained,
                        "store_before": before,
                        "store_after": after,
                    },
                )
            return elapsed

        use_thread_prefetch = getattr(decoder, "_ffmpeg_proc", None) is not None
        stop = _threading.Event()
        prod_exc: dict[str, BaseException] = {}
        prod: Optional[_threading.Thread] = None
        q: Optional[_queue.Queue[Optional[List[object]]]] = None

        def consume_batch(batch: List[object], batch_pts: Optional[List[Optional[int]]] = None) -> None:
            nonlocal frame_num
            if self.max_frames is not None:
                remaining = self.max_frames - frame_num
                if remaining <= 0:
                    return
                if len(batch) > remaining:
                    batch = batch[:remaining]
                    if batch_pts is not None:
                        batch_pts = batch_pts[:remaining]

            if batch_pts is None:
                batch_pts = [None] * len(batch)
            while len(batch_pts) < len(batch):
                batch_pts.append(None)

            t0_prep = time.perf_counter()
            batch_rgb: List[torch.Tensor] = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    t_cpu = item
                    is_nv12 = (
                        t_cpu.ndim == 2 and t_cpu.dtype == torch.uint8 and int(t_cpu.shape[0]) == (h * 3 // 2) and int(t_cpu.shape[1]) == w
                    )
                    if is_nv12:
                        if self.device.type != "cpu":
                            t0_up = time.perf_counter()
                            nv12_dev = t_cpu.to(self.device, non_blocking=True)
                            metrics.t_upload += (time.perf_counter() - t0_up)
                        else:
                            nv12_dev = t_cpu
                        t0_csc = time.perf_counter()
                        rgb = nv12_to_rgb_hwc_u8(nv12_dev, width=w, height=h)
                        metrics.t_csc += (time.perf_counter() - t0_csc)
                    else:
                        rgb = t_cpu
                        if self.device.type != "cpu":
                            t0_up = time.perf_counter()
                            rgb = rgb.to(self.device, non_blocking=True)
                            metrics.t_upload += (time.perf_counter() - t0_up)
                    batch_rgb.append(rgb.contiguous())
                    continue

                t = wrap_surface_as_tensor(item)
                if t.ndim == 3 and t.shape[-1] == 3:
                    rgb = t
                else:
                    rgb = rgbp_chw_to_rgb_hwc_u8(t)
                if self.device.type != "cpu" and rgb.device != self.device:
                    rgb = rgb.to(self.device, non_blocking=True)
                batch_rgb.append(rgb.contiguous())

            metrics.t_prepare += (time.perf_counter() - t0_prep)
            batch_bgr_u8: List[torch.Tensor] = [rgb_hwc_to_bgr_hwc_u8(rgb) for rgb in batch_rgb]

            detections: List[Detection] = []
            if self.analysis_use_synth_rois:
                detections = [Detection(boxes=None, scores=None, classes=None, masks=None, face_metas=None) for _ in batch_rgb]
            elif detector is not None:
                if self.sbs_enabled and self.sbs_det_split:
                    left_frames: List[torch.Tensor] = []
                    right_frames: List[torch.Tensor] = []
                    half_w = w // 2
                    for bgr in batch_bgr_u8:
                        l, r = split_frame_lr(bgr, layout=self.sbs_layout)
                        left_frames.append(l.contiguous())
                        right_frames.append(r.contiguous())

                    t0 = time.perf_counter()
                    det_l = detector.detect_batch(left_frames)
                    det_r = detector.detect_batch(right_frames)
                    metrics.t_det += (time.perf_counter() - t0)

                    for dl, dr in zip(det_l, det_r):
                        boxes_l = _tensor_boxes_to_list_xyxy(dl.boxes, w=half_w, h=h)
                        boxes_r = _tensor_boxes_to_list_xyxy(dr.boxes, w=half_w, h=h)
                        masks_l = _extract_masks_list(dl) if self.use_seg_masks else None
                        masks_r = _extract_masks_list(dr) if self.use_seg_masks else None
                        merged_boxes = unsplit_boxes_layout(boxes_l, boxes_r, half_w=half_w, layout=self.sbs_layout)
                        merged_masks = unsplit_masks_layout(masks_l, masks_r, full_w=w, half_w=half_w, layout=self.sbs_layout)
                        det = Detection(
                            boxes=torch.tensor([[b[1], b[0], b[3], b[2]] for b in merged_boxes], dtype=torch.float32, device="cpu") if merged_boxes else None,
                            scores=None,
                            classes=None,
                            masks=None,
                            face_metas=None,
                        )
                        if merged_masks is not None:
                            try:
                                mm = [m for m in merged_masks if m is not None]
                                if len(mm) == len(merged_masks) and len(mm) > 0:
                                    det.masks = torch.stack(mm, dim=0)
                            except Exception:
                                det.masks = None
                        detections.append(det)
                else:
                    t0 = time.perf_counter()
                    detections = detector.detect_batch(batch_bgr_u8)
                    metrics.t_det += (time.perf_counter() - t0)
            else:
                detections = [Detection(boxes=None, scores=None, classes=None, masks=None, face_metas=None) for _ in batch_rgb]

            if tracker is None or self.mode == "none":
                if self.enc_sync_before_encode:
                    sync_device(self.device)
                t0 = time.perf_counter()
                for i, bgr_u8 in enumerate(batch_bgr_u8):
                    frame_pts = batch_pts[i] if i < len(batch_pts) else None
                    pts_log.append((frame_num, frame_pts))
                    encoder.encode_frame(bgr_u8_to_bgra_u8(bgr_u8))
                    frame_num += 1
                    metrics.processed_frames += 1
                    pbar.update(1)
                metrics.t_encode += (time.perf_counter() - t0)
                return

            safe_before_batch: int = frame_num
            for i, bgr_u8 in enumerate(batch_bgr_u8):
                if self.max_frames is not None and frame_num >= self.max_frames:
                    break

                det = detections[i] if i < len(detections) else Detection(boxes=None, scores=None, classes=None, masks=None, face_metas=None)

                if self.analysis_use_synth_rois and self.analysis_synth_rois:
                    boxes = [clip_box_to_bounds(bx, w=w, h=h) for bx in self.analysis_synth_rois]
                    masks_list = None
                    face_metas_list = None
                else:
                    boxes = _tensor_boxes_to_list_xyxy(det.boxes, w=w, h=h)
                    masks_list = _extract_masks_list(det) if (self.use_seg_masks and det.masks is not None) else None
                    face_metas_list = _extract_face_metas_list(det)

                    if self.roi_dilate > 0 and boxes:
                        dil = self.roi_dilate
                        boxes = [(t - dil, l - dil, b + dil, r + dil) for (t, l, b, r) in boxes]

                    if boxes:
                        boxes = [clip_box_to_bounds(bx, w=w, h=h) for bx in boxes]

                if self.sbs_enabled and boxes:
                    seam_x = w // 2
                    boxes, masks_list = seam_split_boxes(boxes, seam_x=seam_x, full_w=w, full_h=h, masks=masks_list)
                    # For now, face metadata is not re-mapped for seam-split mode.
                    if face_metas_list is not None:
                        face_metas_list = None

                metrics.det_stats.add(boxes, w=w, h=h)
                self._trace_detector_frame(frame_num=frame_num, boxes=boxes, face_metas=face_metas_list)
                frame_pts = batch_pts[i] if i < len(batch_pts) else None
                store.put(frame_num, bgr_u8, pts=frame_pts)

                t0 = time.perf_counter()
                step = tracker.step_frame(frame_num, bgr_u8, boxes, masks_list, face_metas_list)
                metrics.t_track += (time.perf_counter() - t0)

                if step.new_clips and restorer is not None:
                    self._annotate_peer_crop_shapes(step.new_clips)
                    for clip in step.new_clips:
                        t0 = time.perf_counter()
                        restored = restorer.restore_clip(clip)
                        _composite_selected(clip, restored)
                        metrics.t_restore += (time.perf_counter() - t0)

                min_start = tracker.min_active_start()
                safe_before = int(min_start) if min_start is not None else int(frame_num + 1)
                safe_before_batch = safe_before

                if (len(boxes) == 0) and (min_start is None) and (not step.new_clips):
                    metrics.early_passthrough_frames += 1

                frame_num += 1
                metrics.processed_frames += 1
                pbar.update(1)

            t0 = time.perf_counter()
            metrics.t_encode += _drain_store("batch", int(safe_before_batch))

            if store_debug and (metrics.processed_frames % store_debug_every == 0):
                _emit_store_debug(
                    "heartbeat",
                    reason="periodic",
                    safe_before=int(safe_before_batch),
                )
        try:
            if use_thread_prefetch:
                q = _queue.Queue(maxsize=2)

                def _q_put(item: Optional[List[object]]) -> bool:
                    assert q is not None
                    while True:
                        if stop.is_set():
                            return False
                        try:
                            q.put(item, timeout=0.10)
                            return True
                        except _queue.Full:
                            continue

                def producer() -> None:
                    try:
                        while not stop.is_set():
                            t0 = time.perf_counter()
                            batch0 = decoder.read_batch()
                            metrics.t_decode += (time.perf_counter() - t0)
                            if not batch0:
                                _q_put(None)
                                return
                            if not _q_put(list(batch0)):
                                return
                    except BaseException as e:
                        prod_exc["e"] = e
                        _q_put(None)

                prod = _threading.Thread(target=producer, name="decode-producer", daemon=True)
                prod.start()

                while True:
                    if self.max_frames is not None and frame_num >= self.max_frames:
                        break
                    while store.is_full():
                        metrics.backpressure_waits += 1
                        min_start = tracker.min_active_start() if tracker is not None else None
                        sb = int(min_start) if min_start is not None else int(frame_num + 1)
                        drain_store_to_encoder(
                            store=store,
                            safe_before=sb,
                            encoder=encoder,
                            device=self.device,
                            sync_before_encode=self.enc_sync_before_encode,
                            pts_log=pts_log,
                        )
                        if not store.is_full():
                            break
                        if store.max_frames > 0 and min_start is not None and len(store.frames_bgr_u8) > 0:
                            oldest = min(store.frames_bgr_u8.keys())
                            if sb <= oldest:
                                new_max = _compute_emergency_store_max(w, h, self.rest_max_clip_length, store.max_frames, self.device)
                                if new_max > store.max_frames:
                                    old_max = store.max_frames
                                    store.max_frames = new_max
                                    try:
                                        mb = store.vram_mb()
                                        print(f"[FrameStore] backpressure: active scene blocks drain (oldest={oldest}, safe_before={sb}); raising max_frames {old_max}->{new_max} (~{mb:.0f} MB est in-use)")
                                    except Exception:
                                        print(f"[FrameStore] backpressure: active scene blocks drain (oldest={oldest}, safe_before={sb}); raising max_frames {old_max}->{new_max}")

                                    _emit_store_debug(
                                        "grow_store",
                                        reason="backpressure_grow",
                                        safe_before=sb,
                                        extra={
                                            "oldest": int(oldest),
                                            "old_max_frames": int(old_max),
                                            "new_max_frames": int(new_max),
                                        },
                                    )
                                    continue
                                    time.sleep(0.001)

                    t0 = time.perf_counter()
                    batch = q.get()
                    metrics.t_queue_wait += (time.perf_counter() - t0)
                    if batch is None:
                        break
                    consume_batch(batch)
            else:
                while True:
                    if self.max_frames is not None and frame_num >= self.max_frames:
                        break
                    while store.is_full():
                        metrics.backpressure_waits += 1
                        min_start = tracker.min_active_start() if tracker is not None else None
                        sb = int(min_start) if min_start is not None else int(frame_num + 1)
                        drain_store_to_encoder(
                            store=store,
                            safe_before=sb,
                            encoder=encoder,
                            device=self.device,
                            sync_before_encode=self.enc_sync_before_encode,
                            pts_log=pts_log,
                        )
                        if not store.is_full():
                            break
                        if store.max_frames > 0 and min_start is not None and len(store.frames_bgr_u8) > 0:
                            oldest = min(store.frames_bgr_u8.keys())
                            if sb <= oldest:
                                new_max = _compute_emergency_store_max(w, h, self.rest_max_clip_length, store.max_frames, self.device)
                                if new_max > store.max_frames:
                                    old_max = store.max_frames
                                    store.max_frames = new_max
                                    try:
                                        mb = store.vram_mb()
                                        print(f"[FrameStore] backpressure: active scene blocks drain (oldest={oldest}, safe_before={sb}); raising max_frames {old_max}->{new_max} (~{mb:.0f} MB est in-use)")
                                    except Exception:
                                        print(f"[FrameStore] backpressure: active scene blocks drain (oldest={oldest}, safe_before={sb}); raising max_frames {old_max}->{new_max}")

                                    _emit_store_debug(
                                        "grow_store",
                                        reason="backpressure_grow",
                                        safe_before=sb,
                                        extra={
                                            "oldest": int(oldest),
                                            "old_max_frames": int(old_max),
                                            "new_max_frames": int(new_max),
                                        },
                                    )
                                    continue
                                    time.sleep(0.001)

                    t0 = time.perf_counter()
                    batch0, batch_pts_raw = decoder.read_batch_with_pts()
                    metrics.t_decode += (time.perf_counter() - t0)
                    if not batch0:
                        break
                    consume_batch(list(batch0), batch_pts=batch_pts_raw)
        finally:
            stop.set()
            if prod is not None:
                try:
                    prod.join(timeout=2.0)
                except Exception:
                    pass

            try:
                decoder.close()
            except Exception:
                pass

            if "e" in prod_exc:
                raise prod_exc["e"]

            if tracker is not None and restorer is not None:
                eof_clips = tracker.flush_eof()
                self._annotate_peer_crop_shapes(eof_clips)
                for clip in eof_clips:
                    restored = restorer.restore_clip(clip)
                    _composite_selected(clip, restored)
            _drain_store("final_flush", 10**18)

            t_total_no_mux = time.perf_counter() - t0_all
            tc_path: Optional[str] = None
            pts_fps: float = fps
            is_vfr: bool = False
            if pts_log:
                pts_fps, is_vfr = compute_pts_fps(pts_log, fallback_fps=fps)
                tc_path = write_timecodes_v2(pts_log, self.output_path, fps=fps)
                if abs(pts_fps - fps) / max(fps, 0.001) > 0.002:
                    print(f"[PTS] FPS mismatch: metadata={fps:.3f}  pts_derived={pts_fps:.3f}")
                if is_vfr:
                    print("[PTS] WARNING: Variable frame rate detected. Timecodes file written for accurate remux.")
                encoder._pts_fps = pts_fps
                encoder._pts_timecodes_path = tc_path
                encoder._pts_is_vfr = is_vfr

            t0 = time.perf_counter()
            try:
                pbar.refresh()
                pbar.close()
            except Exception:
                pass
            encoder.close()
            metrics.t_mux += (time.perf_counter() - t0)
            metrics.wall_end = _dt.datetime.now()
            sum_parts = metrics.sum_parts()
            overhead = t_total_no_mux - sum_parts
            t_total_with_mux = t_total_no_mux + metrics.t_mux

            print(
                f"[Pipeline] Processed {metrics.processed_frames} frames: "
                f"t_decode={metrics.t_decode:.2f}s t_det={metrics.t_det:.2f}s "
                f"t_track={metrics.t_track:.2f}s t_restore={metrics.t_restore:.2f}s "
                f"t_encode={metrics.t_encode:.2f}s"
            )
            print(
                f"[Pipeline] Prefetch stats: "
                f"t_queue_wait={metrics.t_queue_wait:.2f}s t_prepare={metrics.t_prepare:.2f}s "
                f"t_upload={metrics.t_upload:.2f}s t_csc={metrics.t_csc:.2f}s"
            )
            if metrics.backpressure_waits > 0:
                print(f"[Pipeline] Backpressure waits: {metrics.backpressure_waits} (store peaked at max_frames={store.max_frames})")
            print(
                f"[Pipeline] Processing time (no mux) = {t_total_no_mux:.2f}s "
                f"Overhead = {overhead:.2f}s (sum_parts={sum_parts:.2f}s)"
            )
            print(f"[Pipeline] Total time (with mux) = {t_total_with_mux:.2f}s (mux={metrics.t_mux:.2f}s)")
            print(f"[Pipeline] DONE: Processed  &  Remuxed {metrics.processed_frames} frames")
            print(f"[Pipeline] early_passthrough_frames={metrics.early_passthrough_frames}")
            fw, ft, tb, avg_area, pct = metrics.det_stats.summary()
            print(f"[DetStats] frames_with_det={fw}/{ft} total_boxes={tb} avg_roi_area_px={avg_area:.2f} ({pct:.4f}% of frame)")
            if metrics.wall_start and metrics.wall_end:
                elapsed = metrics.wall_end - metrics.wall_start
                print(f"[Pipeline] Wall clock: start={metrics.wall_start} end={metrics.wall_end} elapsed={elapsed}")
            print(f"[Pipeline] perf_counter elapsed = {t_total_with_mux:.2f}s")
            self._print_face_swap_stats(restorer)

            if tc_path and not is_vfr:
                try:
                    Path(tc_path).unlink(missing_ok=True)
                except Exception:
                    pass

            try:
                pbar.close()
            except Exception:
                pass
