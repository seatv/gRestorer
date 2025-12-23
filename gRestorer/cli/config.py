# gRestorer/cli/config.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

from gRestorer.utils.config_util import Config


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gRestorer", description="GPU video pipeline: decode -> detect -> restore -> encode")

    # Core
    p.add_argument("--input", dest="input_path", required=True, help="Input video path")
    p.add_argument("--output", dest="output_path", required=True, help="Output video path")
    p.add_argument("--config", dest="config_path", default=None, help="Path to config.json (defaults to ./config.json)")

    # Runtime / perf
    p.add_argument("--gpu-id", dest="gpu_id", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    p.add_argument("--debug", dest="debug", action="store_true", default=None)

    # Timing (ok if pipeline ignores it)
    p.add_argument("--profile-sync", dest="profile_sync", action="store_true", default=None)

    # Scene tracking / clip mask fidelity
    p.add_argument(
        "--use-seg-masks",
        dest="use_seg_masks",
        action="store_true",
        default=None,
        help="Use detector segmentation masks for clip compositing (LADA-faithful).",
    )
    p.add_argument(
        "--no-seg-masks",
        dest="use_seg_masks",
        action="store_false",
        default=None,
        help="Disable segmentation masks; use rectangle box masks only.",
    )

    # Restorer selection
    p.add_argument("--restorer", dest="restorer", choices=["none", "pseudo", "pseudo_clip", "grestorer", "basicvsrpp"], default=None)

    # Restorer model
    p.add_argument(
        "--rest-model",
        dest="rest_model_path",
        default=None,
        help="Path to BasicVSR++ checkpoint (.pth). Required for --restorer basicvsrpp",
    )

    # Restorer perf
    p.add_argument(
        "--rest-fp16",
        dest="rest_fp16",
        action="store_true",
        default=None,
        help="Enable FP16 for restoration (CUDA only).",
    )
    p.add_argument(
        "--no-rest-fp16",
        dest="rest_fp16",
        action="store_false",
        default=None,
        help="Disable FP16 for restoration.",
    )


    # Detector
    p.add_argument("--det-model", dest="det_model_path", default=None)
    p.add_argument("--det-conf", dest="det_conf", type=float, default=None)
    p.add_argument("--det-iou", dest="det_iou", type=float, default=None)
    p.add_argument("--det-imgsz", dest="det_imgsz", type=int, default=None)
    p.add_argument("--det-fp16", dest="det_fp16", action="store_true", default=None)

    # Visualization overrides (B,G,R)
    p.add_argument("--box-color", dest="box_color", default=None, help='B,G,R (e.g. "0,255,0")')
    p.add_argument("--box-thickness", dest="box_thickness", type=int, default=None)
    p.add_argument("--fill-color", dest="fill_color", default=None, help='B,G,R (e.g. "128,128,128")')
    p.add_argument("--fill-opacity", dest="fill_opacity", type=float, default=None)

    # Encode overrides
    p.add_argument("--codec", dest="codec", default=None)
    p.add_argument("--preset", dest="preset", default=None)     # e.g. P7
    p.add_argument("--profile", dest="profile", default=None)   # e.g. main
    p.add_argument("--qp", dest="qp", type=int, default=None)
    p.add_argument("--alpha", dest="alpha", type=int, default=None)

    return p


def _apply_if_not_none(cfg: Config, key: str, value: Any) -> None:
    if value is not None:
        cfg.data[key] = value


def _apply_visual_if_not_none(cfg: Config, key: str, value: Any) -> None:
    if value is not None:
        cfg.set("visualization", key, value=value)


def parse_args(argv=None) -> Config:
    args = create_parser().parse_args(argv)

    # 1) Load base config.json
    cfg_path = args.config_path or os.environ.get("GRESTORER_CONFIG") or "config.json"
    base = Config.load_json(cfg_path)
    cfg = Config(base)

    # 2) Mandatory CLI values always win
    cfg.data["input_path"] = args.input_path
    cfg.data["output_path"] = args.output_path

    # 3) Optional overrides (only when specified)
    _apply_if_not_none(cfg, "gpu_id", args.gpu_id)
    _apply_if_not_none(cfg, "batch_size", args.batch_size)
    _apply_if_not_none(cfg, "max_frames", args.max_frames)
    _apply_if_not_none(cfg, "restorer", args.restorer)

    if args.debug is not None and args.debug:
        cfg.data["debug"] = True
    if args.profile_sync is not None and args.profile_sync:
        cfg.data["profile_sync"] = True

    # Scene tracking / clip mask fidelity
    _apply_if_not_none(cfg, "use_seg_masks", args.use_seg_masks)

    # Detector
    _apply_if_not_none(cfg, "det_model_path", args.det_model_path)
    _apply_if_not_none(cfg, "det_conf", args.det_conf)
    _apply_if_not_none(cfg, "det_iou", args.det_iou)
    _apply_if_not_none(cfg, "det_imgsz", args.det_imgsz)
    if args.det_fp16 is not None and args.det_fp16:
        cfg.data["det_fp16"] = True

    # Visualization (nested)
    _apply_visual_if_not_none(cfg, "box_color", args.box_color)
    _apply_visual_if_not_none(cfg, "box_thickness", args.box_thickness)
    _apply_visual_if_not_none(cfg, "fill_color", args.fill_color)
    _apply_visual_if_not_none(cfg, "fill_opacity", args.fill_opacity)

    # Encode
    _apply_if_not_none(cfg, "codec", args.codec)
    _apply_if_not_none(cfg, "preset", args.preset)
    _apply_if_not_none(cfg, "profile", args.profile)
    _apply_if_not_none(cfg, "qp", args.qp)
    _apply_if_not_none(cfg, "alpha", args.alpha)

    # 4) Hard defaults (if neither config nor CLI provided them)
    cfg.data.setdefault("restorer", "none")
    cfg.data.setdefault("gpu_id", 0)
    cfg.data.setdefault("batch_size", 8)
    cfg.data.setdefault("codec", "hevc")
    cfg.data.setdefault("preset", "P7")
    cfg.data.setdefault("profile", "main")
    cfg.data.setdefault("qp", 23)
    cfg.data.setdefault("alpha", 255)
    cfg.data.setdefault("det_conf", 0.25)
    cfg.data.setdefault("det_iou", 0.45)
    cfg.data.setdefault("det_imgsz", 640)
    cfg.data.setdefault("use_seg_masks", True)

    # Visualization defaults
    if cfg.get("visualization", default=None) is None:
        cfg.data["visualization"] = {}
    cfg.set("visualization", "box_color", value=cfg.get("visualization", "box_color", default="0,255,0"))
    cfg.set("visualization", "box_thickness", value=cfg.get("visualization", "box_thickness", default=2))
    # fill_color default: None (outline-only unless set)
    if cfg.get("visualization", "fill_opacity", default=None) is None:
        cfg.set("visualization", "fill_opacity", value=0.5)

    return cfg
