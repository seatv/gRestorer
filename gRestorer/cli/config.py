# gRestorer/cli/config.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from gRestorer.utils.config_util import Config


def _parse_rgb_triplet(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected R,G,B (three comma-separated ints)")
    try:
        r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid R,G,B triplet: {s!r}") from e
    for v in (r, g, b):
        if v < 0 or v > 255:
            raise argparse.ArgumentTypeError("Color values must be 0..255")
    return r, g, b


def _parse_ext_list(s: str) -> list[str]:
    out: list[str] = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        if not p.startswith("."):
            p = "." + p
        out.append(p.lower())
    return out


def _default_config_path() -> Optional[Path]:
    cwd = Path.cwd() / "config.json"
    if cwd.exists():
        return cwd

    here = Path(__file__).resolve()
    for up in (2, 3, 1):
        try:
            candidate = here.parents[up] / "config.json"
            if candidate.exists():
                return candidate
        except Exception:
            pass
    return None


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gRestorer", description="GPU-centric video mosaic remover")

    # Required I/O
    p.add_argument("--input", required=True, help="Input video file")
    p.add_argument("--output", required=True, help="Output video file")

    # Config + high-level mode
    p.add_argument("--config", default=None, help="Path to config.json (defaults to nearest config.json)")
    p.add_argument("--mode", choices=["real", "pseudo", "none"], default=None, help="real=restore, pseudo=overlay, none=passthrough")
    p.add_argument("--restorer", choices=["basicvsrpp", "lada", "pseudo", "none", "face_swap"], default=None, help="Restorer backend (default: basicvsrpp)")
    p.add_argument("--max-frames", type=int, default=None, help="Process at most N frames (debug)")
    p.add_argument("--process", choices=["mosaic", "face"], default=None, help="Select processing family: mosaic or face (defaults to mosaic)")
    p.add_argument("--debug-enabled", action=argparse.BooleanOptionalAction, default=None, help="Enable runtime debug behavior from config/CLI")

    # GPU selection (applies to decoder/encoder unless overridden)
    p.add_argument("--gpu-id", type=int, default=None, help="GPU index (decoder/encoder/inference)")

    # --- Root knobs ---
    p.add_argument("--roi-dilate", type=int, default=None, help="Dilate detected ROIs by N pixels")
    p.add_argument("--batch-size", type=int, default=None, help="Decode/processing batch size")
    p.add_argument("--use-seg-masks", action=argparse.BooleanOptionalAction, default=None, help="Use segmentation masks when available")
    p.add_argument("--dec-gpu-id", type=int, default=None)
    p.add_argument("--dec-output-format", choices=["RGB", "RGBP"], default=None)
    p.add_argument("--dec-ffmpeg-input-args", default=None, help="Extra ffmpeg input args for CPU decode fallback (inserted before -i; must not contain -i)")

    # --- Encoder base ---
    p.add_argument("--enc-codec", choices=["hevc", "h264"], default=None)
    p.add_argument("--enc-preset", default=None)
    p.add_argument("--enc-profile", default=None)
    p.add_argument("--enc-qp", type=int, default=None)
    p.add_argument("--enc-format", default=None)
    p.add_argument("--enc-gpu-id", type=int, default=None)
    p.add_argument("--enc-sync-before-encode", action=argparse.BooleanOptionalAction, default=None)

    # --- Encoder advanced (NVENC knobs) ---
    p.add_argument("--enc-mode", choices=["default", "hq", "preview", "archive", "analysis", "custom"], default=None, help="Encoder preset mode")
    p.add_argument("--enc-options", default=None, help="FFmpeg-style NVENC options string")
    p.add_argument("--enc-opt", action="append", default=None, help="NVENC option KEY=VALUE (repeatable)")
    p.add_argument("--enc-allow-unknown", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--mux-audio", choices=["auto", "copy", "aac", "none"], default=None)
    p.add_argument("--mux-keep-subs", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--mux-extra-args", default=None)
    p.add_argument("--mp4-fast-start", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--det-type", default=None, choices=["yolo", "lada-yolo", "face"])
    p.add_argument("--det-model", default=None)
    p.add_argument("--det-batch-size", type=int, default=None)
    p.add_argument("--det-conf", type=float, default=None)
    p.add_argument("--det-iou", type=float, default=None)
    p.add_argument("--det-imgsz", type=int, default=None)
    p.add_argument("--det-fp16", action=argparse.BooleanOptionalAction, default=None)

    # --- Restoration ---
    p.add_argument("--rest-model", default=None)
    p.add_argument("--rest-fp16", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--rest-max-clip-length", type=int, default=None)
    p.add_argument("--rest-clip-size", type=int, default=None)
    p.add_argument("--rest-border-ratio", type=float, default=None)
    p.add_argument("--rest-pad-mode", default=None)
    p.add_argument("--rest-feather-radius", type=int, default=None)
    p.add_argument("--rest-blendmask", choices=["none", "facefusion", "laplacian"], default=None)
    p.add_argument("--laplacian", action="store_true", help="Use laplacian pyramid compositor (sets --rest-blendmask laplacian)")
    p.add_argument("--source-face", default=None, help="Source/reference face image for face_swap restorer")
    p.add_argument("--swap-model", default=None, help="ONNX face swap model path")
    p.add_argument("--swap-input-size", type=int, default=None)
    p.add_argument("--swap-provider", choices=["auto", "cuda", "cpu"], default=None)
    p.add_argument("--swap-backend", choices=["auto", "inswapper", "simswap", "hyperswap"], default=None, help="Face swap backend override (default: auto by model path)")
    p.add_argument("--face-enhancer-model", default=None, help="Optional ONNX face enhancer model path (GFPGAN-like single-input model)")
    p.add_argument("--face-enhancer-blend", type=int, default=None, help="Optional face enhancer blend 0..100")
    p.add_argument("--rest-compositor-quantize-before-resize", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--rest-compositor-resize-backend", choices=["torch", "image_utils"], default=None)
    p.add_argument("--analysis-use-synth-rois", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--store-max-frames", type=int, default=None)
    p.add_argument("--trk-min-iou", type=float, default=None)
    p.add_argument("--trk-max-clip-frames", type=int, default=None)
    p.add_argument("--trk-min-clip-frames", type=int, default=None)

    # --- Visualization (pseudo / debug overlays) ---
    p.add_argument("--vis-box-color", type=_parse_rgb_triplet, default=None)
    p.add_argument("--vis-box-thickness", type=int, default=None)
    p.add_argument("--vis-show-confidence", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--vis-show-class", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--vis-fill-color", type=_parse_rgb_triplet, default=None)
    p.add_argument("--vis-fill-opacity", type=float, default=None)

    # --- Batch processing ---
    p.add_argument("--batch-video-extensions", type=_parse_ext_list, default=None)
    p.add_argument("--batch-skip-existing", action=argparse.BooleanOptionalAction, default=None)

    # --- Debug section in config.json ---
    p.add_argument("--debug-save-detection-frames", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--debug-save-detection-json", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--debug-output-dir", default=None)
    p.add_argument("--sbs", action="store_true")
    p.add_argument("--no-sbs", action="store_true")
    p.add_argument("--sbs-layout", choices=["lr", "rl"], default=None)
    p.add_argument("--sbs-det-split", action="store_true")
    p.add_argument("--no-sbs-det-split", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--profile-sync", action="store_true")
    return p


def _set_if_not_none(cfg: Config, keys: Sequence[str], value: Any) -> None:
    if value is None:
        return
    cfg.set(*keys, value=value)


def _set_many_if_not_none(cfg: Config, keys_list: Sequence[Sequence[str]], value: Any) -> None:
    if value is None:
        return
    for keys in keys_list:
        cfg.set(*keys, value=value)


def _load_config_json(path: Path) -> dict[str, Any]:
    obj = Config.load_json(path)
    if not isinstance(obj, dict):
        raise ValueError("config.json root must be an object/dict")
    return obj


def _cfg_first(cfg: Config, key_paths: Sequence[Sequence[str]], default: Any = None) -> Any:
    for keys in key_paths:
        v = cfg.get(*keys, default=None)
        if v is not None:
            return v
    return default


def parse_args(argv: list[str] | None = None) -> Config:
    p = create_parser()
    args = p.parse_args(argv)
    cfg_path = Path(args.config) if args.config else _default_config_path()
    cfg = Config({})

    if cfg_path is not None:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        cfg.merge_dict(_load_config_json(cfg_path))

    # Required basics
    cfg.set("input", value=str(args.input))
    cfg.set("output", value=str(args.output))
    _set_if_not_none(cfg, ("max_frames",), args.max_frames)

    # Defaults
    if cfg.get("mode", default=None) is None:
        cfg.set("mode", value="real")
    if cfg.get("process", default=None) is None:
        cfg.set("process", value="mosaic")
    if cfg.get("debug_enabled", default=None) is None:
        cfg.set("debug_enabled", value=False)
    if cfg.get("restoration", "blendmask", default=None) is None:
        cfg.set("restoration", "blendmask", value="none")

    # High-level overrides
    _set_if_not_none(cfg, ("mode",), args.mode)
    _set_if_not_none(cfg, ("process",), args.process)
    _set_if_not_none(cfg, ("restorer",), args.restorer)
    _set_if_not_none(cfg, ("debug_enabled",), args.debug_enabled)

    # Default restorer follows process when not explicitly specified.
    if cfg.get("restorer", default=None) is None:
        process_default = str(cfg.get("process", default="mosaic") or "mosaic").lower()
        cfg.set("restorer", value=("face_swap" if process_default == "face" else "basicvsrpp"))

    # Global GPU id
    if args.gpu_id is not None:
        cfg.set("runtime", "gpu_id", value=int(args.gpu_id))
        cfg.set("decoder", "gpu_id", value=int(args.gpu_id))
        cfg.set("encoder", "gpu_id", value=int(args.gpu_id))

    # Root knobs
    _set_if_not_none(cfg, ("roi_dilate",), args.roi_dilate)
    _set_if_not_none(cfg, ("batch_size",), args.batch_size)
    _set_if_not_none(cfg, ("runtime", "batch_size"), args.batch_size)
    _set_if_not_none(cfg, ("use_seg_masks",), args.use_seg_masks)

    # Decoder
    _set_if_not_none(cfg, ("decoder", "gpu_id"), args.dec_gpu_id)
    _set_if_not_none(cfg, ("decoder", "output_format"), args.dec_output_format)
    _set_if_not_none(cfg, ("decoder", "ffmpeg_input_args"), args.dec_ffmpeg_input_args)

    # Encoder base
    _set_if_not_none(cfg, ("encoder", "codec"), args.enc_codec)
    _set_if_not_none(cfg, ("encoder", "preset"), args.enc_preset)
    _set_if_not_none(cfg, ("encoder", "profile"), args.enc_profile)
    _set_if_not_none(cfg, ("encoder", "qp"), args.enc_qp)
    _set_if_not_none(cfg, ("encoder", "format"), args.enc_format)
    _set_if_not_none(cfg, ("encoder", "gpu_id"), args.enc_gpu_id)
    _set_if_not_none(cfg, ("encoder", "sync_before_encode"), args.enc_sync_before_encode)

    # Encoder advanced
    _set_if_not_none(cfg, ("encoder", "mode"), args.enc_mode)
    _set_if_not_none(cfg, ("encoder", "options"), args.enc_options)
    _set_if_not_none(cfg, ("encoder", "allow_unknown"), args.enc_allow_unknown)

    if args.enc_opt:
        # Merge into existing encoder.opt dict
        cur = cfg.get("encoder", "opt", default={}) or {}
        if not isinstance(cur, dict):
            cur = {}
        merged = dict(cur)
        for item in args.enc_opt:
            if "=" not in item:
                raise ValueError(f"--enc-opt expects KEY=VALUE, got: {item!r}")
            k, v = item.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                raise ValueError(f"--enc-opt invalid key in: {item!r}")
            merged[k] = v
        cfg.set("encoder", "opt", value=merged)

    # Remux / muxing
    _set_if_not_none(cfg, ("encoder", "mux_audio"), args.mux_audio)
    _set_if_not_none(cfg, ("encoder", "mux_keep_subs"), args.mux_keep_subs)
    _set_if_not_none(cfg, ("encoder", "mux_extra_args"), args.mux_extra_args)
    _set_if_not_none(cfg, ("encoder", "mp4_faststart"), args.mp4_fast_start)

    _set_many_if_not_none(cfg, [("detection", "det_type"), ("mosaic_detection", "det_type"), ("face_detection", "det_type")], args.det_type)
    _set_many_if_not_none(cfg, [("detection", "model_path"), ("mosaic_detection", "model_path"), ("face_detection", "model_path")], args.det_model)
    _set_many_if_not_none(cfg, [("detection", "batch_size"), ("mosaic_detection", "batch_size"), ("face_detection", "batch_size")], args.det_batch_size)
    _set_many_if_not_none(cfg, [("detection", "conf_threshold"), ("mosaic_detection", "conf_threshold"), ("face_detection", "conf_threshold")], args.det_conf)
    _set_many_if_not_none(cfg, [("detection", "iou_threshold"), ("mosaic_detection", "iou_threshold"), ("face_detection", "iou_threshold")], args.det_iou)
    _set_many_if_not_none(cfg, [("detection", "imgsz"), ("mosaic_detection", "imgsz"), ("face_detection", "imgsz")], args.det_imgsz)
    _set_many_if_not_none(cfg, [("detection", "fp16"), ("mosaic_detection", "fp16"), ("face_detection", "fp16")], args.det_fp16)

    # Restoration
    _set_many_if_not_none(cfg, [("restoration", "rest_model_path"), ("mosaic_restoration", "rest_model_path")], args.rest_model)
    _set_many_if_not_none(cfg, [("restoration", "fp16"), ("mosaic_restoration", "fp16")], args.rest_fp16)
    _set_many_if_not_none(cfg, [("restoration", "max_clip_length"), ("mosaic_restoration", "max_clip_length"), ("face_restoration", "max_clip_length")], args.rest_max_clip_length)
    _set_many_if_not_none(cfg, [("restoration", "clip_size"), ("mosaic_restoration", "clip_size"), ("face_restoration", "clip_size")], args.rest_clip_size)
    _set_many_if_not_none(cfg, [("restoration", "border_ratio"), ("mosaic_restoration", "border_ratio"), ("face_restoration", "border_ratio")], args.rest_border_ratio)
    _set_many_if_not_none(cfg, [("restoration", "pad_mode"), ("mosaic_restoration", "pad_mode"), ("face_restoration", "pad_mode")], args.rest_pad_mode)
    _set_many_if_not_none(cfg, [("restoration", "feather_radius"), ("mosaic_restoration", "feather_radius")], args.rest_feather_radius)
    _set_many_if_not_none(cfg, [("restoration", "blendmask"), ("mosaic_restoration", "blendmask")], args.rest_blendmask)
    if args.laplacian:
        cfg.set("restoration", "blendmask", value="laplacian")
        cfg.set("mosaic_restoration", "blendmask", value="laplacian")
    _set_many_if_not_none(cfg, [("restoration", "source_face_path"), ("face_restoration", "source_face_path")], args.source_face)
    _set_many_if_not_none(cfg, [("restoration", "swap_model_path"), ("face_restoration", "swap_model_path")], args.swap_model)
    _set_many_if_not_none(cfg, [("restoration", "swap_input_size"), ("face_restoration", "swap_input_size")], args.swap_input_size)
    _set_many_if_not_none(cfg, [("restoration", "swap_provider"), ("face_restoration", "provider")], args.swap_provider)
    _set_many_if_not_none(cfg, [("restoration", "swap_backend"), ("face_restoration", "swap_backend")], args.swap_backend)
    _set_many_if_not_none(cfg, [("restoration", "face_enhancer_model_path"), ("enhancement", "model_path")], args.face_enhancer_model)
    _set_many_if_not_none(cfg, [("restoration", "face_enhancer_blend"), ("enhancement", "blend")], args.face_enhancer_blend)
    if args.face_enhancer_model is not None:
        cfg.set("enhancement", "enabled", value=bool(str(args.face_enhancer_model).strip()))
    _set_if_not_none(cfg, ("restoration", "compositor_quantize_before_resize"), args.rest_compositor_quantize_before_resize)
    _set_if_not_none(cfg, ("restoration", "compositor_resize_backend"), args.rest_compositor_resize_backend)
    _set_if_not_none(cfg, ("restoration", "analysis_use_synth_rois"), args.analysis_use_synth_rois)

    # [CHANGE 2] FrameStore backpressure
    _set_if_not_none(cfg, ("store_max_frames",), args.store_max_frames)
    _set_if_not_none(cfg, ("runtime", "store_max_frames"), args.store_max_frames)

    # Scene tracking
    _set_if_not_none(cfg, ("scene_tracking", "min_iou"), args.trk_min_iou)
    _set_if_not_none(cfg, ("scene_tracking", "max_clip_frames"), args.trk_max_clip_frames)
    _set_if_not_none(cfg, ("scene_tracking", "min_clip_frames"), args.trk_min_clip_frames)

    # Visualization
    _set_if_not_none(cfg, ("visualization", "box_color"), list(args.vis_box_color) if args.vis_box_color is not None else None)
    _set_if_not_none(cfg, ("visualization", "box_thickness"), args.vis_box_thickness)
    _set_if_not_none(cfg, ("visualization", "show_confidence"), args.vis_show_confidence)
    _set_if_not_none(cfg, ("visualization", "show_class"), args.vis_show_class)
    _set_if_not_none(cfg, ("visualization", "fill_color"), list(args.vis_fill_color) if args.vis_fill_color is not None else None)
    _set_if_not_none(cfg, ("visualization", "fill_opacity"), args.vis_fill_opacity)

    # Batch processing
    _set_if_not_none(cfg, ("batch_processing", "video_extensions"), args.batch_video_extensions)
    _set_if_not_none(cfg, ("batch_processing", "skip_existing"), args.batch_skip_existing)

    # Debug section
    _set_if_not_none(cfg, ("debug", "save_detection_frames"), args.debug_save_detection_frames)
    _set_if_not_none(cfg, ("debug", "save_detection_json"), args.debug_save_detection_json)
    _set_if_not_none(cfg, ("debug", "output_dir"), args.debug_output_dir)

    # SBS
    if args.sbs:
        cfg.set("sbs_enabled", value=True)
    if args.no_sbs:
        cfg.set("sbs_enabled", value=False)
    _set_if_not_none(cfg, ("sbs_layout",), args.sbs_layout)
    if args.sbs_det_split:
        cfg.set("sbs_det_split", value=True)
    if args.no_sbs_det_split:
        cfg.set("sbs_det_split", value=False)

    # Runtime-only toggles
    if args.debug:
        cfg.set("debug_enabled", value=True)
    if args.profile_sync:
        cfg.set("profile_sync", value=True)

    # Validation
    inp = Path(str(cfg.get("input", default="")))
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    mode = str(cfg.get("mode", default="real")).lower()
    restorer = str(cfg.get("restorer", default="basicvsrpp")).lower()
    process = str(cfg.get("process", default="mosaic") or "mosaic").lower()
    if process not in ("mosaic", "face"):
        raise ValueError(f"Invalid process: {process!r}. Expected 'mosaic' or 'face'.")

    if restorer == "face_swap" and process != "face":
        raise ValueError("restorer=face_swap requires process=face. Unless explicitly specified, gRestorer operates on mosaic restoration.")

    if mode in ("real", "pseudo"):
        if process == "face":
            det_type = str(_cfg_first(cfg, [("face_detection", "det_type"), ("detection", "det_type")], default="face")).lower()
            det_s = str(_cfg_first(cfg, [("face_detection", "model_path"), ("detection", "model_path")], default="") or "").strip()
        else:
            det_type = str(_cfg_first(cfg, [("mosaic_detection", "det_type"), ("detection", "det_type")], default="yolo")).lower()
            det_s = str(_cfg_first(cfg, [("mosaic_detection", "model_path"), ("detection", "model_path")], default="") or "").strip()
        if not det_s:
            raise FileNotFoundError("Detector model path is empty (check config.json or --det-model)")
        det_path = Path(det_s)
        if not det_path.exists():
            raise FileNotFoundError(f"Detector model not found: {det_path}")

    if mode == "real" and process == "mosaic" and restorer in ("basicvsrpp", "real_basicvsrpp", "lada"):
        rest_s = str(_cfg_first(cfg, [("mosaic_restoration", "rest_model_path"), ("restoration", "rest_model_path")], default="") or "").strip()
        if not rest_s:
            raise FileNotFoundError("Restoration model path is empty (check config.json or --rest-model)")
        rest_path = Path(rest_s)
        if not rest_path.exists():
            raise FileNotFoundError(f"Restoration model not found: {rest_path}")

    if mode == "real" and process == "face" and restorer == "face_swap":
        source_face_s = str(_cfg_first(cfg, [("face_restoration", "source_face_path"), ("restoration", "source_face_path")], default="") or "").strip()
        if not source_face_s:
            raise FileNotFoundError("Source face path is empty (check config.json or --source-face)")
        source_face_path = Path(source_face_s)
        if not source_face_path.exists():
            raise FileNotFoundError(f"Source face image not found: {source_face_path}")

        swap_model_s = str(_cfg_first(cfg, [("face_restoration", "swap_model_path"), ("restoration", "swap_model_path")], default="") or "").strip()
        if not swap_model_s:
            raise FileNotFoundError("Swap model path is empty (check config.json or --swap-model)")
        swap_model_path = Path(swap_model_s)
        if not swap_model_path.exists():
            raise FileNotFoundError(f"Swap model not found: {swap_model_path}")

        face_enhancer_enabled = bool(_cfg_first(cfg, [("enhancement", "enabled")], default=False))
        face_enhancer_s = str(_cfg_first(cfg, [("enhancement", "model_path"), ("restoration", "face_enhancer_model_path")], default="") or "").strip()
        if face_enhancer_enabled and face_enhancer_s:
            face_enhancer_path = Path(face_enhancer_s)
            if not face_enhancer_path.exists():
                raise FileNotFoundError(f"Face enhancer model not found: {face_enhancer_path}")
        blend = int(_cfg_first(cfg, [("enhancement", "blend"), ("restoration", "face_enhancer_blend")], default=80))
        if blend < 0 or blend > 100:
            raise ValueError("enhancement.blend must be in 0..100")

        occluder_enabled = bool(_cfg_first(cfg, [("occlusion", "enabled")], default=False))
        occluder_model_s = str(_cfg_first(cfg, [("occlusion", "model_path")], default="") or "").strip()
        if occluder_enabled and occluder_model_s:
            occluder_model_path = Path(occluder_model_s)
            if not occluder_model_path.exists():
                raise FileNotFoundError(f"Occluder model not found: {occluder_model_path}")
        occ_blend = int(_cfg_first(cfg, [("occlusion", "blend")], default=100))
        if occ_blend < 0 or occ_blend > 100:
            raise ValueError("occlusion.blend must be in 0..100")
    return cfg
