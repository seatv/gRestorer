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
    p.add_argument(
        "--mode",
        choices=["real", "pseudo", "none"],
        default=None,
        help="real=restore, pseudo=overlay, none=passthrough",
    )
    p.add_argument(
        "--restorer",
        choices=["basicvsrpp", "lada", "pseudo", "none"],
        default=None,
        help="Restorer backend (default: basicvsrpp)",
    )

    p.add_argument("--max-frames", type=int, default=None, help="Process at most N frames (debug)")

    # GPU selection (applies to decoder/encoder unless overridden)
    p.add_argument("--gpu-id", type=int, default=None, help="GPU index (decoder/encoder/inference)")

    # --- Root knobs ---
    p.add_argument("--roi-dilate", type=int, default=None, help="Dilate detected ROIs by N pixels")
    p.add_argument("--batch-size", type=int, default=None, help="Decode/processing batch size")
    p.add_argument(
        "--use-seg-masks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use segmentation masks when available",
    )

    # --- Decoder ---
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
    p.add_argument("--enc-mode", choices=["hq", "preview", "archive", "custom"], default=None, help="Encoder preset mode")
    p.add_argument("--enc-options", default=None, help="FFmpeg-style NVENC options string (e.g. \"-rc constqp -qp 18 -spatial_aq 1\")")
    p.add_argument("--enc-opt", action="append", default=None, help="NVENC option KEY=VALUE (repeatable)")
    p.add_argument("--enc-allow-unknown", action=argparse.BooleanOptionalAction, default=None, help="Allow passing unknown NVENC option keys (best-effort)")

    # --- Remux / muxing ---
    p.add_argument("--mux-audio", choices=["auto", "copy", "aac", "none"], default=None, help="Audio mux policy")
    p.add_argument("--mux-keep-subs", action=argparse.BooleanOptionalAction, default=None, help="Keep subtitles (mkv: copy; mp4: mov_text)")
    p.add_argument("--mux-extra-args", default=None, help="Extra ffmpeg args appended to remux step (must not include -i)")
    p.add_argument("--mp4-fast-start", action=argparse.BooleanOptionalAction, default=None, help="MP4 faststart (+faststart)")

    # --- Detection ---
    p.add_argument("--det-type", default=None, choices=["yolo", "lada-yolo"])
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

    # --- Scene tracking (kept for config parity) ---
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

    # --- SBS ---
    p.add_argument("--sbs", action="store_true", help="Enable side-by-side (SBS) handling")
    p.add_argument("--no-sbs", action="store_true", help="Disable SBS handling")
    p.add_argument("--sbs-layout", choices=["lr", "rl"], default=None, help="SBS layout: lr=left|right, rl=right|left")
    p.add_argument("--sbs-det-split", action="store_true", help="Run detector per-half (better per-eye)")
    p.add_argument("--no-sbs-det-split", action="store_true", help="Disable per-half detection")

    # --- Runtime debug flags (not config.json leaf keys) ---
    p.add_argument("--debug", action="store_true", help="Verbose debug logging")
    p.add_argument("--profile-sync", action="store_true", help="torch synchronize() around timings")

    return p


def _set_if_not_none(cfg: Config, keys: Sequence[str], value: Any) -> None:
    if value is None:
        return
    cfg.set(*keys, value=value)


def _load_config_json(path: Path) -> dict[str, Any]:
    obj = Config.load_json(path)
    if not isinstance(obj, dict):
        raise ValueError("config.json root must be an object/dict")
    return obj


def parse_args(argv: list[str] | None = None) -> Config:
    p = create_parser()
    args = p.parse_args(argv)

    cfg_path: Optional[Path]
    if args.config:
        cfg_path = Path(args.config)
    else:
        cfg_path = _default_config_path()

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
    if cfg.get("restorer", default=None) is None:
        cfg.set("restorer", value="basicvsrpp")

    # High-level overrides
    _set_if_not_none(cfg, ("mode",), args.mode)
    _set_if_not_none(cfg, ("restorer",), args.restorer)

    # Global GPU id
    if args.gpu_id is not None:
        cfg.set("decoder", "gpu_id", value=int(args.gpu_id))
        cfg.set("encoder", "gpu_id", value=int(args.gpu_id))

    # Root knobs
    _set_if_not_none(cfg, ("roi_dilate",), args.roi_dilate)
    _set_if_not_none(cfg, ("batch_size",), args.batch_size)
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

    # Detection
    _set_if_not_none(cfg, ("detection", "model_path"), args.det_model)
    _set_if_not_none(cfg, ("detection", "batch_size"), args.det_batch_size)
    _set_if_not_none(cfg, ("detection", "conf_threshold"), args.det_conf)
    _set_if_not_none(cfg, ("detection", "iou_threshold"), args.det_iou)
    _set_if_not_none(cfg, ("detection", "imgsz"), args.det_imgsz)
    _set_if_not_none(cfg, ("detection", "fp16"), args.det_fp16)

    # Restoration
    _set_if_not_none(cfg, ("restoration", "rest_model_path"), args.rest_model)
    _set_if_not_none(cfg, ("restoration", "fp16"), args.rest_fp16)
    _set_if_not_none(cfg, ("restoration", "max_clip_length"), args.rest_max_clip_length)
    _set_if_not_none(cfg, ("restoration", "clip_size"), args.rest_clip_size)
    _set_if_not_none(cfg, ("restoration", "border_ratio"), args.rest_border_ratio)
    _set_if_not_none(cfg, ("restoration", "pad_mode"), args.rest_pad_mode)
    _set_if_not_none(cfg, ("restoration", "feather_radius"), args.rest_feather_radius)

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

    # Detector Switch (CLI overrides config/default)
    if args.det_type is not None:
        cfg.set("detection", "dete_type", value=str(args.det_type))

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

    if mode in ("real", "pseudo"):
        det_s = str(cfg.get("detection", "model_path", default="") or "").strip()
        if not det_s:
            raise FileNotFoundError("Detector model path is empty (check config.json or --det-model)")
        det_path = Path(det_s)
        if not det_path.exists():
            raise FileNotFoundError(f"Detector model not found: {det_path}")

    if mode == "real" and restorer in ("basicvsrpp", "real_basicvsrpp"):
        rest_s = str(cfg.get("restoration", "rest_model_path", default="") or "").strip()
        if not rest_s:
            raise FileNotFoundError("Restoration model path is empty (check config.json or --rest-model)")
        rest_path = Path(rest_s)
        if not rest_path.exists():
            raise FileNotFoundError(f"Restoration model not found: {rest_path}")

    return cfg
