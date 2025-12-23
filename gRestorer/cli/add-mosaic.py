# gRestorer/cli/mosaic_cli.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from gRestorer.video.decoder import Decoder
from gRestorer.video.encoder import Encoder

# Reuse the same zero-copy + colorspace helpers your pipeline uses.
from gRestorer.cli.pipeline_utils import (
    wrap_surface_as_tensor,
    rgb_to_bgr_u8_inplace,
    pack_bgr_u8_to_bgra_u8_inplace,
)

Box = Tuple[int, int, int, int]  # (t,l,b,r)


def _parse_roi(s: str) -> Box:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be 't,l,b,r'")
    try:
        t, l, b, r = map(int, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be integers") from exc
    return (t, l, b, r)


def _cfg_get(d: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _infer_video_props(dec: Decoder) -> Tuple[int, int, float]:
    """
    Robustly get (width,height,fps) from your Decoder without guessing exact attribute names.
    """
    # Common direct attrs
    for w_key, h_key, fps_key in [
        ("width", "height", "fps"),
        ("video_width", "video_height", "video_fps"),
    ]:
        if hasattr(dec, w_key) and hasattr(dec, h_key) and hasattr(dec, fps_key):
            return int(getattr(dec, w_key)), int(getattr(dec, h_key)), float(getattr(dec, fps_key))

    # metadata object/dict
    if hasattr(dec, "metadata"):
        md = getattr(dec, "metadata")
        if isinstance(md, dict):
            w = md.get("width") or md.get("video_width")
            h = md.get("height") or md.get("video_height")
            fps = md.get("fps") or md.get("video_fps")
            if w and h and fps:
                return int(w), int(h), float(fps)
        else:
            # object
            for w_key, h_key, fps_key in [
                ("width", "height", "fps"),
                ("video_width", "video_height", "video_fps"),
            ]:
                if hasattr(md, w_key) and hasattr(md, h_key) and hasattr(md, fps_key):
                    return int(getattr(md, w_key)), int(getattr(md, h_key)), float(getattr(md, fps_key))

    raise RuntimeError(
        "Unable to infer width/height/fps from Decoder. "
        "Please add properties on Decoder or expose a .metadata with width/height/fps."
    )


def _clamp_roi_inclusive(t: int, l: int, b: int, r: int, h: int, w: int) -> Optional[Box]:
    """
    Treat ROI as inclusive (t,l,b,r) like LADA Boxes.
    """
    t2 = max(0, min(h - 1, t))
    b2 = max(0, min(h - 1, b))
    l2 = max(0, min(w - 1, l))
    r2 = max(0, min(w - 1, r))
    if b2 < t2 or r2 < l2:
        return None
    return (t2, l2, b2, r2)


@torch.inference_mode()
def pixelate_roi_bgr_u8_inplace(frame_bgr: torch.Tensor, roi: Box, block: int) -> None:
    """
    Pixelate one ROI in-place on GPU.
    frame_bgr: uint8 [H,W,3]
    roi: inclusive (t,l,b,r)
    block: mosaic block size in pixels (>=2 recommended)
    """
    if frame_bgr.dtype != torch.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[-1] != 3:
        raise ValueError(f"frame_bgr must be uint8 [H,W,3], got {frame_bgr.dtype} {tuple(frame_bgr.shape)}")

    h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    t, l, b, r = roi
    roi2 = _clamp_roi_inclusive(t, l, b, r, h=h, w=w)
    if roi2 is None:
        return
    t, l, b, r = roi2

    # inclusive -> python slice end is +1
    patch = frame_bgr[t : b + 1, l : r + 1, :]  # [ph,pw,3]
    ph, pw = int(patch.shape[0]), int(patch.shape[1])
    if ph <= 1 or pw <= 1:
        return

    block = max(1, int(block))
    # Downsample size: roughly one sample per block
    sh = max(1, (ph + block - 1) // block)
    sw = max(1, (pw + block - 1) // block)

    x = patch.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)  # [1,3,ph,pw]
    small = F.interpolate(x, size=(sh, sw), mode="area")
    up = F.interpolate(small, size=(ph, pw), mode="nearest")
    out = up.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)  # [ph,pw,3]

    patch.copy_(out)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="gRestorer-mosaic",
        description="GPU-only synthetic mosaic generator (3+ ROIs) for SFW debugging.",
    )
    default_cfg = Path("config.json")
    ap.add_argument("--config", type=str, default=str(default_cfg) if default_cfg.is_file() else None, help="Config path (defaults to ./config.json if present)")
    ap.add_argument("--input", required=True, help="Input video")
    ap.add_argument("--output", required=True, help="Output mosaiced video")

    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)

    ap.add_argument("--roi", action="append", type=_parse_roi, default=[], help="ROI 't,l,b,r' (inclusive). Repeat 3x.")
    ap.add_argument("--block", type=int, default=None, help="Mosaic block size (default 16 or config synth_mosaic.block_size)")

    # Encoder overrides (optional)
    ap.add_argument("--codec", type=str, default=None)
    ap.add_argument("--preset", type=str, default=None)
    ap.add_argument("--profile", type=str, default=None)
    ap.add_argument("--qp", type=int, default=None)
    ap.add_argument("--container", type=str, default=None)
    ap.add_argument("--bframes", type=int, default=None)
    ap.add_argument("--ffmpeg-path", type=str, default=None)

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    gpu_id = int(args.gpu_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = _cfg_get(cfg, ("detection", "batch_size"), None)
    if batch_size is None:
        batch_size = 80  # matches your Decoder default
    batch_size = int(batch_size)

    block = args.block
    if block is None:
        block = _cfg_get(cfg, ("synth_mosaic", "block_size"), None)
    if block is None:
        block = 16
    block = int(block)

    rois: List[Box] = list(args.roi)
    if not rois:
        rois_cfg = _cfg_get(cfg, ("synth_mosaic", "rois"), [])
        if isinstance(rois_cfg, list):
            for item in rois_cfg:
                if isinstance(item, list) and len(item) == 4:
                    rois.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))

    if not rois:
        raise SystemExit("No ROIs provided. Use --roi t,l,b,r (repeat) or config synth_mosaic.rois.")

    # Encoder settings: config -> CLI override
    codec = args.codec or _cfg_get(cfg, ("encoder", "codec"), "hevc")
    preset = args.preset or _cfg_get(cfg, ("encoder", "preset"), "P7")
    profile = args.profile or _cfg_get(cfg, ("encoder", "profile"), "main")

    qp = args.qp
    if qp is None:
        qp = _cfg_get(cfg, ("encoder", "qp"), 23)
    qp = int(qp)

    container = args.container
    if container is None:
        container = _cfg_get(cfg, ("encoder", "container"), None)

    bframes = args.bframes
    if bframes is None:
        bframes = _cfg_get(cfg, ("encoder", "bframes"), 0)
    bframes = int(bframes)

    ffmpeg_path = args.ffmpeg_path or _cfg_get(cfg, ("encoder", "ffmpeg_path"), "ffmpeg")

    decoder = Decoder(args.input, gpu_id=gpu_id, batch_size=batch_size)
    width, height, fps = _infer_video_props(decoder)

    encoder = Encoder(
        args.output,
        width=width,
        height=height,
        fps=float(fps),
        codec=str(codec),
        preset=str(preset),
        profile=str(profile),
        qp=int(qp),
        gpu_id=gpu_id,
        container=container,
        bframes=bframes,
        ffmpeg_path=str(ffmpeg_path),
    )

    frames_done = 0
    try:
        while True:
            if args.max_frames is not None and frames_done >= int(args.max_frames):
                break

            surfaces = decoder.read_batch()
            if not surfaces:
                break

            bgra_batch: List[torch.Tensor] = []
            for s in surfaces:
                rgb = wrap_surface_as_tensor(s)  # HWC RGB or CHW RGBP, uint8

                # Convert -> BGR uint8 HWC
                bgr = torch.empty((height, width, 3), device=rgb.device, dtype=torch.uint8)
                rgb_to_bgr_u8_inplace(bgr, rgb)

                # Mosaic all ROIs
                for roi in rois:
                    pixelate_roi_bgr_u8_inplace(bgr, roi=roi, block=block)

                # Pack -> BGRA uint8 HWC
                bgra = torch.empty((height, width, 4), device=bgr.device, dtype=torch.uint8)
                pack_bgr_u8_to_bgra_u8_inplace(bgra, bgr)

                bgra_batch.append(bgra)
                frames_done += 1
                if args.max_frames is not None and frames_done >= int(args.max_frames):
                    break

            # Encode the batch
            encoder.encode_frames(bgra_batch)

    finally:
        encoder.flush()
        encoder.close()
        decoder.close()

    print(f"[Mosaic] done frames={frames_done} block={block} rois={len(rois)}")


if __name__ == "__main__":
    main()
