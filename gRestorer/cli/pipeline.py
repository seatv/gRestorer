# gRestorer/cli/pipeline.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from dataclasses import dataclass, field


from gRestorer.core.scene_tracker import SceneTracker, TrackerConfig
from gRestorer.detector.core import Detection, Detector as YoloDetector
from gRestorer.restorer.basicvsrpp_clip_restorer import BasicVSRPPClipRestorer
from gRestorer.restorer.compositor import _composite_clip_into_store
from gRestorer.restorer.pseudo_clip_restorer import PseudoClipRestorer
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
    drain_store_to_encoder,
    rgb_hwc_to_bgr_hwc_u8,
    rgbp_chw_to_rgb_hwc_u8,
    seam_split_boxes,
    split_frame_lr,
    sync_device,
    unsplit_boxes_layout,
    unsplit_masks_layout,
    wrap_surface_as_tensor,
)

import datetime as _dt
from dataclasses import dataclass

# Box = (t,l,b,r) inclusive (matches your pipeline_utils.Box)
# If Box isn't in scope here, import it:
# from .pipeline_utils import Box


@dataclass
class DetStats:
    frames_total: int = 0
    frames_with_det: int = 0
    total_boxes: int = 0
    total_roi_area_px: float = 0.0  # sum of box areas (not union), accumulated over frames with det
    frame_area_px: int = 0

    def add(self, boxes, w: int, h: int) -> None:
        self.frames_total += 1
        if self.frame_area_px == 0:
            self.frame_area_px = int(w) * int(h)

        if not boxes:
            return

        self.frames_with_det += 1
        self.total_boxes += len(boxes)

        # area = sum of box areas (ignores overlap; fast + stable)
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

    wall_start: _dt.datetime | None = None
    wall_end: _dt.datetime | None = None

    det_stats: DetStats = field(default_factory=DetStats)

    def sum_parts(self) -> float:
        return self.t_decode + self.t_det + self.t_track + self.t_restore + self.t_encode



def _pick_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():  # type: ignore[attr-defined]
        return torch.device(f"xpu:{gpu_id}")
    return torch.device("cpu")


def _tensor_boxes_to_list_xyxy(boxes_xyxy: Optional[torch.Tensor]) -> List[Box]:
    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        return []
    out: List[Box] = []
    for row in boxes_xyxy.tolist():
        x1, y1, x2, y2 = row
        l = int(round(x1))
        t = int(round(y1))
        r = int(round(x2))
        b = int(round(y2))
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


@dataclass
class Pipeline:
    cfg: Config

    def __post_init__(self) -> None:
        self.input_path = str(self.cfg.get("input"))
        self.output_path = str(self.cfg.get("output"))
        self.max_frames: Optional[int] = self.cfg.get("max_frames", default=None)

        self.debug: bool = bool(self.cfg.get("debug_enabled", default=False))
        self.profile_sync: bool = bool(self.cfg.get("profile_sync", default=False))

        self.batch_size: int = int(self.cfg.get("batch_size", default=8))

        self.dec_gpu_id: int = int(cfg_first(self.cfg, [("decoder", "gpu_id")], default=0))
        self.enc_gpu_id: int = int(cfg_first(self.cfg, [("encoder", "gpu_id")], default=self.dec_gpu_id))
        self.device: torch.device = _pick_device(self.dec_gpu_id)

        self.mode: str = str(self.cfg.get("mode", default="real")).lower()
        self.restorer_name: str = str(self.cfg.get("restorer", default="basicvsrpp")).lower()

        self.det_model: str = cfg_path(self.cfg, ("detection", "model_path"), default="")
        self.det_imgsz: int = int(self.cfg.get("detection", "imgsz", default=640))
        self.det_conf: float = float(self.cfg.get("detection", "conf_threshold", default=0.30))
        self.det_iou: float = float(self.cfg.get("detection", "iou_threshold", default=0.70))
        self.det_fp16: bool = bool(self.cfg.get("detection", "fp16", default=True))
        self.det_batch_size: int = int(self.cfg.get("detection", "batch_size", default=self.batch_size))

        self.roi_dilate: int = int(self.cfg.get("roi_dilate", default=0))
        self.use_seg_masks: bool = bool(self.cfg.get("use_seg_masks", default=True))

        self.sbs_enabled: bool = bool(self.cfg.get("sbs_enabled", default=False))
        self.sbs_layout: str = str(self.cfg.get("sbs_layout", default="lr")).lower()
        self.sbs_det_split: bool = bool(self.cfg.get("sbs_det_split", default=False))

        self.rest_model: str = cfg_path(self.cfg, ("restoration", "rest_model_path"), default="")
        self.rest_fp16: bool = bool(self.cfg.get("restoration", "fp16", default=True))
        self.rest_max_clip_length: int = int(self.cfg.get("restoration", "max_clip_length", default=9))
        self.rest_clip_size: int = int(self.cfg.get("restoration", "clip_size", default=256))
        self.rest_border_ratio: float = float(self.cfg.get("restoration", "border_ratio", default=0.06))
        self.rest_pad_mode: str = str(self.cfg.get("restoration", "pad_mode", default="reflect"))
        self.feather_radius: int = int(self.cfg.get("restoration", "feather_radius", default=0))

        self.enc_codec: str = str(self.cfg.get("encoder", "codec", default="hevc")).lower()
        self.enc_preset: str = str(self.cfg.get("encoder", "preset", default="P6"))
        self.enc_profile: str = str(self.cfg.get("encoder", "profile", default="main"))
        self.enc_qp: int = int(self.cfg.get("encoder", "qp", default=20))
        self.enc_sync_before_encode: bool = bool(self.cfg.get("encoder", "sync_before_encode", default=True))

    def _build_detector(self) -> Optional[YoloDetector]:
        if self.mode == "none":
            return None
        if not self.det_model:
            raise FileNotFoundError("Detector model path is empty (check config.json or --det-model)")
        return YoloDetector(
            model_path=self.det_model,
            device=self.device,
            imgsz=self.det_imgsz,
            conf_thres=self.det_conf,
            iou_thres=self.det_iou,
            fp16=self.det_fp16,
        )

    def _build_restorer(self):
        if self.mode == "none":
            return None

        if self.mode == "pseudo" or self.restorer_name == "pseudo":
            fill = self.cfg.get("visualization", "fill_color", default=[255, 0, 255])
            op = float(self.cfg.get("visualization", "fill_opacity", default=0.70))
            r, g, b = [int(x) for x in fill]  # config is RGB
            return PseudoClipRestorer(device=self.device, fill_color_bgr=(b, g, r), fill_opacity=op)

        if self.restorer_name in ("none", "noop"):
            return None

        if not self.rest_model:
            raise FileNotFoundError("Restoration model path is empty (check config.json or --rest-model)")
        return BasicVSRPPClipRestorer(
            device=self.device,
            checkpoint_path=self.rest_model,
            fp16=self.rest_fp16,
            config=None,
        )

    def run(self) -> None:
        inp = Path(self.input_path)
        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Init timers
        metrics = PipelineMetrics()
        metrics.wall_start = _dt.datetime.now()
        t0_all = time.perf_counter()  # overall wall time (no mux)

        decoder = Decoder(
            input_path=str(inp),
            gpu_id=self.dec_gpu_id,
            batch_size=self.batch_size,
        )

        w = int(decoder.metadata.width)
        h = int(decoder.metadata.height)
        fps = float(decoder.metadata.fps)
        total_frames = int(decoder.metadata.num_frames or 0)

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
        )

        detector = self._build_detector()
        restorer = self._build_restorer()

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
            tracker = SceneTracker(cfg=tracker_cfg)

        store = FrameStore()

        pbar_total = (self.max_frames if self.max_frames is not None else (total_frames if total_frames > 0 else None))
        pbar = tqdm(total=pbar_total, disable=self.debug)

        frame_num = 0
        t0_all = time.perf_counter()

        try:
            while True:
                if self.max_frames is not None and frame_num >= self.max_frames:
                    break

                t0 = time.perf_counter()
                batch = decoder.read_batch()
                metrics.t_decode += (time.perf_counter() - t0)

                if not batch:
                    break

                batch_rgb: List[torch.Tensor] = []
                for surf in batch:
                    # NVDEC backend yields PyNvVideoCodec surfaces (RGBP on GPU).
                    # ffmpeg CPU fallback yields torch.Tensor frames (RGB HWC uint8 on CPU).
                    if isinstance(surf, torch.Tensor):
                        rgb = surf
                    else:
                        t = wrap_surface_as_tensor(surf)
                        # Be tolerant: if upstream gives HWC already, keep it.
                        if t.ndim == 3 and t.shape[-1] == 3:
                            rgb = t
                        else:
                            rgb = rgbp_chw_to_rgb_hwc_u8(t)

                    # Ensure frames live on the pipeline device (no-op if already there).
                    if self.device.type != "cpu":
                        rgb = rgb.to(self.device, non_blocking=True)

                    batch_rgb.append(rgb)

                detections: List[Detection] = []
                if detector is not None:
                    if self.sbs_enabled and self.sbs_det_split:
                        left_frames: List[torch.Tensor] = []
                        right_frames: List[torch.Tensor] = []
                        half_w = w // 2
                        for rgb in batch_rgb:
                            l, r = split_frame_lr(rgb, layout=self.sbs_layout)
                            left_frames.append(l.contiguous())
                            right_frames.append(r.contiguous())

                        t0 = time.perf_counter()
                        det_l = detector.detect_batch(left_frames)
                        det_r = detector.detect_batch(right_frames)
                        metrics.t_det += (time.perf_counter() - t0)

                        for dl, dr in zip(det_l, det_r):
                            boxes_l = _tensor_boxes_to_list_xyxy(dl.boxes)
                            boxes_r = _tensor_boxes_to_list_xyxy(dr.boxes)
                            masks_l = _extract_masks_list(dl) if self.use_seg_masks else None
                            masks_r = _extract_masks_list(dr) if self.use_seg_masks else None

                            merged_boxes = unsplit_boxes_layout(boxes_l, boxes_r, half_w=half_w, layout=self.sbs_layout)
                            merged_masks = unsplit_masks_layout(masks_l, masks_r, full_w=w, half_w=half_w, layout=self.sbs_layout)

                            det = Detection(
                                boxes=torch.tensor(
                                    [[b[1], b[0], b[3], b[2]] for b in merged_boxes],
                                    dtype=torch.float32,
                                    device="cpu",
                                )
                                if merged_boxes
                                else None,
                                scores=None,
                                classes=None,
                                masks=None,
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
                        detections = detector.detect_batch(batch_rgb)
                        metrics.t_det += (time.perf_counter() - t0)
                else:
                    detections = [Detection(boxes=None, scores=None, classes=None, masks=None) for _ in batch_rgb]

                for i, rgb in enumerate(batch_rgb):
                    if self.max_frames is not None and frame_num >= self.max_frames:
                        break

                    det = detections[i] if i < len(detections) else Detection(boxes=None, scores=None, classes=None, masks=None)

                    bgr_u8 = rgb_hwc_to_bgr_hwc_u8(rgb)

                    if tracker is None or self.mode == "none":
                        if self.enc_sync_before_encode:
                            sync_device(self.device)
                        t0 = time.perf_counter()
                        encoder.encode_frame(bgr_u8_to_bgra_u8(bgr_u8))
                        metrics.t_encode += (time.perf_counter() - t0)
                        frame_num += 1
                        metrics.processed_frames += 1
                        pbar.update(1)
                        continue

                    boxes = _tensor_boxes_to_list_xyxy(det.boxes)
                    masks_list = _extract_masks_list(det) if (self.use_seg_masks and det.masks is not None) else None

                    if self.roi_dilate > 0 and boxes:
                        dil = self.roi_dilate
                        boxes = [(t - dil, l - dil, b + dil, r + dil) for (t, l, b, r) in boxes]

                    if boxes:
                        boxes = [clip_box_to_bounds(bx, w=w, h=h) for bx in boxes]

                    if self.sbs_enabled and boxes:
                        seam_x = w // 2
                        boxes, masks_list = seam_split_boxes(boxes, seam_x=seam_x, full_w=w, full_h=h, masks=masks_list)

                    metrics.det_stats.add(boxes, w=w, h=h)

                    store.put(frame_num, bgr_u8)

                    t0 = time.perf_counter()
                    step = tracker.step_frame(frame_num, bgr_u8, boxes, masks_list)
                    metrics.t_track += (time.perf_counter() - t0)

                    if step.new_clips and restorer is not None:
                        for clip in step.new_clips:
                            t0 = time.perf_counter()
                            restored = restorer.restore_clip(clip)
                            _composite_clip_into_store(
                                clip=clip,
                                restored_frames=restored,
                                store_bgr_u8=store.frames_bgr_u8,
                                feather_radius=self.feather_radius,
                            )
                            metrics.t_restore += (time.perf_counter() - t0)

                    safe_before = tracker.min_active_start()
                    if safe_before is None:
                        safe_before = frame_num + 1

                    # "early passthrough" heuristic: no active scenes (min_active_start None) AND no detections on this frame
                    # => current frame becomes safe immediately (safe_before == frame_num+1).
                    if (len(boxes) == 0) and (tracker.min_active_start() is None) and (not step.new_clips):
                        metrics.early_passthrough_frames += 1

                    t0 = time.perf_counter()
                    drain_store_to_encoder(
                        store=store,
                        safe_before=int(safe_before),
                        encoder=encoder,
                        device=self.device,
                        sync_before_encode=self.enc_sync_before_encode,
                    )
                    metrics.t_encode += (time.perf_counter() - t0)

                    frame_num += 1
                    metrics.processed_frames += 1

                    pbar.update(1)

        finally:
            try:
                decoder.close()
            except Exception:
                pass

            if tracker is not None and restorer is not None:
                for clip in tracker.flush_eof():
                    restored = restorer.restore_clip(clip)
                    _composite_clip_into_store(
                        clip=clip,
                        restored_frames=restored,
                        store_bgr_u8=store.frames_bgr_u8,
                        feather_radius=self.feather_radius,
                    )

            drain_store_to_encoder(
                store=store,
                safe_before=10**18,
                encoder=encoder,
                device=self.device,
                sync_before_encode=self.enc_sync_before_encode,
            )

            # --- measure processing time (no mux) before close() ---
            t_total_no_mux = time.perf_counter() - t0_all

            # Close does audio remux by calling an internal subprocess and cleans up
            t0 = time.perf_counter()
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
                f"[Pipeline] Processing time (no mux) = {t_total_no_mux:.2f}s "
                f"Overhead = {overhead:.2f}s (sum_parts={sum_parts:.2f}s)"
            )
            print(f"[Pipeline] Total time (with mux) = {t_total_with_mux:.2f}s (mux={metrics.t_mux:.2f}s)")
            print(f"[Pipeline] DONE: Processed  &  Remuxed {metrics.processed_frames} frames")
            print(f"[Pipeline] early_passthrough_frames={metrics.early_passthrough_frames}")

            fw, ft, tb, avg_area, pct = metrics.det_stats.summary()
            print(
                f"[DetStats] frames_with_det={fw}/{ft} total_boxes={tb} "
                f"avg_roi_area_px={avg_area:.2f} ({pct:.4f}% of frame)"
            )

            if metrics.wall_start and metrics.wall_end:
                elapsed = metrics.wall_end - metrics.wall_start
                print(f"[Pipeline] Wall clock: start={metrics.wall_start} end={metrics.wall_end} elapsed={elapsed}")
            print(f"[Pipeline] perf_counter elapsed = {t_total_with_mux:.2f}s")

            try:
                pbar.close()
            except Exception:
                pass

        dt = time.perf_counter() - t0_all
        if self.debug:
            print(f"[Pipeline] Done. Frames={frame_num}, wall={dt:.2f}s")
