# gRestorer/cli/pipeline.py
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from gRestorer.core.scene import Box, Clip, xyxy_to_tlbr
from gRestorer.core.scene_tracker import SceneTracker, TrackerConfig
from gRestorer.detector.core import Detection, MosaicDetector
from gRestorer.restorer import NoneRestorer, PseudoRestorer
from gRestorer.restorer.pseudo_clip_restorer import PseudoClipRestorer
from gRestorer.restorer.basicvsrpp_clip_restorer import BasicVSRPPClipRestorer
from gRestorer.utils.config_util import Config
from gRestorer.video import Decoder, Encoder

from gRestorer.cli.pipeline_utils import (
    _cfg_first,
    _cfg_get,
    _parse_color_bgr,
    _sync_device,
    pack_bgr_u8_to_bgra_u8_inplace,
    rgb_to_bgr_u8_inplace,
    wrap_surface_as_tensor,
)
from gRestorer.restorer.compositor import _composite_clip_into_store

class Pipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

        # Debug / instrumentation
        self.debug = bool(_cfg_first(cfg, [("debug",), ("verbose",)], default=False))

        self.input_path = str(_cfg_get(cfg, "input_path"))
        self.output_path = str(_cfg_get(cfg, "output_path"))

        # Device / GPU id
        self.gpu_id = int(_cfg_first(cfg, [("decoder", "gpu_id"), ("encoder", "gpu_id"), ("gpu_id",)], default=0))
        self.device_str = str(_cfg_first(cfg, [("device",)], default=(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")))
        self.device = torch.device(self.device_str)

        # Restorer selection
        self.restorer_name = str(_cfg_first(cfg, [("restorer",)], default="none"))

        # Diagnostic knob: if True, synchronize device before handing frames to the encoder.
        # This avoids rare torch->NVENC stream races (seen as occasional unshaded / stale frames).
        # Clip modes do GPU work right before NVENC reads the surfaces; default to syncing on CUDA/XPU.
        rn = self.restorer_name.lower().strip()
        default_sync = (self.device.type in ("cuda", "xpu")) and (rn in ("pseudo_clip", "basicvsrpp", "basicvsrpp_clip"))
        self.sync_before_encode = bool(
            _cfg_first(cfg, [("encoder", "sync_before_encode"), ("sync_before_encode",)], default=default_sync)
        )

        # Decode / encode settings
        self.batch_size = int(_cfg_first(cfg, [("detection", "batch_size"), ("batch_size",)], default=8))
        self.max_frames = _cfg_first(cfg, [("max_frames",)], default=None)
        self.max_frames = int(self.max_frames) if self.max_frames is not None else None

        # Detector settings
        self.det_model = _cfg_first(cfg, [("det_model_path",), ("det_model",), ("detection", "model_path")], default=None)
        self.det_imgsz = int(_cfg_first(cfg, [("detection", "imgsz"), ("det_imgsz",)], default=640))
        self.det_conf = float(_cfg_first(cfg, [("detection", "conf_threshold"), ("det_conf",)], default=0.25))
        self.det_iou = float(_cfg_first(cfg, [("detection", "iou_threshold"), ("det_iou",)], default=0.45))
        self.det_fp16 = bool(_cfg_first(cfg, [("restoration", "fp16"), ("det_fp16",)], default=True))

        # Clip/tracking settings
        self.clip_size = int(_cfg_first(cfg, [("restoration", "clip_size"), ("clip_size",)], default=256))
        self.max_clip_length = int(_cfg_first(cfg, [("restoration", "max_clip_length"), ("max_clip_length",)], default=30))
        self.border_size = float(_cfg_first(cfg, [("restoration", "border_ratio"), ("border_size",)], default=0.06))
        self.pad_mode = str(_cfg_first(cfg, [("restoration", "pad_mode"), ("pad_mode",)], default="reflect"))
        self.roi_dilate = int(_cfg_first(cfg, [("roi_dilate",)], default=0))

        # Visualization settings (mainly for pseudo modes)
        self.box_color = _parse_color_bgr(_cfg_first(cfg, [("visualization", "box_color"), ("box_color",)], default=[0, 255, 0]))
        self.box_thickness = int(_cfg_first(cfg, [("visualization", "box_thickness"), ("box_thickness",)], default=2))
        self.fill_color = _parse_color_bgr(_cfg_first(cfg, [("visualization", "fill_color"), ("fill_color",)], default=None))
        self.fill_opacity = float(_cfg_first(cfg, [("visualization", "fill_opacity"), ("fill_opacity",)], default=0.5))

        #Restoratin - BasicVSR++
        self.feather_radius = int( _cfg_first(self.cfg, [("restoration", "feather_radius"), ("feather_radius",)], default=3))

    def _build_detector(self) -> MosaicDetector:
        if not self.det_model:
            raise ValueError("--det-model is required when using a restorer that needs detection")

        return MosaicDetector(
            model_path=str(self.det_model),
            device=self.device_str,
            imgsz=self.det_imgsz,
            conf_thres=self.det_conf,
            iou_thres=self.det_iou,
            fp16=self.det_fp16,
        )

    def _build_tracker(self) -> SceneTracker:
        cfg = TrackerConfig(
            clip_size=self.clip_size,
            max_clip_length=self.max_clip_length,
            pad_mode=self.pad_mode,
            border_size=self.border_size,
            max_box_expansion_factor=1.0,
            use_seg_masks=bool(_cfg_first(self.cfg, [("use_seg_masks",), ("tracker", "use_seg_masks")], default=True)),
            debug=bool(_cfg_first(self.cfg, [("debug",), ("tracker_debug",)], default=False)),
        )
        return SceneTracker(cfg)

    def run(self) -> None:
        #Let us initialize the timer
        t_start = time.perf_counter()

        # Decoder
        decoder = Decoder(self.input_path, batch_size=self.batch_size, gpu_id=self.gpu_id)
        width = int(decoder.metadata.width)
        height = int(decoder.metadata.height)
        fps = float(decoder.metadata.fps) if decoder.metadata.fps else 30.0

        # Detection coverage counters (per run)
        frames_with_det = 0
        total_boxes = 0
        total_roi_pixels = 0
        frame_pixels = width * height

        # Encoder
        enc_codec = str(_cfg_first(self.cfg, [("encoder", "codec"), ("codec",)], default="hevc"))
        enc_preset = str(_cfg_first(self.cfg, [("encoder", "preset"), ("preset",)], default="P7"))
        enc_profile = str(_cfg_first(self.cfg, [("encoder", "profile"), ("profile",)], default="main"))
        enc_qp = int(_cfg_first(self.cfg, [("encoder", "qp"), ("qp",)], default=23))
        # Encoder always expects BGRA surfaces (ARGB in PyNvVideoCodec terms) in this project.
        # The Encoder implementation enforces that internally.
        enc_container = _cfg_first(self.cfg, [("encoder", "container"), ("container",)], default=None)

        encoder = Encoder(
            self.output_path,
            width=width,
            height=height,
            fps=fps,
            codec=enc_codec,
            preset=enc_preset,
            profile=enc_profile,
            qp=enc_qp,
            gpu_id=self.gpu_id,
            container=enc_container,
            input_path=self.input_path,
        )

        restorer_name = self.restorer_name.lower().strip()

        if restorer_name == "none":
            restorer = NoneRestorer(device=self.device, width=width, height=height)
            detector = None
            tracker = None
            clip_restorer = None
            clip_mode = False

        elif restorer_name == "pseudo":
            restorer = PseudoRestorer(
                width=width,
                height=height,
                box_color=self.box_color or (0, 255, 0),
                box_thickness=self.box_thickness,
                fill_color=self.fill_color,
                fill_opacity=self.fill_opacity,
                gpu_id=self.gpu_id,
            )
            detector = self._build_detector()
            tracker = None
            clip_restorer = None
            clip_mode = False

        elif restorer_name == "pseudo_clip":
            restorer = None
            detector = self._build_detector()
            tracker = self._build_tracker()
            clip_restorer = PseudoClipRestorer(
                device=self.device,
                fill_color_bgr=self.fill_color or (255, 0, 255),
                fill_opacity=self.fill_opacity,
            )
            clip_mode = True

        elif restorer_name == "basicvsrpp":
            restorer = None
            detector = self._build_detector()
            tracker = self._build_tracker()

            ckpt = _cfg_first(self.cfg, [("restoration", "rest_model_path"), ("rest_model_path",)], default=None)
            fp16 = bool(_cfg_first(self.cfg, [("restoration", "fp16"), ("rest_fp16",)], default=True))
            clip_restorer = BasicVSRPPClipRestorer(
                device=self.device,
                checkpoint_path=str(ckpt),
                fp16=fp16,
                config=None,  # use LADA default gan inference config
            )
            clip_mode = True

        else:
            raise ValueError(f"Unknown restorer: {self.restorer_name}")

        print(f"[Pipeline] input='{self.input_path}' -> output='{self.output_path}'")
        print(f"[Pipeline] device={self.device_str} restorer={restorer_name} batch={self.batch_size}")

        # --- Streaming loop ---
        frame_num_base = 0
        frames_done = 0

        # For clip-mode, we buffer full frames (uint8 BGR) until they are safe to emit.
        store_bgr_u8: Dict[int, torch.Tensor] = {}
        next_out = 0

        t_decode_total = 0.0
        t_det_total = 0.0
        t_track_total = 0.0
        t_restore_total = 0.0
        t_encode_total = 0.0

        def drain_ready(safe_before: int) -> None:
            nonlocal next_out, t_encode_total
            if safe_before <= next_out:
                return

            while next_out < safe_before:
                # Batch consecutive frames if available.
                batch_nums: List[int] = []
                for _ in range(self.batch_size):
                    # Never drain/encode frame >= safe_before (safe_before is exclusive).
                    if next_out >= safe_before:
                        break
                    if next_out in store_bgr_u8:
                        batch_nums.append(next_out)
                        next_out += 1
                    else:
                        break

                if not batch_nums:
                    # We don't have the next frame yet.
                    return

                t0 = time.perf_counter()
                bgra_list: List[torch.Tensor] = []
                for n in batch_nums:
                    bgr = store_bgr_u8.pop(n)
                    bgra = torch.empty((height, width, 4), device=bgr.device, dtype=torch.uint8)
                    pack_bgr_u8_to_bgra_u8_inplace(bgra, bgr)
                    bgra_list.append(bgra)
                # Optional diagnostic sync: helps detect async races between torch ops and NVENC.
                if self.sync_before_encode:
                    _sync_device(self.device)

                encoder.encode_frames(bgra_list)
                t1 = time.perf_counter()
                t_encode_total += (t1 - t0)

        while True:
            if self.max_frames is not None and frames_done >= self.max_frames:
                break

            t0 = time.perf_counter()
            frames = decoder.read_batch()
            t1 = time.perf_counter()
            t_decode_total += (t1 - t0)

            if not frames:
                break

            # Convert decoder output to BGR uint8 HWC.
            rgb_list = [wrap_surface_as_tensor(f) for f in frames]
            bgr_list: List[torch.Tensor] = []
            for rgb in rgb_list:
                bgr = torch.empty((height, width, 3), device=rgb.device, dtype=torch.uint8)
                rgb_to_bgr_u8_inplace(bgr, rgb)
                bgr_list.append(bgr)

            # Detection
            detections: Optional[List[Detection]] = None
            if detector is not None:
                td0 = time.perf_counter()
                detections = detector.detect_batch(bgr_list)
                td1 = time.perf_counter()
                t_det_total += (td1 - td0)

            if not clip_mode:
                # Frame-mode output (none/pseudo): process and encode immediately.
                out_bgr: List[torch.Tensor] = bgr_list
                if restorer is not None:
                    tr0 = time.perf_counter()
                    out_bgr = restorer.process_batch(bgr_list, detections_per_frame=detections)
                    tr1 = time.perf_counter()
                    t_restore_total += (tr1 - tr0)

                te0 = time.perf_counter()
                bgra_list: List[torch.Tensor] = []
                for bgr in out_bgr:
                    bgra = torch.empty((height, width, 4), device=bgr.device, dtype=torch.uint8)
                    pack_bgr_u8_to_bgra_u8_inplace(bgra, bgr)
                    bgra_list.append(bgra)
                if self.sync_before_encode:
                    _sync_device(self.device)
                encoder.encode_frames(bgra_list)
                te1 = time.perf_counter()
                t_encode_total += (te1 - te0)

                frames_done += len(frames)
                frame_num_base += len(frames)
                continue

            # --- Clip-mode: buffer frames + track + restore completed clips ---
            assert detections is not None
            assert tracker is not None
            assert clip_restorer is not None

            # Store frames
            for i, bgr in enumerate(bgr_list):
                fn = frame_num_base + i
                store_bgr_u8[fn] = bgr

            # Track per frame, restore completed clips, composite into buffer.
            for i, det in enumerate(detections):
                fn = frame_num_base + i

                roi_boxes: List[Box] = []
                roi_masks: Optional[List[torch.Tensor]] = None

                if det.boxes is not None and det.boxes.numel() > 0:
                    # det.boxes: [N,4] xyxy float
                    n = int(det.boxes.shape[0])
                    if det.masks is not None and det.masks.shape[0] == n:
                        roi_masks = []

                    for j in range(n):
                        box_xyxy = det.boxes[j]
                        x1, y1, x2, y2 = [float(v.item()) for v in box_xyxy]
                        # xyxy_to_tlbr signature is (xyxy, h, w)
                        roi_boxes.append(xyxy_to_tlbr((x1, y1, x2, y2), height, width))

                        if roi_masks is not None:
                            # det.masks is CPU uint8 [N,H,W] in original resolution.
                            roi_masks.append(det.masks[j])
                if roi_boxes:
                    frames_with_det += 1
                    total_boxes += len(roi_boxes)
                    for (t, l, b, r) in roi_boxes:
                        total_roi_pixels += (b - t + 1) * (r - l + 1)

                tt0 = time.perf_counter()
                step = tracker.step_frame(fn, store_bgr_u8[fn], roi_boxes, roi_masks)
                tt1 = time.perf_counter()
                t_track_total += (tt1 - tt0)

                if self.debug:
                    det_n = len(roi_boxes)
                    ov_n = len(step.overlay_boxes) if step.overlay_boxes is not None else 0
                    print(
                        f"[Dbg] f={fn:6d} det={det_n:2d} overlay={ov_n:2d} "
                        f"active_scenes={step.active_scenes:2d} new_clips={len(step.new_clips):2d}"
                    )

                if step.new_clips:
                    # Ensure stable ordering for deterministic composites.
                    clips_sorted = sorted(step.new_clips, key=lambda c: (c.frame_start, c.id))
                    for clip in clips_sorted:
                        tr0 = time.perf_counter()
                        restored = clip_restorer.restore_clip(clip)
                        _composite_clip_into_store(
                            clip=clip,
                            restored_frames=restored,
                            store_bgr_u8=store_bgr_u8,
                            feather_radius=self.feather_radius,
                        )
                        tr1 = time.perf_counter()
                        t_restore_total += (tr1 - tr0)

            frames_done += len(frames)
            frame_num_base += len(frames)

            # Drain only frames that are definitely no longer part of any active (unfinished) scene.
            safe_before = tracker.min_active_start()
            # Extra safety: derive from actual stored frame numbers in active scenes (guards against bugs)
            try:
                active = getattr(tracker, '_scenes', None)
                if active:
                    safe_by_frame0 = min(int(s.frame_nums[0]) for s in active if getattr(s, 'frame_nums', None))
                    safe_before = safe_by_frame0 if safe_before is None else min(int(safe_before), safe_by_frame0)
            except Exception:
                safe_before = None

            if safe_before is None:
                # Conservative: if we can't prove frames are safe, don't drain in clip-mode.
                safe_before = next_out
            else:
                safe_before = int(safe_before)

            if self.debug:
                print(f'[Dbg] drain safe_before={safe_before} next_out={next_out} store={len(store_bgr_u8)}')
            drain_ready(safe_before)

        # EOF: flush tracker/remaining clips and emit remaining frames.
        if clip_mode:
            assert tracker is not None
            assert clip_restorer is not None

            remaining_clips = tracker.flush_eof(frames_done)
            remaining_clips = sorted(remaining_clips, key=lambda c: (c.frame_start, c.id))
            for clip in remaining_clips:
                tr0 = time.perf_counter()
                restored = clip_restorer.restore_clip(clip)
                _composite_clip_into_store(
                    clip=clip,
                    restored_frames=restored,
                    store_bgr_u8=store_bgr_u8,
                    feather_radius=self.feather_radius,
                )
                tr1 = time.perf_counter()
                t_restore_total += (tr1 - tr0)

            drain_ready(frame_num_base)

        # IMPORTANT: Encoder.close() does not flush tail packets; flush explicitly.
        encoder.flush()
        # Capture processing time
        t_processing = time.perf_counter() - t_start
        encoder.close()
        decoder.close()

        # Compute total time taken
        t1 = time.perf_counter()
        t_total = t1 - t_start
        t_known = t_decode_total + t_det_total + t_track_total + t_restore_total + t_encode_total
        t_other_processing = t_processing - t_known
        t_other = t_total - t_known

        print(
            f"[Pipeline] Processed {frames_done} frames: "
            f"t_decode={t_decode_total:.2f}s t_det={t_det_total:.2f}s "
            f"t_track={t_track_total:.2f}s t_restore={t_restore_total:.2f}s "
            f"t_encode={t_encode_total:.2f}s"
        )

        if t_other_processing < -0.05:
            print(f"[Pipeline] Processing time (no mux) = {t_processing:.2f}s " f"(parts overlap/async by {-t_other_processing:.2f}s; sum_parts={t_known:.2f}s)")
        else:
            print(f"[Pipeline] Processing time (no mux) = {t_processing:.2f}s " f"Overhead = {max(0.0, t_other_processing):.2f}s (sum_parts={t_known:.2f}s)")

        # Full end-to-end: includes remux
        mux_time = max(0.0, t_total - t_processing)
        print(f"[Pipeline] Total time (with mux) = {t_total:.2f}s " f"(mux={mux_time:.2f}s)")
        print( f"[Pipeline] DONE: Processed  &  Remuxed {frames_done} frames" )

