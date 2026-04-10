from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from gRestorer.core.scene import Clip, face_meta_clip_to_crop
from gRestorer.detector.core import FaceMetadata
from gRestorer.restorer.clip_restorer import BaseClipRestorer
from gRestorer.restorer.face_swap_worker import FaceSwapWorker
from gRestorer.restorer.face_enhancer import FaceEnhancer


@dataclass
class FaceSwapRestoreStats:
    clips_processed: int = 0
    frames_total: int = 0
    frames_with_detector_face_meta: int = 0
    frames_without_detector_face_meta: int = 0
    frames_gap_filled_forward: int = 0
    frames_gap_filled_backward: int = 0
    frames_worker_called: int = 0
    frames_worker_returned: int = 0
    frames_worker_returned_none: int = 0
    frames_materially_changed: int = 0
    mean_abs_diff_accum: float = 0.0
    frames_enhancer_called: int = 0
    frames_enhancer_returned: int = 0
    frames_enhancer_failed: int = 0
    frames_enhancer_materially_changed: int = 0
    enhancer_mean_abs_diff_accum: float = 0.0

    def avg_mean_abs_diff(self) -> float:
        return self.mean_abs_diff_accum / max(1, self.frames_worker_returned)

    def avg_enhancer_mean_abs_diff(self) -> float:
        return self.enhancer_mean_abs_diff_accum / max(1, self.frames_enhancer_returned)


class FaceSwapClipRestorer(BaseClipRestorer):
    """ROI-authoritative face-swap wrapper with corrected geometry.

    Policy lives here:
      - choose/stabilize the target face track within a clip
      - repad/resize behavior for the clip interface

    Worker contract:
      swap(one_original_crop_image, one_target_face_metadata_in_that_crop_space)
    """

    def __init__(
        self,
        device: torch.device,
        source_face_path: str,
        swap_model_path: str,
        *,
        swap_input_size: int = 128,
        provider: str = "auto",
        face_enhancer_model_path: str = "",
        face_enhancer_blend: int = 80,
    ) -> None:
        super().__init__(device=device)
        self.source_face_path = str(source_face_path)
        self.swap_model_path = str(swap_model_path)
        self.swap_input_size = int(swap_input_size)
        self.provider = str(provider or "auto").lower()
        self.face_enhancer_model_path = str(face_enhancer_model_path or "")
        self.face_enhancer_blend = int(face_enhancer_blend)

        self.debug_enabled = str(os.getenv("GR_FS_DEBUG", "0")).strip().lower() not in ("", "0", "false", "no")
        self.debug_dir = Path(os.getenv("GR_FS_DEBUG_DIR", "fs_debug"))
        self.debug_start = int(os.getenv("GR_FS_DEBUG_START", "-1"))
        self.debug_end = int(os.getenv("GR_FS_DEBUG_END", "-1"))
        if self.debug_enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.material_change_mad_threshold = float(os.getenv("GR_FS_MATERIAL_MAD", "1.0"))
        self.stats = FaceSwapRestoreStats()

        self.worker = FaceSwapWorker(
            device=self.device,
            source_face_path=self.source_face_path,
            swap_model_path=self.swap_model_path,
            provider=self.provider,
        )
        self.enhancer = None
        if self.face_enhancer_model_path:
            self.enhancer = FaceEnhancer(
                device=self.device,
                enhancer_model_path=self.face_enhancer_model_path,
                provider=self.provider,
                blend=self.face_enhancer_blend,
            )

    @staticmethod
    def _tensor_hwc_float_to_numpy_bgr_u8(x: torch.Tensor) -> np.ndarray:
        y = x.detach()
        if y.device.type != "cpu":
            y = y.cpu()
        y = y.to(torch.float32).clamp(0.0, 1.0)
        y = (y * 255.0).round().to(torch.uint8).contiguous()
        return np.ascontiguousarray(y.numpy())

    def _numpy_bgr_u8_to_tensor_hwc_float(self, x: np.ndarray) -> torch.Tensor:
        y = torch.from_numpy(np.ascontiguousarray(x)).to(torch.float32) / 255.0
        if self.device.type != "cpu":
            y = y.to(self.device, non_blocking=True)
        return y.contiguous()

    @staticmethod
    def _unpad_hwc_numpy(x: np.ndarray, pad: tuple[int, int, int, int]) -> np.ndarray:
        pt, pb, pl, pr = [int(v) for v in pad]
        h, w = int(x.shape[0]), int(x.shape[1])
        y0 = pt
        y1 = h - pb if pb > 0 else h
        x0 = pl
        x1 = w - pr if pr > 0 else w
        return np.ascontiguousarray(x[y0:y1, x0:x1, :])

    @staticmethod
    def _pad_hwc_numpy(x: np.ndarray, pad: tuple[int, int, int, int], clip_size: int) -> np.ndarray:
        pt, pb, pl, pr = [int(v) for v in pad]
        out = np.zeros((int(clip_size), int(clip_size), 3), dtype=x.dtype)
        h, w = int(x.shape[0]), int(x.shape[1])
        out[pt:pt + h, pl:pl + w, :] = x
        return out

    def _debug_this_frame(self, frame_num: int) -> bool:
        if not self.debug_enabled:
            return False
        if self.debug_start >= 0 and frame_num < self.debug_start:
            return False
        if self.debug_end >= 0 and frame_num > self.debug_end:
            return False
        return True

    def _save_debug_image(self, frame_num: int, name: str, img: np.ndarray) -> None:
        if not self._debug_this_frame(frame_num):
            return
        out = self.debug_dir / f"f{frame_num:06d}_{name}.png"
        cv2.imwrite(str(out), img)

    def _save_debug_text(self, frame_num: int, text: str) -> None:
        if not self._debug_this_frame(frame_num):
            return
        out = self.debug_dir / f"f{frame_num:06d}.txt"
        with open(out, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    @staticmethod
    def _diff_image(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return cv2.absdiff(a, b)

    @staticmethod
    def _face_area(face_meta: FaceMetadata) -> float:
        x1, y1, x2, y2 = face_meta.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _bbox_center_distance_sq_to_crop_center(face_meta: FaceMetadata, crop_shape: tuple[int, int, int]) -> float:
        h, w = int(crop_shape[0]), int(crop_shape[1])
        cx = 0.5 * w
        cy = 0.5 * h
        x1, y1, x2, y2 = face_meta.bbox_xyxy
        fx = 0.5 * (x1 + x2)
        fy = 0.5 * (y1 + y2)
        dx = fx - cx
        dy = fy - cy
        return dx * dx + dy * dy

    def _pick_anchor(self, face_metas: List[Optional[FaceMetadata]], crops: List[np.ndarray]):
        best_idx = None
        best_key = None
        for i, face_meta in enumerate(face_metas):
            if face_meta is None:
                continue
            key = (
                -self._bbox_center_distance_sq_to_crop_center(face_meta, crops[i].shape),
                self._face_area(face_meta),
                -i,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_idx = i
        return best_idx

    def _stabilize_face_metas(
        self,
        face_metas: List[Optional[FaceMetadata]],
        crops: List[np.ndarray],
        frame_nums: List[int],
    ) -> List[Optional[FaceMetadata]]:
        selected = list(face_metas)
        self.stats.frames_total += len(face_metas)
        self.stats.frames_with_detector_face_meta += sum(1 for fm in face_metas if fm is not None)
        self.stats.frames_without_detector_face_meta += sum(1 for fm in face_metas if fm is None)

        anchor_idx = self._pick_anchor(face_metas, crops)
        if anchor_idx is None:
            return selected

        self._save_debug_text(frame_nums[anchor_idx], "target_face_source=anchor_from_detector")

        ref = selected[anchor_idx]
        for i in range(anchor_idx + 1, len(selected)):
            if selected[i] is None:
                selected[i] = ref
                if ref is not None:
                    self.stats.frames_gap_filled_forward += 1
                    self._save_debug_text(frame_nums[i], "target_face_source=forward_fill")
            else:
                ref = selected[i]

        ref = selected[anchor_idx]
        for i in range(anchor_idx - 1, -1, -1):
            if selected[i] is None:
                selected[i] = ref
                if ref is not None:
                    self.stats.frames_gap_filled_backward += 1
                    self._save_debug_text(frame_nums[i], "target_face_source=backward_fill")
            else:
                ref = selected[i]

        return selected

    def get_stats_lines(self) -> List[str]:
        return [
            f"[FaceSwapStats] clips_processed={self.stats.clips_processed} frames_total={self.stats.frames_total}",
            f"[FaceSwapStats] detector_face_meta frames={self.stats.frames_with_detector_face_meta}/{self.stats.frames_total} missing={self.stats.frames_without_detector_face_meta}",
            f"[FaceSwapStats] gap_fill forward={self.stats.frames_gap_filled_forward} backward={self.stats.frames_gap_filled_backward}",
            f"[FaceSwapStats] worker_called={self.stats.frames_worker_called} returned={self.stats.frames_worker_returned} returned_none={self.stats.frames_worker_returned_none}",
            f"[FaceSwapStats] materially_changed={self.stats.frames_materially_changed} mad_threshold={self.material_change_mad_threshold:.3f} avg_mad={self.stats.avg_mean_abs_diff():.4f}",
            f"[FaceSwapStats] enhancer_called={self.stats.frames_enhancer_called} returned={self.stats.frames_enhancer_returned} failed={self.stats.frames_enhancer_failed}",
            f"[FaceSwapStats] enhancer_materially_changed={self.stats.frames_enhancer_materially_changed} avg_enhancer_mad={self.stats.avg_enhancer_mean_abs_diff():.4f}",
        ]

    @torch.inference_mode()
    def restore_clip(self, clip: Clip) -> List[torch.Tensor]:
        self.stats.clips_processed += 1
        out_frames: List[torch.Tensor] = []
        crops: list[np.ndarray] = []
        crop_resized_shapes: list[tuple[int, int]] = []
        pads: list[tuple[int, int, int, int]] = []
        clip_sizes: list[int] = []
        frame_nums: list[int] = []
        face_metas: list[Optional[FaceMetadata]] = []

        for i, clip_frame in enumerate(clip.frames):
            frame_num = int(clip.frame_nums[i])
            clip_np = self._tensor_hwc_float_to_numpy_bgr_u8(clip_frame)
            crop_h, crop_w = clip.crop_shapes[i]
            pad = clip.pad_after_resizes[i]
            clip_size = int(clip.clip_size)

            crop_resized_np = self._unpad_hwc_numpy(clip_np, pad)
            if int(crop_resized_np.shape[0]) != int(crop_h) or int(crop_resized_np.shape[1]) != int(crop_w):
                crop_np = cv2.resize(
                    crop_resized_np,
                    (int(crop_w), int(crop_h)),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                crop_np = crop_resized_np

            clip_face_meta = clip.face_metas[i] if hasattr(clip, "face_metas") else None
            crop_face_meta = face_meta_clip_to_crop(
                clip_face_meta,
                crop_shape=(crop_h, crop_w),
                out_hw=(int(crop_resized_np.shape[0]), int(crop_resized_np.shape[1])),
                pad=pad,
            )

            crops.append(crop_np)
            crop_resized_shapes.append((int(crop_resized_np.shape[0]), int(crop_resized_np.shape[1])))
            pads.append(pad)
            clip_sizes.append(clip_size)
            frame_nums.append(frame_num)
            face_metas.append(crop_face_meta)

            self._save_debug_text(
                frame_num,
                f"frame={frame_num} clip_id={clip.id} idx={i} crop_shape={crop_np.shape} clip_size={clip_size} pad={pad} has_face_meta={crop_face_meta is not None}",
            )
            if clip_face_meta is not None:
                self._save_debug_text(frame_num, f"clip_space_bbox=({clip_face_meta.bbox_xyxy[0]:.1f},{clip_face_meta.bbox_xyxy[1]:.1f},{clip_face_meta.bbox_xyxy[2]:.1f},{clip_face_meta.bbox_xyxy[3]:.1f})")
            if crop_face_meta is not None:
                self._save_debug_text(frame_num, f"crop_space_bbox=({crop_face_meta.bbox_xyxy[0]:.1f},{crop_face_meta.bbox_xyxy[1]:.1f},{crop_face_meta.bbox_xyxy[2]:.1f},{crop_face_meta.bbox_xyxy[3]:.1f})")
            self._save_debug_image(frame_num, "01_crop", crop_np)

        selected_faces = self._stabilize_face_metas(face_metas, crops, frame_nums)

        for i, crop_np in enumerate(crops):
            frame_num = frame_nums[i]
            face_meta = selected_faces[i]
            swapped_np = crop_np

            if face_meta is not None:
                x1, y1, x2, y2 = face_meta.bbox_xyxy
                self._save_debug_text(frame_num, f"target_face_bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                if face_meta.kps is not None:
                    self._save_debug_text(frame_num, f"target_face_kps_shape={tuple(face_meta.kps.shape)}")

                before = crop_np.copy()
                self.stats.frames_worker_called += 1
                try:
                    out = self.worker.swap(crop_np, face_meta)
                except Exception as e:
                    self._save_debug_text(frame_num, f"swap_exception={e!r}")
                    out = None

                if out is None:
                    self.stats.frames_worker_returned_none += 1
                    self._save_debug_text(frame_num, "swap_returned=None")
                else:
                    self.stats.frames_worker_returned += 1
                    mad = float(np.mean(np.abs(out.astype(np.int16) - before.astype(np.int16))))
                    self.stats.mean_abs_diff_accum += mad
                    material = mad >= self.material_change_mad_threshold
                    if material:
                        self.stats.frames_materially_changed += 1
                    self._save_debug_text(frame_num, f"mean_abs_diff={mad:.4f} material={material}")
                    self._save_debug_image(frame_num, "03_swap", out)
                    self._save_debug_image(frame_num, "04_diff", self._diff_image(before, out))
                    swapped_np = out

                    if self.enhancer is not None:
                        self.stats.frames_enhancer_called += 1
                        try:
                            enhanced = self.enhancer.enhance(swapped_np, face_meta)
                        except Exception as e:
                            self.stats.frames_enhancer_failed += 1
                            self._save_debug_text(frame_num, f"enhancer_exception={e!r}")
                            enhanced = None
                        if enhanced is not None:
                            self.stats.frames_enhancer_returned += 1
                            emad = float(np.mean(np.abs(enhanced.astype(np.int16) - swapped_np.astype(np.int16))))
                            self.stats.enhancer_mean_abs_diff_accum += emad
                            if emad >= self.material_change_mad_threshold:
                                self.stats.frames_enhancer_materially_changed += 1
                            self._save_debug_text(frame_num, f"enhancer_mean_abs_diff={emad:.4f}")
                            self._save_debug_image(frame_num, "03b_enhanced", enhanced)
                            self._save_debug_image(frame_num, "04b_enhancer_diff", self._diff_image(swapped_np, enhanced))
                            swapped_np = enhanced
            else:
                self._save_debug_text(frame_num, "target_face_source=none")

            target_h, target_w = crop_resized_shapes[i]
            if int(swapped_np.shape[0]) != target_h or int(swapped_np.shape[1]) != target_w:
                swapped_resized_np = cv2.resize(
                    swapped_np,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                swapped_resized_np = swapped_np

            swapped_clip_np = self._pad_hwc_numpy(swapped_resized_np, pads[i], clip_sizes[i])
            self._save_debug_image(frame_num, "05_clip_return", swapped_clip_np)
            out_frames.append(self._numpy_bgr_u8_to_tensor_hwc_float(swapped_clip_np))

        return out_frames


__all__ = ["FaceSwapClipRestorer"]
