from __future__ import annotations

from dataclasses import dataclass
import json
from typing import List, Optional
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from gRestorer.core.scene import Clip, face_meta_clip_to_crop
from gRestorer.detector.core import FaceMetadata
from gRestorer.restorer.clip_restorer import BaseClipRestorer
from gRestorer.restorer.face_enhancer import FaceEnhancer
from gRestorer.restorer.face_occluder import FaceOccluder
from gRestorer.restorer.face_landmarker import FaceLandmarker
from gRestorer.restorer.face_expression_restorer import FaceExpressionRestorer



@dataclass
class FaceSwapRestoreStats:
    clips_processed: int = 0
    frames_total: int = 0
    frames_with_detector_face_meta: int = 0
    frames_without_detector_face_meta: int = 0
    frames_gap_filled_forward: int = 0
    frames_gap_filled_backward: int = 0
    frames_force_swap_attempted: int = 0
    frames_synthetic_face_meta_used: int = 0
    frames_detector_face_meta_authoritative: int = 0
    frames_bbox_landmark_fallback: int = 0
    frames_previous_swap_reused: int = 0
    frames_landmarker_called: int = 0
    frames_landmarker_returned: int = 0
    frames_landmarker_failed: int = 0
    frames_selector_called: int = 0
    frames_selector_candidates_found: int = 0
    frames_selector_replaced: int = 0
    frames_selector_kept: int = 0
    frames_selector_no_viable: int = 0
    frames_worker_called: int = 0
    frames_worker_returned: int = 0
    frames_worker_returned_none: int = 0
    worker_exceptions: int = 0
    last_worker_exception: str = ""
    frames_materially_changed: int = 0
    mean_abs_diff_accum: float = 0.0
    frames_expression_called: int = 0
    frames_expression_returned: int = 0
    frames_expression_failed: int = 0
    frames_expression_materially_changed: int = 0
    expression_mean_abs_diff_accum: float = 0.0
    frames_enhancer_called: int = 0
    frames_enhancer_returned: int = 0
    frames_enhancer_failed: int = 0
    frames_enhancer_materially_changed: int = 0
    enhancer_mean_abs_diff_accum: float = 0.0
    frames_occluder_called: int = 0
    frames_occluder_returned: int = 0
    frames_occluder_failed: int = 0
    frames_occluder_materially_changed: int = 0
    occluder_mean_abs_diff_accum: float = 0.0
    frames_metrics_written: int = 0
    frames_mixed_embedding_used: int = 0
    frames_target_embedding_used: int = 0
    frames_target_embedding_missing: int = 0
    embedding_source_to_target_cos_accum: float = 0.0
    embedding_source_to_target_cos_count: int = 0
    embedding_source_to_mixed_cos_accum: float = 0.0
    embedding_source_to_mixed_cos_count: int = 0
    embedding_target_to_mixed_cos_accum: float = 0.0
    embedding_target_to_mixed_cos_count: int = 0
    frames_geometry_stabilized: int = 0
    geometry_bbox_shift_accum: float = 0.0
    geometry_bbox_shift_count: int = 0
    geometry_kps_shift_accum: float = 0.0
    geometry_kps_shift_count: int = 0

    def avg_mean_abs_diff(self) -> float:
        return self.mean_abs_diff_accum / max(1, self.frames_worker_returned)

    def avg_expression_mean_abs_diff(self) -> float:
        return self.expression_mean_abs_diff_accum / max(1, self.frames_expression_returned)

    def avg_enhancer_mean_abs_diff(self) -> float:
        return self.enhancer_mean_abs_diff_accum / max(1, self.frames_enhancer_returned)

    def avg_occluder_mean_abs_diff(self) -> float:
        return self.occluder_mean_abs_diff_accum / max(1, self.frames_occluder_returned)

    def avg_embedding_source_to_target_cos(self) -> float:
        return self.embedding_source_to_target_cos_accum / max(1, self.embedding_source_to_target_cos_count)

    def avg_embedding_source_to_mixed_cos(self) -> float:
        return self.embedding_source_to_mixed_cos_accum / max(1, self.embedding_source_to_mixed_cos_count)

    def avg_embedding_target_to_mixed_cos(self) -> float:
        return self.embedding_target_to_mixed_cos_accum / max(1, self.embedding_target_to_mixed_cos_count)

    def avg_geometry_bbox_shift(self) -> float:
        return self.geometry_bbox_shift_accum / max(1, self.geometry_bbox_shift_count)

    def avg_geometry_kps_shift(self) -> float:
        return self.geometry_kps_shift_accum / max(1, self.geometry_kps_shift_count)

class BaseFaceSwapClipRestorer(BaseClipRestorer):
    """ROI-authoritative face-swap wrapper with global post-processing.

    Policy lives here:
      - choose/stabilize the target face track within a clip
      - optional landmark refinement for the already chosen target face
      - run the selected swapper's native paste-back path
      - apply global ROI-space enhancer and occluder preserve stages
      - repad/resize behavior for the clip interface

    Worker contract:
      swap(one_original_crop_image, one_target_face_metadata_in_that_crop_space)
          -> swapped crop image in the same ROI/crop space

    Important design choice:
      Each swapper owns its own align/infer/native paste-back semantics.
      This class uses native swapper paste-back, not the inactive aligned-space compositor path.
    """

    def __init__(
        self,
        device: torch.device,
        source_face_path: str,
        swap_model_path: str,
        *,
        swap_input_size: int = 128,
        provider: str = "auto",
        face_swapper_weight: float = 1.0,
        swap_policy: str = "detected",
        force_swap_detected_faces: bool = False,
        force_swap_synthesize_meta: bool = True,
        allow_bbox_landmark_fallback: bool = True,
        reuse_previous_on_swap_failure: bool = False,
        geometry_stabilization_enabled: bool = False,
        geometry_stabilization_window: int = 5,
        geometry_stabilization_bbox: float = 0.65,
        geometry_stabilization_kps: float = 0.65,
        geometry_stabilization_max_bbox_jump_px: float = 0.0,
        geometry_stabilization_max_kps_jump_px: float = 0.0,
        metrics_enabled: bool = False,
        metrics_path: str = "",
        face_comp_mask_mode: str = "geom_backend_intersection",
        face_comp_geom_expand: float = 1.05,
        face_comp_mask_erode: int = 0,
        face_comp_mask_dilate: int = 2,
        face_comp_mask_blur: int = 5,
        face_comp_blend_mode: str = "alpha",
        face_comp_color_transfer: str = "none",
        face_comp_face_scale: float = 0.0,
        face_comp_debug: bool = False,
        expression_restorer_enabled: bool = False,
        expression_restorer_model: str = "live_portrait",
        expression_restorer_feature_extractor_path: str = "",
        expression_restorer_motion_extractor_path: str = "",
        expression_restorer_generator_path: str = "",
        expression_restorer_provider: str = "auto",
        expression_restorer_factor: int = 80,
        expression_restorer_areas: Optional[List[str]] = None,
        expression_restorer_mask_blur: float = 0.3,
        face_enhancer_enabled: bool = False,
        face_enhancer_model_path: str = "",
        face_enhancer_provider: str = "auto",
        face_enhancer_blend: int = 80,
        face_occluder_enabled: bool = False,
        face_occluder_model_path: str = "",
        face_occluder_provider: str = "auto",
        face_occluder_threshold: float = 0.5,
        face_occluder_blur: int = 5,
        face_occluder_blend: int = 100,
        face_occluder_invert: bool = False,
        landmark_refiner_enabled: bool = False,
        landmark_model: str = "2dfan4",
        landmark_model_path: str = "",
        landmark_provider: str = "auto",
        landmark_score: float = 0.5,
        debug_enabled: bool = False,
        debug_dir: str = "fs_debug",
        debug_start: int = -1,
        debug_end: int = -1,
        material_change_mad_threshold: float = 1.0,
        **_ignored_kwargs,
    ) -> None:
        super().__init__(device=device)
        self.source_face_path = str(source_face_path)
        self.swap_model_path = str(swap_model_path)
        self.swap_input_size = int(swap_input_size)
        self.provider = str(provider or "auto").lower()
        self.face_swapper_weight = float(max(0.0, min(1.0, float(face_swapper_weight))))
        self.swap_policy = str(swap_policy or "detected").strip().lower()
        if self.swap_policy not in ("strict", "detected", "aggressive"):
            raise ValueError("swap_policy must be one of: strict, detected, aggressive")
        self.force_swap_detected_faces = bool(force_swap_detected_faces)
        self.force_swap_synthesize_meta = bool(force_swap_synthesize_meta)
        self.allow_bbox_landmark_fallback = bool(allow_bbox_landmark_fallback)
        self.reuse_previous_on_swap_failure = bool(reuse_previous_on_swap_failure) or self.swap_policy == "aggressive"
        self.geometry_stabilization_enabled = bool(geometry_stabilization_enabled)
        self.geometry_stabilization_window = max(1, int(geometry_stabilization_window))
        if self.geometry_stabilization_window % 2 == 0:
            self.geometry_stabilization_window += 1
        self.geometry_stabilization_bbox = float(max(0.0, min(1.0, float(geometry_stabilization_bbox))))
        self.geometry_stabilization_kps = float(max(0.0, min(1.0, float(geometry_stabilization_kps))))
        self.geometry_stabilization_max_bbox_jump_px = float(max(0.0, float(geometry_stabilization_max_bbox_jump_px)))
        self.geometry_stabilization_max_kps_jump_px = float(max(0.0, float(geometry_stabilization_max_kps_jump_px)))
        self.metrics_enabled = bool(metrics_enabled)
        self.metrics_path = Path(str(metrics_path or "").strip()) if str(metrics_path or "").strip() else None

        self.expression_restorer_enabled = bool(expression_restorer_enabled)
        self.expression_restorer_model = str(expression_restorer_model or "live_portrait").lower()
        self.expression_restorer_feature_extractor_path = str(expression_restorer_feature_extractor_path or "")
        self.expression_restorer_motion_extractor_path = str(expression_restorer_motion_extractor_path or "")
        self.expression_restorer_generator_path = str(expression_restorer_generator_path or "")
        self.expression_restorer_provider = str(expression_restorer_provider or self.provider or "auto").lower()
        self.expression_restorer_factor = int(max(0, min(100, int(expression_restorer_factor))))
        self.expression_restorer_areas = list(expression_restorer_areas or ["upper-face", "lower-face"])
        self.expression_restorer_mask_blur = float(expression_restorer_mask_blur)

        self.face_enhancer_enabled = bool(face_enhancer_enabled)
        self.face_enhancer_model_path = str(face_enhancer_model_path or "")
        self.face_enhancer_provider = str(face_enhancer_provider or self.provider or "auto").lower()
        self.face_enhancer_blend = int(face_enhancer_blend)

        self.face_occluder_enabled = bool(face_occluder_enabled)
        self.face_occluder_model_path = str(face_occluder_model_path or os.getenv("GR_FS_OCCLUDER_MODEL", "") or "")
        self.face_occluder_provider = str(face_occluder_provider or self.provider or "auto").lower()
        self.face_occluder_threshold = float(face_occluder_threshold)
        self.face_occluder_blur = int(face_occluder_blur)
        self.face_occluder_blend = int(face_occluder_blend)
        self.face_occluder_invert = bool(face_occluder_invert)

        self.landmark_refiner_enabled = bool(landmark_refiner_enabled)
        self.landmark_model = str(landmark_model or "2dfan4")
        self.landmark_model_path = str(landmark_model_path or "")
        self.landmark_provider = str(landmark_provider or self.provider or "auto").lower()
        self.landmark_score = float(max(0.0, min(1.0, landmark_score)))

        self.debug_enabled = bool(debug_enabled)
        self.debug_dir = Path(debug_dir)
        self.debug_start = int(debug_start)
        self.debug_end = int(debug_end)
        if self.debug_enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.material_change_mad_threshold = float(material_change_mad_threshold)
        self.stats = FaceSwapRestoreStats()
        if self.metrics_enabled:
            if self.metrics_path is None:
                self.metrics_path = Path("face_swap_metrics.jsonl")
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_metrics_event(
                {
                    "event": "face_swap_run_start",
                    "worker_backend": self.__class__.__name__,
                    "source_face_path": self.source_face_path,
                    "swap_model_path": self.swap_model_path,
                    "swap_input_size": self.swap_input_size,
                    "provider": self.provider,
                    "face_swapper_weight": self.face_swapper_weight,
                    "swap_policy": self.swap_policy,
                    "force_swap_detected_faces": self.force_swap_detected_faces,
                    "force_swap_synthesize_meta": self.force_swap_synthesize_meta,
                    "allow_bbox_landmark_fallback": self.allow_bbox_landmark_fallback,
                    "reuse_previous_on_swap_failure": self.reuse_previous_on_swap_failure,
                    "geometry_stabilization_enabled": self.geometry_stabilization_enabled,
                    "geometry_stabilization_window": self.geometry_stabilization_window,
                    "geometry_stabilization_bbox": self.geometry_stabilization_bbox,
                    "geometry_stabilization_kps": self.geometry_stabilization_kps,
                    "geometry_stabilization_max_bbox_jump_px": self.geometry_stabilization_max_bbox_jump_px,
                    "geometry_stabilization_max_kps_jump_px": self.geometry_stabilization_max_kps_jump_px,
                }
            )
        if self.geometry_stabilization_enabled:
            print(
                "[FaceGeometryStabilizer] enabled "
                f"window={self.geometry_stabilization_window} "
                f"bbox={self.geometry_stabilization_bbox:.3f} "
                f"kps={self.geometry_stabilization_kps:.3f} "
                f"max_bbox_jump_px={self.geometry_stabilization_max_bbox_jump_px:.1f} "
                f"max_kps_jump_px={self.geometry_stabilization_max_kps_jump_px:.1f}"
            )

        self._swap_exception_print_budget = 5
        self.copy_guard = str(os.getenv("GR_FS_COPY_GUARD", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        self.worker = self._build_worker()

        # Keep face_comp_* constructor arguments for CLI/config compatibility,
        # but do not use the experimental shared face compositor in the active path.
        self.landmarker = None
        if self.landmark_refiner_enabled:
            if not self.landmark_model_path:
                raise FileNotFoundError("Landmark refiner is enabled but landmark_model_path is empty")
            self.landmarker = FaceLandmarker(
                device=self.device,
                model_name=self.landmark_model,
                model_path=self.landmark_model_path,
                provider=self.landmark_provider,
                score=self.landmark_score,
            )

        self.expression_restorer = None
        if self.expression_restorer_enabled:
            if not self.expression_restorer_feature_extractor_path:
                raise FileNotFoundError("Expression restorer is enabled but feature_extractor_path is empty")
            if not self.expression_restorer_motion_extractor_path:
                raise FileNotFoundError("Expression restorer is enabled but motion_extractor_path is empty")
            if not self.expression_restorer_generator_path:
                raise FileNotFoundError("Expression restorer is enabled but generator_path is empty")
            self.expression_restorer = FaceExpressionRestorer(
                device=self.device,
                model=self.expression_restorer_model,
                feature_extractor_path=self.expression_restorer_feature_extractor_path,
                motion_extractor_path=self.expression_restorer_motion_extractor_path,
                generator_path=self.expression_restorer_generator_path,
                provider=self.expression_restorer_provider,
                factor=self.expression_restorer_factor,
                areas=self.expression_restorer_areas,
                mask_blur=self.expression_restorer_mask_blur,
            )

        self.enhancer = None
        if self.face_enhancer_enabled and self.face_enhancer_model_path:
            self.enhancer = FaceEnhancer(
                device=self.device,
                enhancer_model_path=self.face_enhancer_model_path,
                provider=self.face_enhancer_provider,
                blend=self.face_enhancer_blend,
            )

        self.occluder = None
        if self.face_occluder_enabled and self.face_occluder_model_path:
            self.occluder = FaceOccluder(
                device=self.device,
                occluder_model_path=self.face_occluder_model_path,
                provider=self.face_occluder_provider,
                threshold=self.face_occluder_threshold,
                blur=self.face_occluder_blur,
                blend=self.face_occluder_blend,
                invert=self.face_occluder_invert,
            )

    def _build_worker(self):
        raise NotImplementedError("Subclasses must create the concrete face-swap worker.")

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
    def _tensor_hw_to_numpy_u8(x: torch.Tensor) -> np.ndarray:
        y = x.detach()
        if y.device.type != "cpu":
            y = y.cpu()
        if y.ndim == 3 and y.shape[-1] == 1:
            y = y[..., 0]
        if y.dtype != torch.uint8:
            if y.is_floating_point():
                y = torch.where(y > 0.5, 255, 0).to(torch.uint8)
            else:
                y = y.to(torch.uint8)
        return np.ascontiguousarray(y.contiguous().numpy())

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
    def _unpad_hw_numpy(x: np.ndarray, pad: tuple[int, int, int, int]) -> np.ndarray:
        pt, pb, pl, pr = [int(v) for v in pad]
        h, w = int(x.shape[0]), int(x.shape[1])
        y0 = pt
        y1 = h - pb if pb > 0 else h
        x0 = pl
        x1 = w - pr if pr > 0 else w
        return np.ascontiguousarray(x[y0:y1, x0:x1])

    @staticmethod
    def _pad_hwc_numpy(x: np.ndarray, pad: tuple[int, int, int, int], clip_size: int) -> np.ndarray:
        pt, pb, pl, pr = [int(v) for v in pad]
        out = np.zeros((int(clip_size), int(clip_size), 3), dtype=x.dtype)
        h, w = int(x.shape[0]), int(x.shape[1])
        out[pt:pt + h, pl:pl + w, :] = x
        return out

    def _maybe_copy_guard(self, x: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if x is None:
            return None
        arr = np.ascontiguousarray(x)
        if self.copy_guard:
            arr = arr.copy()
        return arr

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

    def _save_debug_mask(self, frame_num: int, name: str, mask: np.ndarray) -> None:
        if not self._debug_this_frame(frame_num):
            return
        arr = np.asarray(mask)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            return
        if arr.dtype != np.uint8:
            y = arr.astype(np.float32, copy=False)
            if y.max() <= 1.0:
                y = y * 255.0
            y = np.clip(y, 0.0, 255.0).round().astype(np.uint8)
        else:
            y = arr
        out = self.debug_dir / f"f{frame_num:06d}_{name}.png"
        cv2.imwrite(str(out), y)

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

    @staticmethod
    def _clone_face_meta(face_meta: FaceMetadata) -> FaceMetadata:
        kps = None
        if face_meta.kps is not None:
            kps = face_meta.kps.clone().to(dtype=torch.float32)
        return FaceMetadata(
            bbox_xyxy=tuple(float(v) for v in face_meta.bbox_xyxy),
            kps=kps,
            det_score=face_meta.det_score,
        )

    @staticmethod
    def _face_metrics(face_meta: FaceMetadata, crop_shape: tuple[int, int, int]) -> dict[str, float]:
        crop_h, crop_w = int(crop_shape[0]), int(crop_shape[1])
        x1, y1, x2, y2 = [float(v) for v in face_meta.bbox_xyxy]
        face_w = max(1.0, x2 - x1)
        face_h = max(1.0, y2 - y1)
        area = face_w * face_h
        crop_area = float(max(1, crop_h * crop_w))
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        crop_cx = 0.5 * crop_w
        crop_cy = 0.5 * crop_h
        dx = cx - crop_cx
        dy = cy - crop_cy
        half_diag = max(1e-6, float(np.hypot(0.5 * crop_w, 0.5 * crop_h)))
        center_dist_norm = float(np.hypot(dx, dy) / half_diag)
        return {
            "face_w": face_w,
            "face_h": face_h,
            "area": area,
            "area_ratio": float(area / crop_area),
            "w_ratio": float(face_w / max(1, crop_w)),
            "h_ratio": float(face_h / max(1, crop_h)),
            "center_dist_norm": center_dist_norm,
        }

    @staticmethod
    def _face_from_app_candidate(candidate) -> Optional[FaceMetadata]:
        try:
            bbox = getattr(candidate, "bbox", None)
            if bbox is None or len(bbox) < 4:
                return None
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            kps = getattr(candidate, "kps", None)
            if kps is not None:
                kps = torch.as_tensor(np.asarray(kps, dtype=np.float32)).clone()
            det_score = getattr(candidate, "det_score", None)
            if det_score is not None:
                det_score = float(det_score)
            return FaceMetadata(
                bbox_xyxy=(x1, y1, x2, y2),
                kps=kps,
                det_score=det_score,
            )
        except Exception:
            return None

    @staticmethod
    def _selector_should_run(face_meta: FaceMetadata, crop_np: np.ndarray) -> tuple[bool, str]:
        m = BaseFaceSwapClipRestorer._face_metrics(face_meta, crop_np.shape)
        suspicious = (
            m["area_ratio"] < 0.05
            or m["w_ratio"] < 0.18
            or m["h_ratio"] < 0.18
        )
        if suspicious:
            return True, (
                f"suspicious_current:"
                f"area_ratio={m['area_ratio']:.4f} "
                f"w_ratio={m['w_ratio']:.4f} "
                f"h_ratio={m['h_ratio']:.4f}"
            )
        return False, (
            f"current_ok:"
            f"area_ratio={m['area_ratio']:.4f} "
            f"w_ratio={m['w_ratio']:.4f} "
            f"h_ratio={m['h_ratio']:.4f}"
        )

    @staticmethod
    def _selector_score(face_meta: FaceMetadata, crop_np: np.ndarray) -> float:
        m = BaseFaceSwapClipRestorer._face_metrics(face_meta, crop_np.shape)
        det_score = float(face_meta.det_score) if face_meta.det_score is not None else 1.0
        return (
            2.0 * m["area_ratio"]
            + 0.05 * det_score
            - 0.15 * m["center_dist_norm"]
        )

    def _run_target_face_selector(
        self,
        crop_np: np.ndarray,
        current_face_meta: FaceMetadata,
        frame_num: int,
    ) -> tuple[FaceMetadata, str]:
        should_run, why = self._selector_should_run(current_face_meta, crop_np)
        if not should_run:
            return current_face_meta, f"selector_skip:{why}"

        self.stats.frames_selector_called += 1
        try:
            raw_candidates = self.worker._app.get(crop_np)  # reuse worker's FaceAnalysis stack
        except Exception as e:
            self._save_debug_text(frame_num, f"selector_exception={e!r}")
            self.stats.frames_selector_no_viable += 1
            return current_face_meta, "selector_error"

        candidates: list[FaceMetadata] = []
        for cand in raw_candidates or []:
            fm = self._face_from_app_candidate(cand)
            if fm is not None:
                candidates.append(fm)

        self.stats.frames_selector_candidates_found += len(candidates)
        if not candidates:
            self.stats.frames_selector_no_viable += 1
            return current_face_meta, "selector_no_candidates"

        viable: list[tuple[float, FaceMetadata, dict[str, float]]] = []
        for cand in candidates:
            m = self._face_metrics(cand, crop_np.shape)
            if m["area_ratio"] < 0.015:
                continue
            score = self._selector_score(cand, crop_np)
            viable.append((score, cand, m))

        if not viable:
            self.stats.frames_selector_no_viable += 1
            return current_face_meta, "selector_no_viable"

        viable.sort(key=lambda x: x[0], reverse=True)
        best_score, best_face, best_metrics = viable[0]

        cur_metrics = self._face_metrics(current_face_meta, crop_np.shape)
        cur_score = self._selector_score(current_face_meta, crop_np)
        area_gain = best_metrics["area"] / max(1.0, cur_metrics["area"])

        replace = (
            (cur_metrics["area_ratio"] < 0.05 and area_gain >= 3.0)
            or (best_score > cur_score + 0.10 and area_gain >= 1.5)
        )

        if replace:
            self.stats.frames_selector_replaced += 1
            return best_face, (
                f"selector_replaced:"
                f"cur_area_ratio={cur_metrics['area_ratio']:.4f} "
                f"best_area_ratio={best_metrics['area_ratio']:.4f} "
                f"area_gain={area_gain:.2f} "
                f"cur_score={cur_score:.4f} "
                f"best_score={best_score:.4f}"
            )

        self.stats.frames_selector_kept += 1
        return current_face_meta, (
            f"selector_kept:"
            f"cur_area_ratio={cur_metrics['area_ratio']:.4f} "
            f"best_area_ratio={best_metrics['area_ratio']:.4f} "
            f"area_gain={area_gain:.2f} "
            f"cur_score={cur_score:.4f} "
            f"best_score={best_score:.4f}"
        )


    @staticmethod
    def _bbox_to_five_points(face_meta: FaceMetadata) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in face_meta.bbox_xyxy]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return np.array(
            [
                [x1 + 0.32 * w, y1 + 0.38 * h],
                [x1 + 0.68 * w, y1 + 0.38 * h],
                [x1 + 0.50 * w, y1 + 0.56 * h],
                [x1 + 0.37 * w, y1 + 0.75 * h],
                [x1 + 0.63 * w, y1 + 0.75 * h],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _landmarks68_to_five_points(pts68: np.ndarray) -> np.ndarray:
        pts68 = np.asarray(pts68, dtype=np.float32)
        left_eye = np.mean(pts68[36:42], axis=0)
        right_eye = np.mean(pts68[42:48], axis=0)
        nose = pts68[30]
        mouth_left = pts68[48]
        mouth_right = pts68[54]
        return np.stack([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0).astype(np.float32)

    @staticmethod
    def _mask_bbox(mask_u8: Optional[np.ndarray]) -> Optional[tuple[float, float, float, float]]:
        if mask_u8 is None:
            return None
        arr = np.asarray(mask_u8)
        if arr.ndim == 3:
            arr = arr[..., 0]
        ys, xs = np.where(arr > 0)
        if xs.size <= 0 or ys.size <= 0:
            return None
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    @staticmethod
    def _tighten_detector_roi_bbox(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        # FaceDetector expands tight detector boxes by approximately:
        # side=15%, top=5%, bottom=10%. When only the ROI/mask survives,
        # invert that expansion to synthesize a tighter face box.
        w = max(1.0, float(x2) - float(x1))
        h = max(1.0, float(y2) - float(y1))
        return (
            float(x1) + 0.1154 * w,
            float(y1) + 0.0435 * h,
            float(x2) - 0.1154 * w,
            float(y2) - 0.0870 * h,
        )

    @staticmethod
    def _median_bbox(face_metas: List[Optional[FaceMetadata]], indices: list[int]) -> Optional[np.ndarray]:
        vals = []
        for j in indices:
            fm = face_metas[j]
            if fm is None:
                continue
            try:
                arr = np.asarray(fm.bbox_xyxy, dtype=np.float32)
                if arr.shape == (4,) and np.all(np.isfinite(arr)):
                    vals.append(arr)
            except Exception:
                continue
        if not vals:
            return None
        return np.median(np.stack(vals, axis=0), axis=0).astype(np.float32)

    @staticmethod
    def _median_kps(face_metas: List[Optional[FaceMetadata]], indices: list[int], shape: tuple[int, int]) -> Optional[np.ndarray]:
        vals = []
        for j in indices:
            fm = face_metas[j]
            if fm is None or fm.kps is None:
                continue
            try:
                arr = fm.kps.detach().cpu().numpy().astype(np.float32, copy=True)
                if arr.shape == shape and np.all(np.isfinite(arr)):
                    vals.append(arr)
            except Exception:
                continue
        if not vals:
            return None
        return np.median(np.stack(vals, axis=0), axis=0).astype(np.float32)

    @staticmethod
    def _clamp_array_jump(current: np.ndarray, previous: Optional[np.ndarray], max_jump_px: float) -> np.ndarray:
        if previous is None or max_jump_px <= 0.0:
            return current
        cur = np.asarray(current, dtype=np.float32)
        prev = np.asarray(previous, dtype=np.float32)
        if cur.shape != prev.shape:
            return current
        delta = cur - prev
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        if max_abs <= max_jump_px or max_abs <= 1e-6:
            return current
        return (prev + delta * (float(max_jump_px) / max_abs)).astype(np.float32)

    def _stabilize_face_geometry(
        self,
        face_metas: List[Optional[FaceMetadata]],
        crops: List[np.ndarray],
        frame_nums: List[int],
    ) -> tuple[List[Optional[FaceMetadata]], list[dict]]:
        infos: list[dict] = []
        if not self.geometry_stabilization_enabled or not face_metas:
            return face_metas, [
                {"geometry_stabilized": False, "geometry_reason": "disabled"}
                for _ in face_metas
            ]

        n = len(face_metas)
        half = max(0, int(self.geometry_stabilization_window) // 2)
        out: List[Optional[FaceMetadata]] = []
        prev_bbox: Optional[np.ndarray] = None
        prev_kps: Optional[np.ndarray] = None

        for i, fm in enumerate(face_metas):
            info = {
                "geometry_stabilized": False,
                "geometry_reason": "no_face_meta",
                "geometry_bbox_shift": 0.0,
                "geometry_kps_shift": 0.0,
            }
            if fm is None:
                out.append(None)
                infos.append(info)
                continue

            indices = list(range(max(0, i - half), min(n, i + half + 1)))
            orig_bbox = np.asarray(fm.bbox_xyxy, dtype=np.float32)
            new_bbox = orig_bbox.copy()
            med_bbox = self._median_bbox(face_metas, indices)
            if med_bbox is not None and self.geometry_stabilization_bbox > 0.0:
                w = float(self.geometry_stabilization_bbox)
                new_bbox = orig_bbox * (1.0 - w) + med_bbox * w

            # Keep bbox ordered and inside the current crop.
            h, w_crop = int(crops[i].shape[0]), int(crops[i].shape[1])
            x1, y1, x2, y2 = [float(v) for v in new_bbox]
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            new_bbox = np.array([
                np.clip(x1, 0.0, max(0.0, float(w_crop - 1))),
                np.clip(y1, 0.0, max(0.0, float(h - 1))),
                np.clip(x2, 0.0, max(0.0, float(w_crop - 1))),
                np.clip(y2, 0.0, max(0.0, float(h - 1))),
            ], dtype=np.float32)
            new_bbox = self._clamp_array_jump(new_bbox, prev_bbox, self.geometry_stabilization_max_bbox_jump_px)

            kps_t = None
            orig_kps_arr = None
            new_kps = None
            if fm.kps is not None:
                try:
                    orig_kps_arr = fm.kps.detach().cpu().numpy().astype(np.float32, copy=True)
                    if orig_kps_arr.ndim == 2 and orig_kps_arr.shape[1] == 2:
                        new_kps = orig_kps_arr.copy()
                        med_kps = self._median_kps(face_metas, indices, tuple(orig_kps_arr.shape))
                        if med_kps is not None and self.geometry_stabilization_kps > 0.0:
                            wk = float(self.geometry_stabilization_kps)
                            new_kps = orig_kps_arr * (1.0 - wk) + med_kps * wk
                        new_kps = self._clamp_array_jump(new_kps, prev_kps, self.geometry_stabilization_max_kps_jump_px)
                        if w_crop > 0:
                            new_kps[:, 0] = np.clip(new_kps[:, 0], 0.0, float(max(0, w_crop - 1)))
                        if h > 0:
                            new_kps[:, 1] = np.clip(new_kps[:, 1], 0.0, float(max(0, h - 1)))
                        kps_t = torch.from_numpy(np.ascontiguousarray(new_kps)).to(dtype=torch.float32)
                except Exception:
                    kps_t = fm.kps.clone().to(dtype=torch.float32) if fm.kps is not None else None

            bbox_shift = float(np.mean(np.abs(new_bbox - orig_bbox))) if orig_bbox.shape == new_bbox.shape else 0.0
            kps_shift = 0.0
            if orig_kps_arr is not None and new_kps is not None and orig_kps_arr.shape == new_kps.shape:
                kps_shift = float(np.mean(np.abs(new_kps - orig_kps_arr)))

            if bbox_shift > 1e-4 or kps_shift > 1e-4:
                self.stats.frames_geometry_stabilized += 1
                if bbox_shift > 1e-4:
                    self.stats.geometry_bbox_shift_accum += bbox_shift
                    self.stats.geometry_bbox_shift_count += 1
                if kps_shift > 1e-4:
                    self.stats.geometry_kps_shift_accum += kps_shift
                    self.stats.geometry_kps_shift_count += 1
                info.update({
                    "geometry_stabilized": True,
                    "geometry_reason": "smoothed",
                    "geometry_bbox_shift": bbox_shift,
                    "geometry_kps_shift": kps_shift,
                })
            else:
                info.update({"geometry_reason": "unchanged"})

            prev_bbox = new_bbox.copy()
            if new_kps is not None:
                prev_kps = new_kps.copy()

            out.append(
                FaceMetadata(
                    bbox_xyxy=tuple(float(v) for v in new_bbox),
                    kps=kps_t,
                    det_score=fm.det_score,
                )
            )
            infos.append(info)

        return out, infos

    def _synthesize_face_meta(
        self,
        crop_np: np.ndarray,
        crop_mask_np: Optional[np.ndarray],
    ) -> tuple[Optional[FaceMetadata], str]:
        h, w = int(crop_np.shape[0]), int(crop_np.shape[1])
        if h <= 1 or w <= 1:
            return None, "synthetic_face_meta=invalid_crop"

        bbox = self._mask_bbox(crop_mask_np)
        source = "mask_bbox"
        if bbox is None:
            # Last-resort center-ish box. This is intentionally used only for
            # force-swap mode when a detector ROI reached this restorer without
            # usable FaceMetadata.
            bbox = (0.0, 0.0, float(w - 1), float(h - 1))
            source = "crop_bbox"

        x1, y1, x2, y2 = self._tighten_detector_roi_bbox(*bbox)
        x1 = float(np.clip(x1, 0.0, float(max(0, w - 1))))
        x2 = float(np.clip(x2, 0.0, float(max(0, w - 1))))
        y1 = float(np.clip(y1, 0.0, float(max(0, h - 1))))
        y2 = float(np.clip(y2, 0.0, float(max(0, h - 1))))
        if x2 <= x1 + 2.0 or y2 <= y1 + 2.0:
            return None, f"synthetic_face_meta={source}_invalid_bbox"

        meta = FaceMetadata(
            bbox_xyxy=(x1, y1, x2, y2),
            kps=None,
            det_score=None,
        )
        meta.kps = torch.from_numpy(np.ascontiguousarray(self._bbox_to_five_points(meta))).to(torch.float32)
        return meta, f"synthetic_face_meta={source}"

    def _ensure_swappable_face_meta(self, face_meta: FaceMetadata) -> tuple[FaceMetadata, str]:
        if not self.allow_bbox_landmark_fallback:
            return face_meta, "landmark_fallback=disabled"

        kps = None
        if face_meta.kps is not None:
            try:
                kps = face_meta.kps.detach().cpu().numpy().astype(np.float32, copy=True)
            except Exception:
                kps = None

        replacement = None
        reason = "landmark_fallback=not_needed"
        if kps is None or kps.ndim != 2 or kps.shape[1] != 2 or kps.shape[0] < 5 or not np.isfinite(kps).all():
            replacement = self._bbox_to_five_points(face_meta)
            reason = "landmark_fallback=bbox5"
        elif kps.shape[0] >= 68 and kps.shape[0] < 100:
            replacement = self._landmarks68_to_five_points(kps[:68])
            reason = "landmark_fallback=landmarks68_to_5"

        if replacement is None:
            return face_meta, reason

        self.stats.frames_bbox_landmark_fallback += 1
        new_meta = self._clone_face_meta(face_meta)
        new_meta.kps = torch.from_numpy(np.ascontiguousarray(replacement)).to(torch.float32)
        return new_meta, reason

    def get_stats_lines(self) -> List[str]:
        return [
            f"[FaceSwapStats] clips_processed={self.stats.clips_processed} frames_total={self.stats.frames_total}",
            f"[FaceSwapStats] detector_face_meta frames={self.stats.frames_with_detector_face_meta}/{self.stats.frames_total} missing={self.stats.frames_without_detector_face_meta}",
            f"[FaceSwapStats] gap_fill forward={self.stats.frames_gap_filled_forward} backward={self.stats.frames_gap_filled_backward}",
            f"[FaceSwapStats] force_swap attempted={self.stats.frames_force_swap_attempted} detector_authoritative={self.stats.frames_detector_face_meta_authoritative} synthetic_meta={self.stats.frames_synthetic_face_meta_used}",
            f"[FaceSwapStats] bbox_landmark_fallback={self.stats.frames_bbox_landmark_fallback} previous_swap_reused={self.stats.frames_previous_swap_reused}",
            f"[FaceSwapStats] metrics_written={self.stats.frames_metrics_written} metrics_path={str(self.metrics_path) if self.metrics_path is not None else '-'}",
            f"[FaceSwapStats] embedding mixed_used={self.stats.frames_mixed_embedding_used} target_used={self.stats.frames_target_embedding_used} target_missing={self.stats.frames_target_embedding_missing}",
            f"[FaceSwapStats] embedding_cos avg_src_target={self.stats.avg_embedding_source_to_target_cos():.4f} avg_src_mixed={self.stats.avg_embedding_source_to_mixed_cos():.4f} avg_target_mixed={self.stats.avg_embedding_target_to_mixed_cos():.4f}",
            f"[FaceSwapStats] geometry_stabilized={self.stats.frames_geometry_stabilized} avg_bbox_shift={self.stats.avg_geometry_bbox_shift():.4f} avg_kps_shift={self.stats.avg_geometry_kps_shift():.4f}",
            f"[FaceSwapStats] landmarker_called={self.stats.frames_landmarker_called} returned={self.stats.frames_landmarker_returned} failed={self.stats.frames_landmarker_failed}",
            f"[FaceSwapStats] selector_called={self.stats.frames_selector_called} candidates_found={self.stats.frames_selector_candidates_found} replaced={self.stats.frames_selector_replaced}",
            f"[FaceSwapStats] selector_kept={self.stats.frames_selector_kept} selector_no_viable={self.stats.frames_selector_no_viable}",
            f"[FaceSwapStats] worker_called={self.stats.frames_worker_called} returned={self.stats.frames_worker_returned} returned_none={self.stats.frames_worker_returned_none}",
            f"[FaceSwapStats] worker_exceptions={self.stats.worker_exceptions} last_worker_exception={self.stats.last_worker_exception or '-'}",
            f"[FaceSwapStats] materially_changed={self.stats.frames_materially_changed} mad_threshold={self.material_change_mad_threshold:.3f} avg_mad={self.stats.avg_mean_abs_diff():.4f}",
            f"[FaceSwapStats] expression_called={self.stats.frames_expression_called} returned={self.stats.frames_expression_returned} failed={self.stats.frames_expression_failed}",
            f"[FaceSwapStats] expression_materially_changed={self.stats.frames_expression_materially_changed} avg_expression_mad={self.stats.avg_expression_mean_abs_diff():.4f}",
            f"[FaceSwapStats] enhancer_called={self.stats.frames_enhancer_called} returned={self.stats.frames_enhancer_returned} failed={self.stats.frames_enhancer_failed}",
            f"[FaceSwapStats] enhancer_materially_changed={self.stats.frames_enhancer_materially_changed} avg_enhancer_mad={self.stats.avg_enhancer_mean_abs_diff():.4f}",
            f"[FaceSwapStats] occluder_called={self.stats.frames_occluder_called} returned={self.stats.frames_occluder_returned} failed={self.stats.frames_occluder_failed}",
            f"[FaceSwapStats] occluder_materially_changed={self.stats.frames_occluder_materially_changed} avg_occluder_mad={self.stats.avg_occluder_mean_abs_diff():.4f}",
        ]

    @staticmethod
    def _json_safe(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): BaseFaceSwapClipRestorer._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [BaseFaceSwapClipRestorer._json_safe(v) for v in value]
        return str(value)

    def _write_metrics_event(self, payload: dict) -> None:
        if not self.metrics_enabled or self.metrics_path is None:
            return
        try:
            safe = self._json_safe(payload)
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(safe, sort_keys=True, separators=(",", ":")) + "\n")
            if safe.get("event") == "face_swap_frame":
                self.stats.frames_metrics_written += 1
        except Exception as e:
            # Metrics must never fail the video run.
            if self._swap_exception_print_budget > 0:
                print(f"[FaceSwap][metrics_exception] {e!r}")
                self._swap_exception_print_budget -= 1

    def _read_worker_metrics(self) -> dict:
        metrics = getattr(self.worker, "last_swap_metrics", None)
        if isinstance(metrics, dict):
            return dict(metrics)
        return {}

    def _accumulate_worker_metrics(self, worker_metrics: dict) -> None:
        if not worker_metrics:
            return
        if bool(worker_metrics.get("mixed_embedding_used", False)):
            self.stats.frames_mixed_embedding_used += 1
        if bool(worker_metrics.get("target_embedding_used", False)):
            self.stats.frames_target_embedding_used += 1
        if bool(worker_metrics.get("target_embedding_missing", False)):
            self.stats.frames_target_embedding_missing += 1
        for key, accum_name, count_name in (
            ("source_to_target_cos", "embedding_source_to_target_cos_accum", "embedding_source_to_target_cos_count"),
            ("source_to_mixed_cos", "embedding_source_to_mixed_cos_accum", "embedding_source_to_mixed_cos_count"),
            ("target_to_mixed_cos", "embedding_target_to_mixed_cos_accum", "embedding_target_to_mixed_cos_count"),
        ):
            value = worker_metrics.get(key)
            if value is None:
                continue
            try:
                v = float(value)
            except Exception:
                continue
            if np.isfinite(v):
                setattr(self.stats, accum_name, getattr(self.stats, accum_name) + v)
                setattr(self.stats, count_name, getattr(self.stats, count_name) + 1)

    def _apply_global_face_postprocess(
        self,
        *,
        frame_num: int,
        original_roi: np.ndarray,
        swapped_roi: np.ndarray,
        active_face_meta: FaceMetadata,
        stage_metrics: Optional[dict] = None,
    ) -> np.ndarray:
        """Apply global ROI-space post-swap stages shared by all swappers.

        Swappers are intentionally responsible for their own align/infer/native
        paste-back semantics. This method is the common control point for
        enhancer and occluder policy after a worker has returned a same-size ROI.
        """
        stage_img = swapped_roi
        if stage_metrics is None:
            stage_metrics = {}

        if self.expression_restorer is not None:
            stage_metrics["expression_called"] = True
            self.stats.frames_expression_called += 1
            try:
                expression_restored = self.expression_restorer.restore(original_roi, stage_img, active_face_meta)
            except Exception as e:
                self.stats.frames_expression_failed += 1
                stage_metrics["expression_failed"] = True
                stage_metrics["expression_exception"] = repr(e)
                self._save_debug_text(frame_num, f"expression_restorer_exception={e!r}")
                expression_restored = None
            if expression_restored is not None:
                expression_restored = self._maybe_copy_guard(expression_restored)
                self.stats.frames_expression_returned += 1
                stage_metrics["expression_returned"] = True
                xmad = float(np.mean(np.abs(expression_restored.astype(np.int16) - stage_img.astype(np.int16))))
                stage_metrics["expression_mad"] = xmad
                self.stats.expression_mean_abs_diff_accum += xmad
                if xmad >= self.material_change_mad_threshold:
                    self.stats.frames_expression_materially_changed += 1
                self._save_debug_text(frame_num, f"expression_restorer_mean_abs_diff={xmad:.4f}")
                self._save_debug_image(frame_num, "03a_expression_restored", expression_restored)
                self._save_debug_image(frame_num, "04a_expression_diff", self._diff_image(stage_img, expression_restored))
                stage_img = expression_restored

        if self.enhancer is not None:
            stage_metrics["enhancer_called"] = True
            self.stats.frames_enhancer_called += 1
            try:
                enhanced = self.enhancer.enhance(stage_img, active_face_meta)
            except Exception as e:
                self.stats.frames_enhancer_failed += 1
                stage_metrics["enhancer_failed"] = True
                stage_metrics["enhancer_exception"] = repr(e)
                self._save_debug_text(frame_num, f"enhancer_exception={e!r}")
                enhanced = None
            if enhanced is not None:
                enhanced = self._maybe_copy_guard(enhanced)
                self.stats.frames_enhancer_returned += 1
                stage_metrics["enhancer_returned"] = True
                emad = float(np.mean(np.abs(enhanced.astype(np.int16) - stage_img.astype(np.int16))))
                stage_metrics["enhancer_mad"] = emad
                self.stats.enhancer_mean_abs_diff_accum += emad
                if emad >= self.material_change_mad_threshold:
                    self.stats.frames_enhancer_materially_changed += 1
                self._save_debug_text(frame_num, f"enhancer_mean_abs_diff={emad:.4f}")
                self._save_debug_image(frame_num, "03b_enhanced", enhanced)
                self._save_debug_image(frame_num, "04b_enhancer_diff", self._diff_image(stage_img, enhanced))
                stage_img = enhanced

        if self.occluder is not None:
            stage_metrics["occluder_called"] = True
            self.stats.frames_occluder_called += 1
            try:
                occluded = self.occluder.preserve(original_roi, stage_img, active_face_meta)
            except Exception as e:
                self.stats.frames_occluder_failed += 1
                stage_metrics["occluder_failed"] = True
                stage_metrics["occluder_exception"] = repr(e)
                self._save_debug_text(frame_num, f"occluder_exception={e!r}")
                occluded = None
            if occluded is not None:
                occluded = self._maybe_copy_guard(occluded)
                self.stats.frames_occluder_returned += 1
                stage_metrics["occluder_returned"] = True
                omad = float(np.mean(np.abs(occluded.astype(np.int16) - stage_img.astype(np.int16))))
                stage_metrics["occluder_mad"] = omad
                self.stats.occluder_mean_abs_diff_accum += omad
                if omad >= self.material_change_mad_threshold:
                    self.stats.frames_occluder_materially_changed += 1
                self._save_debug_text(frame_num, f"occluder_mean_abs_diff={omad:.4f}")
                self._save_debug_image(frame_num, "03c_occluded", occluded)
                self._save_debug_image(frame_num, "04c_occluder_diff", self._diff_image(stage_img, occluded))
                stage_img = occluded

        return self._maybe_copy_guard(stage_img)

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
        crop_masks_np: list[Optional[np.ndarray]] = []

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

            clip_mask = clip.masks[i] if hasattr(clip, "masks") and i < len(clip.masks) else None
            crop_mask_np = None
            if clip_mask is not None:
                try:
                    clip_mask_np = self._tensor_hw_to_numpy_u8(clip_mask)
                    crop_mask_np = self._unpad_hw_numpy(clip_mask_np, pad)
                    if int(crop_mask_np.shape[0]) != int(crop_h) or int(crop_mask_np.shape[1]) != int(crop_w):
                        crop_mask_np = cv2.resize(
                            crop_mask_np,
                            (int(crop_w), int(crop_h)),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    crop_mask_np = np.ascontiguousarray(crop_mask_np)
                except Exception:
                    crop_mask_np = None

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
            crop_masks_np.append(crop_mask_np)

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
        selected_faces, geometry_infos = self._stabilize_face_geometry(selected_faces, crops, frame_nums)
        for _frame_num, _ginfo in zip(frame_nums, geometry_infos):
            if _ginfo.get("geometry_stabilized"):
                self._save_debug_text(
                    int(_frame_num),
                    "geometry_stabilized=True "
                    f"bbox_shift={float(_ginfo.get('geometry_bbox_shift', 0.0)):.4f} "
                    f"kps_shift={float(_ginfo.get('geometry_kps_shift', 0.0)):.4f}",
                )
        last_good_swapped_np: Optional[np.ndarray] = None

        for i, crop_np in enumerate(crops):
            frame_num = frame_nums[i]
            face_meta = selected_faces[i]
            synthetic_reason = "synthetic_face_meta=not_needed"
            synthetic_used = False
            if face_meta is None and self.force_swap_detected_faces and self.force_swap_synthesize_meta:
                synthetic_meta, synthetic_reason = self._synthesize_face_meta(crop_np, crop_masks_np[i] if i < len(crop_masks_np) else None)
                if synthetic_meta is not None:
                    face_meta = synthetic_meta
                    synthetic_used = True
                    self.stats.frames_synthetic_face_meta_used += 1

            swapped_np = self._maybe_copy_guard(crop_np)
            frame_metrics = {
                "event": "face_swap_frame",
                "clip_id": str(clip.id),
                "frame_num": int(frame_num),
                "frame_index_in_clip": int(i),
                "crop_shape": [int(crop_np.shape[0]), int(crop_np.shape[1]), int(crop_np.shape[2])],
                "has_selected_face_meta": face_meta is not None,
                "worker_backend": type(self.worker).__name__,
                "face_swapper_weight": self.face_swapper_weight,
                "swap_policy": self.swap_policy,
                "force_swap_detected_faces": self.force_swap_detected_faces,
                "force_swap_synthesize_meta": self.force_swap_synthesize_meta,
                "force_swap_attempt": False,
                "synthetic_face_meta_used": synthetic_used,
                "synthetic_face_meta_reason": synthetic_reason,
                "allow_bbox_landmark_fallback": self.allow_bbox_landmark_fallback,
                "reuse_previous_on_swap_failure": self.reuse_previous_on_swap_failure,
                "worker_called": False,
                "worker_returned": False,
                "worker_returned_none": False,
                "worker_exception": None,
                "previous_swap_reused": False,
            }
            if i < len(geometry_infos):
                frame_metrics.update(geometry_infos[i])

            if synthetic_reason != "synthetic_face_meta=not_needed":
                self._save_debug_text(frame_num, synthetic_reason)

            if face_meta is not None:
                original_roi = crop_np.copy()
                before = crop_np.copy()
                active_face_meta = self._clone_face_meta(face_meta)

                if self.force_swap_detected_faces:
                    selector_reason = "selector_skip:force_swap_detected_faces"
                    self.stats.frames_detector_face_meta_authoritative += 1
                else:
                    active_face_meta, selector_reason = self._run_target_face_selector(crop_np, active_face_meta, frame_num)
                frame_metrics["selector_reason"] = selector_reason
                self._save_debug_text(frame_num, selector_reason)

                if self.landmarker is not None:
                    self.stats.frames_landmarker_called += 1
                    try:
                        refined = self.landmarker.refine(crop_np, active_face_meta)
                    except Exception as e:
                        self.stats.frames_landmarker_failed += 1
                        self._save_debug_text(frame_num, f"landmarker_exception={e!r}")
                        refined = None
                    if refined is not None:
                        active_face_meta = refined
                        self.stats.frames_landmarker_returned += 1
                        self._save_debug_text(frame_num, "landmarker_refined=True")
                    else:
                        self._save_debug_text(frame_num, "landmarker_refined=False")

                active_face_meta, landmark_policy_reason = self._ensure_swappable_face_meta(active_face_meta)
                frame_metrics["landmark_policy"] = landmark_policy_reason
                frame_metrics["bbox_landmark_fallback_used"] = landmark_policy_reason in ("landmark_fallback=bbox5", "landmark_fallback=landmarks68_to_5")
                self._save_debug_text(frame_num, landmark_policy_reason)

                x1, y1, x2, y2 = active_face_meta.bbox_xyxy
                frame_metrics["target_face_bbox"] = [float(x1), float(y1), float(x2), float(y2)]
                self._save_debug_text(frame_num, f"target_face_bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                if active_face_meta.kps is not None:
                    self._save_debug_text(frame_num, f"target_face_kps_shape={tuple(active_face_meta.kps.shape)}")

                if self.force_swap_detected_faces:
                    self.stats.frames_force_swap_attempted += 1
                    frame_metrics["force_swap_attempt"] = True
                self.stats.frames_worker_called += 1
                frame_metrics["worker_called"] = True
                out = None

                self._save_debug_text(
                    frame_num,
                    "worker_runtime="
                    f"{type(self.worker).__module__}.{type(self.worker).__name__} "
                    f"has_swap={hasattr(self.worker, 'swap')}"
                )
                self._save_debug_text(frame_num, "worker_branch=native_swap")

                try:
                    out = self.worker.swap(crop_np, active_face_meta)
                    if out is not None:
                        out = self._maybe_copy_guard(out)
                except Exception as e:
                    self.stats.worker_exceptions += 1
                    self.stats.last_worker_exception = repr(e)
                    frame_metrics["worker_exception"] = repr(e)
                    self._save_debug_text(frame_num, f"swap_exception={e!r}")
                    if self._swap_exception_print_budget > 0:
                        print(f"[FaceSwap][swap_exception] frame={frame_num} {e!r}")
                        self._swap_exception_print_budget -= 1
                    out = None

                worker_metrics = self._read_worker_metrics()
                frame_metrics["worker_metrics"] = worker_metrics
                self._accumulate_worker_metrics(worker_metrics)

                if out is None:
                    self.stats.frames_worker_returned_none += 1
                    frame_metrics["worker_returned_none"] = True
                    self._save_debug_text(frame_num, "worker_returned=None")
                    if self.reuse_previous_on_swap_failure and last_good_swapped_np is not None and last_good_swapped_np.shape == crop_np.shape:
                        swapped_np = self._maybe_copy_guard(last_good_swapped_np)
                        self.stats.frames_previous_swap_reused += 1
                        frame_metrics["previous_swap_reused"] = True
                        self._save_debug_text(frame_num, "swap_failure_fallback=previous_good")
                else:
                    self.stats.frames_worker_returned += 1
                    frame_metrics["worker_returned"] = True
                    mad = float(np.mean(np.abs(out.astype(np.int16) - before.astype(np.int16))))
                    frame_metrics["worker_mad"] = mad
                    self.stats.mean_abs_diff_accum += mad
                    if mad >= self.material_change_mad_threshold:
                        self.stats.frames_materially_changed += 1

                    self._save_debug_text(frame_num, f"worker_mean_abs_diff={mad:.4f}")
                    self._save_debug_image(frame_num, "02_swapped", out)
                    self._save_debug_image(frame_num, "04_swap_diff", self._diff_image(before, out))

                    stage_metrics = {}
                    swapped_np = self._apply_global_face_postprocess(
                        frame_num=frame_num,
                        original_roi=original_roi,
                        swapped_roi=out,
                        active_face_meta=active_face_meta,
                        stage_metrics=stage_metrics,
                    )
                    frame_metrics["stage_metrics"] = stage_metrics
                    if swapped_np is not None:
                        frame_metrics["postprocess_total_mad"] = float(np.mean(np.abs(swapped_np.astype(np.int16) - out.astype(np.int16))))
                    if swapped_np is not None and swapped_np.shape == crop_np.shape:
                        last_good_swapped_np = self._maybe_copy_guard(swapped_np.copy())

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
            frame_metrics["clip_return_shape"] = [int(swapped_clip_np.shape[0]), int(swapped_clip_np.shape[1]), int(swapped_clip_np.shape[2])]
            self._save_debug_image(frame_num, "05_clip_return", swapped_clip_np)
            self._write_metrics_event(frame_metrics)
            out_frames.append(self._numpy_bgr_u8_to_tensor_hwc_float(swapped_clip_np))

        self._write_metrics_event(
            {
                "event": "face_swap_clip_summary",
                "clip_id": str(clip.id),
                "frames_in_clip": len(crops),
                "stats": dict(self.stats.__dict__),
                "face_swapper_weight": self.face_swapper_weight,
                "swap_policy": self.swap_policy,
                "force_swap_detected_faces": self.force_swap_detected_faces,
                "force_swap_synthesize_meta": self.force_swap_synthesize_meta,
                "geometry_stabilization_enabled": self.geometry_stabilization_enabled,
                "geometry_stabilization_window": self.geometry_stabilization_window,
                "geometry_stabilization_bbox": self.geometry_stabilization_bbox,
                "geometry_stabilization_kps": self.geometry_stabilization_kps,
                "geometry_stabilization_max_bbox_jump_px": self.geometry_stabilization_max_bbox_jump_px,
                "geometry_stabilization_max_kps_jump_px": self.geometry_stabilization_max_kps_jump_px,
            }
        )
        return out_frames


__all__ = ["BaseFaceSwapClipRestorer", "FaceSwapRestoreStats"]