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
from gRestorer.restorer.face_compositor import FaceCompositor
from gRestorer.restorer.face_enhancer import FaceEnhancer
from gRestorer.restorer.face_occluder import FaceOccluder
from gRestorer.restorer.face_landmarker import FaceLandmarker
from gRestorer.restorer.face_types import FaceCompositorConfig, FaceSwapBackendResult


@dataclass
class FaceSwapRestoreStats:
    clips_processed: int = 0
    frames_total: int = 0
    frames_with_detector_face_meta: int = 0
    frames_without_detector_face_meta: int = 0
    frames_gap_filled_forward: int = 0
    frames_gap_filled_backward: int = 0
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
    frames_materially_changed: int = 0
    mean_abs_diff_accum: float = 0.0
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

    def avg_mean_abs_diff(self) -> float:
        return self.mean_abs_diff_accum / max(1, self.frames_worker_returned)

    def avg_enhancer_mean_abs_diff(self) -> float:
        return self.enhancer_mean_abs_diff_accum / max(1, self.frames_enhancer_returned)

    def avg_occluder_mean_abs_diff(self) -> float:
        return self.occluder_mean_abs_diff_accum / max(1, self.frames_occluder_returned)

class BaseFaceSwapClipRestorer(BaseClipRestorer):
    """ROI-authoritative face-swap wrapper with enhancer and optional occluder.

    Policy lives here:
      - choose/stabilize the target face track within a clip
      - optional landmark refinement for the already chosen target face
      - shared face compositor for backends that can return aligned swap results
      - repad/resize behavior for the clip interface

    Worker contracts:
      legacy:
          swap(one_original_crop_image, one_target_face_metadata_in_that_crop_space)
      new-style:
          swap_result(one_original_crop_image, one_target_face_metadata_in_that_crop_space)
          -> FaceSwapBackendResult
    """

    def __init__(
        self,
        device: torch.device,
        source_face_path: str,
        swap_model_path: str,
        *,
        swap_input_size: int = 128,
        provider: str = "auto",
        face_comp_mask_mode: str = "geom_backend_intersection",
        face_comp_geom_expand: float = 1.05,
        face_comp_mask_erode: int = 0,
        face_comp_mask_dilate: int = 2,
        face_comp_mask_blur: int = 5,
        face_comp_blend_mode: str = "alpha",
        face_comp_color_transfer: str = "none",
        face_comp_face_scale: float = 0.0,
        face_comp_debug: bool = False,
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

        self.worker = self._build_worker()

        self.face_compositor = FaceCompositor(
            FaceCompositorConfig(
                mask_mode=str(face_comp_mask_mode or "geom_backend_intersection").lower(),
                geom_expand=float(face_comp_geom_expand),
                mask_erode=int(face_comp_mask_erode),
                mask_dilate=int(face_comp_mask_dilate),
                mask_blur=int(face_comp_mask_blur),
                blend_mode=str(face_comp_blend_mode or "alpha").lower(),
                color_transfer=str(face_comp_color_transfer or "none").lower(),
                face_scale=float(face_comp_face_scale),
                debug=bool(face_comp_debug),
            )
        )
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

    def _worker_supports_backend_result(self) -> bool:
        return hasattr(self.worker, "swap_result") and callable(getattr(self.worker, "swap_result"))

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

    def _save_backend_result_debug(self, frame_num: int, result: FaceSwapBackendResult) -> None:
        if not self._debug_this_frame(frame_num):
            return
        try:
            if result.aligned_target_bgr_u8 is not None:
                self._save_debug_image(frame_num, "03a_aligned_target", result.aligned_target_bgr_u8)
            self._save_debug_image(frame_num, "03b_aligned_swapped", result.aligned_swapped_bgr_u8)
            if result.aligned_backend_mask_f32 is not None:
                self._save_debug_mask(frame_num, "03c_backend_mask", result.aligned_backend_mask_f32)
            self._save_debug_text(
                frame_num,
                f"backend_result aligned_size={int(result.aligned_size)} "
                f"has_target={result.aligned_target_bgr_u8 is not None} "
                f"has_backend_mask={result.aligned_backend_mask_f32 is not None}",
            )
            if result.debug:
                self._save_debug_text(frame_num, f"backend_debug={result.debug}")
        except Exception as e:
            self._save_debug_text(frame_num, f"backend_result_debug_exception={e!r}")

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

    def get_stats_lines(self) -> List[str]:
        return [
            f"[FaceSwapStats] clips_processed={self.stats.clips_processed} frames_total={self.stats.frames_total}",
            f"[FaceSwapStats] detector_face_meta frames={self.stats.frames_with_detector_face_meta}/{self.stats.frames_total} missing={self.stats.frames_without_detector_face_meta}",
            f"[FaceSwapStats] gap_fill forward={self.stats.frames_gap_filled_forward} backward={self.stats.frames_gap_filled_backward}",
            f"[FaceSwapStats] landmarker_called={self.stats.frames_landmarker_called} returned={self.stats.frames_landmarker_returned} failed={self.stats.frames_landmarker_failed}",
            f"[FaceSwapStats] selector_called={self.stats.frames_selector_called} candidates_found={self.stats.frames_selector_candidates_found} replaced={self.stats.frames_selector_replaced}",
            f"[FaceSwapStats] selector_kept={self.stats.frames_selector_kept} selector_no_viable={self.stats.frames_selector_no_viable}",
            f"[FaceSwapStats] worker_called={self.stats.frames_worker_called} returned={self.stats.frames_worker_returned} returned_none={self.stats.frames_worker_returned_none}",
            f"[FaceSwapStats] materially_changed={self.stats.frames_materially_changed} mad_threshold={self.material_change_mad_threshold:.3f} avg_mad={self.stats.avg_mean_abs_diff():.4f}",
            f"[FaceSwapStats] enhancer_called={self.stats.frames_enhancer_called} returned={self.stats.frames_enhancer_returned} failed={self.stats.frames_enhancer_failed}",
            f"[FaceSwapStats] enhancer_materially_changed={self.stats.frames_enhancer_materially_changed} avg_enhancer_mad={self.stats.avg_enhancer_mean_abs_diff():.4f}",
            f"[FaceSwapStats] occluder_called={self.stats.frames_occluder_called} returned={self.stats.frames_occluder_returned} failed={self.stats.frames_occluder_failed}",
            f"[FaceSwapStats] occluder_materially_changed={self.stats.frames_occluder_materially_changed} avg_occluder_mad={self.stats.avg_occluder_mean_abs_diff():.4f}",
        ]

    def _save_compositor_debug(self, frame_num: int, debug: dict[str, np.ndarray]) -> None:
        if not (self.face_compositor.cfg.debug or self.debug_enabled):
            return
        if self.debug_start >= 0 and frame_num < self.debug_start:
            return
        if self.debug_end >= 0 and frame_num > self.debug_end:
            return

        self.debug_dir.mkdir(parents=True, exist_ok=True)

        def _save_mask(name: str, mask: np.ndarray) -> None:
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

        def _save_img(name: str, img: np.ndarray) -> None:
            arr = np.asarray(img)
            if arr.ndim != 3 or arr.shape[2] != 3:
                return
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)
            out = self.debug_dir / f"f{frame_num:06d}_{name}.png"
            cv2.imwrite(str(out), np.ascontiguousarray(arr))

        try:

            if "aligned_geom_mask_f32" in debug:
                _save_mask("03d_geom_mask", debug["aligned_geom_mask_f32"])
            if "aligned_pred_src_mask_f32" in debug:
                _save_mask("03e_pred_src_mask", debug["aligned_pred_src_mask_f32"])
            if "aligned_pred_dst_mask_f32" in debug:
                _save_mask("03f_pred_dst_mask", debug["aligned_pred_dst_mask_f32"])
            if "aligned_combined_mask_f32" in debug:
                _save_mask("03g_combined_mask", debug["aligned_combined_mask_f32"])
            if "roi_warped_alpha_f32" in debug:
                _save_mask("03h_warped_alpha", debug["roi_warped_alpha_f32"])
            if "roi_warped_face_bgr_u8" in debug:
                _save_img("03i_warped_face", debug["roi_warped_face_bgr_u8"])
            if "roi_keep_mask_f32" in debug:
                _save_mask("03j_keep_mask", debug["roi_keep_mask_f32"])

            txt_path = self.debug_dir / f"f{frame_num:06d}.txt"
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write(f"compositor_debug_keys={sorted(debug.keys())}\n")

        except Exception as e:
            txt_path = self.debug_dir / f"f{frame_num:06d}.txt"
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write(f"compositor_debug_exception={e!r}\n")


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
                original_roi = crop_np.copy()
                before = crop_np.copy()
                active_face_meta = self._clone_face_meta(face_meta)

                active_face_meta, selector_reason = self._run_target_face_selector(crop_np, active_face_meta, frame_num)
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

                x1, y1, x2, y2 = active_face_meta.bbox_xyxy
                self._save_debug_text(frame_num, f"target_face_bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                if active_face_meta.kps is not None:
                    self._save_debug_text(frame_num, f"target_face_kps_shape={tuple(active_face_meta.kps.shape)}")

                self.stats.frames_worker_called += 1
                out = None
                backend_result: Optional[FaceSwapBackendResult] = None

                self._save_debug_text(
                    frame_num,
                    "worker_runtime="
                    f"{type(self.worker).__module__}.{type(self.worker).__name__} "
                    f"has_swap={hasattr(self.worker, 'swap')} "
                    f"has_swap_result={hasattr(self.worker, 'swap_result')}"
                )

                if self._worker_supports_backend_result():
                    self._save_debug_text(frame_num, "worker_branch=backend_result")

                    try:
                        backend_result = self.worker.swap_result(crop_np, active_face_meta)
                    except Exception as e:
                        self._save_debug_text(frame_num, f"swap_result_exception={e!r}")
                        backend_result = None

                    if backend_result is not None:
                        self._save_backend_result_debug(frame_num, backend_result)
                        try:
                            if self.face_compositor.cfg.debug or self.debug_enabled:
                                out, comp_debug = self.face_compositor.compose_debug(
                                    original_roi_bgr_u8=crop_np,
                                    target_face_meta=active_face_meta,
                                    backend_result=backend_result,
                                    occlusion_keep_mask_f32=None,
                                )
                                self._save_compositor_debug(frame_num, comp_debug)
                                self._save_debug_text(frame_num, f"compositor_debug_keys={sorted(comp_debug.keys())}")
                            else:
                                out = self.face_compositor.compose(
                                    original_roi_bgr_u8=crop_np,
                                    target_face_meta=active_face_meta,
                                    backend_result=backend_result,
                                    occlusion_keep_mask_f32=None,
                                )
                        except Exception as e:
                            self._save_debug_text(frame_num, f"face_compositor_exception={e!r}")
                            out = None
                    else:
                        self._save_debug_text(frame_num, "swap_result_returned=None")
                else:
                    self._save_debug_text(frame_num, "worker_branch=legacy_swap")

                    try:
                        out = self.worker.swap(crop_np, active_face_meta)
                    except Exception as e:
                        self._save_debug_text(frame_num, f"swap_exception={e!r}")
                        out = None

                if out is None:
                    self.stats.frames_worker_returned_none += 1
                    self._save_debug_text(frame_num, "swap_returned=None")
                else:
                    self.stats.frames_worker_returned += 1
                    self._save_debug_image(frame_num, "03_swap", out)

                    stage_img = out

                    if self.enhancer is not None:
                        self.stats.frames_enhancer_called += 1
                        try:
                            enhanced = self.enhancer.enhance(stage_img, active_face_meta)
                        except Exception as e:
                            self.stats.frames_enhancer_failed += 1
                            self._save_debug_text(frame_num, f"enhancer_exception={e!r}")
                            enhanced = None
                        if enhanced is not None:
                            self.stats.frames_enhancer_returned += 1
                            emad = float(np.mean(np.abs(enhanced.astype(np.int16) - stage_img.astype(np.int16))))
                            self.stats.enhancer_mean_abs_diff_accum += emad
                            if emad >= self.material_change_mad_threshold:
                                self.stats.frames_enhancer_materially_changed += 1
                            self._save_debug_text(frame_num, f"enhancer_mean_abs_diff={emad:.4f}")
                            self._save_debug_image(frame_num, "03b_enhanced", enhanced)
                            self._save_debug_image(frame_num, "04b_enhancer_diff", self._diff_image(stage_img, enhanced))
                            stage_img = enhanced

                    if self.occluder is not None:
                        self.stats.frames_occluder_called += 1
                        try:
                            occluded = self.occluder.preserve(original_roi, stage_img, active_face_meta)
                        except Exception as e:
                            self.stats.frames_occluder_failed += 1
                            self._save_debug_text(frame_num, f"occluder_exception={e!r}")
                            occluded = None
                        if occluded is not None:
                            self.stats.frames_occluder_returned += 1
                            omad = float(np.mean(np.abs(occluded.astype(np.int16) - stage_img.astype(np.int16))))
                            self.stats.occluder_mean_abs_diff_accum += omad
                            if omad >= self.material_change_mad_threshold:
                                self.stats.frames_occluder_materially_changed += 1
                            self._save_debug_text(frame_num, f"occluder_mean_abs_diff={omad:.4f}")
                            self._save_debug_image(frame_num, "03c_occluded", occluded)
                            self._save_debug_image(frame_num, "04c_occluder_diff", self._diff_image(stage_img, occluded))
                            stage_img = occluded

                    swapped_np = stage_img
                    mad = float(np.mean(np.abs(swapped_np.astype(np.int16) - before.astype(np.int16))))
                    self.stats.mean_abs_diff_accum += mad
                    material = mad >= self.material_change_mad_threshold
                    if material:
                        self.stats.frames_materially_changed += 1
                    self._save_debug_text(frame_num, f"mean_abs_diff={mad:.4f} material={material}")
                    self._save_debug_image(frame_num, "04_diff", self._diff_image(before, swapped_np))
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


__all__ = ["BaseFaceSwapClipRestorer", "FaceSwapRestoreStats"]