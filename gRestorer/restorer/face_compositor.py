from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

from gRestorer.detector.core import FaceMetadata
from gRestorer.restorer.face_types import FaceSwapBackendResult, FaceCompositorConfig


class FaceCompositor:
    def __init__(self, cfg: FaceCompositorConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _arcface_template(size: int) -> np.ndarray:
        tmpl = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        return tmpl * (float(size) / 112.0)

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
    def _five_points(face_meta: FaceMetadata) -> np.ndarray:
        if face_meta.kps is not None:
            kps = face_meta.kps
            if isinstance(kps, torch.Tensor):
                kps = kps.detach().cpu().numpy()
            kps = np.asarray(kps, dtype=np.float32)
            if kps.ndim == 2 and kps.shape == (5, 2):
                return kps.copy()
        return FaceCompositor._bbox_to_five_points(face_meta)

    @staticmethod
    def _ensure_hwc_u8(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HWC BGR image, got shape={tuple(arr.shape)}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).round().astype(np.uint8)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _ensure_mask_f32(mask: Optional[np.ndarray], *, shape_hw: tuple[int, int] | None = None) -> Optional[np.ndarray]:
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                arr = arr[..., 0]
            elif arr.shape[2] == 3:
                arr = np.max(arr, axis=2)
            else:
                arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"Expected HW mask, got shape={tuple(arr.shape)}")
        arr = arr.astype(np.float32, copy=False)
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        if shape_hw is not None and (arr.shape[0] != shape_hw[0] or arr.shape[1] != shape_hw[1]):
            arr = cv2.resize(arr, (int(shape_hw[1]), int(shape_hw[0])), interpolation=cv2.INTER_LINEAR)
            arr = np.clip(arr, 0.0, 1.0)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _transform_points(pts_xy: np.ndarray, M_2x3: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.transform(pts, M_2x3).reshape(-1, 2)

    def _build_geom_mask_aligned(
            self,
            face_meta: FaceMetadata,
            roi_to_aligned: np.ndarray,
            aligned_size: int,
    ) -> np.ndarray:
        size = int(aligned_size)
        mask = np.zeros((size, size), dtype=np.float32)

        # Stable aligned-space feature-priority mask.
        # Landmarks are still used upstream for alignment; here we only decide
        # which aligned regions of the swapped face are trusted during paste-back.
        expand = float(self.cfg.geom_expand) if float(self.cfg.geom_expand) > 0.0 else 1.0

        def _ellipse(cx_f: float, cy_f: float, rx_f: float, ry_f: float) -> None:
            cx = int(round(cx_f * size))
            cy = int(round(cy_f * size))
            rx = max(1, int(round(rx_f * size * expand)))
            ry = max(1, int(round(ry_f * size * expand)))
            cv2.ellipse(
                mask,
                center=(cx, cy),
                axes=(rx, ry),
                angle=0.0,
                startAngle=0.0,
                endAngle=360.0,
                color=1.0,
                thickness=-1,
            )

        # Eyes: prioritize identity-critical regions, but keep them compact.
        _ellipse(0.37, 0.40, 0.115, 0.090)
        _ellipse(0.63, 0.40, 0.115, 0.090)

        # Nose / central face bridge.
        _ellipse(0.50, 0.53, 0.110, 0.145)

        # Mouth / philtrum / lower face.
        _ellipse(0.50, 0.67, 0.180, 0.110)

        # Chin support so lower face does not cut off too abruptly.
        _ellipse(0.50, 0.79, 0.135, 0.080)

        # Soft bridges so the mask does not look like disconnected blobs.
        # Vertical bridge through center face.
        bridge_w = max(1, int(round(0.18 * size * expand)))
        bridge_h = max(1, int(round(0.36 * size * expand)))
        bx1 = int(round(0.50 * size - 0.5 * bridge_w))
        by1 = int(round(0.46 * size - 0.5 * bridge_h))
        bx2 = bx1 + bridge_w
        by2 = by1 + bridge_h
        cv2.rectangle(mask, (bx1, by1), (bx2, by2), 1.0, thickness=-1)

        # Subtle eye-to-nose bridges.
        cheek_w = max(1, int(round(0.10 * size * expand)))
        cheek_h = max(1, int(round(0.14 * size * expand)))

        # Left bridge
        lx1 = int(round(0.42 * size - 0.5 * cheek_w))
        ly1 = int(round(0.47 * size - 0.5 * cheek_h))
        lx2 = lx1 + cheek_w
        ly2 = ly1 + cheek_h
        cv2.rectangle(mask, (lx1, ly1), (lx2, ly2), 1.0, thickness=-1)

        # Right bridge
        rx1 = int(round(0.58 * size - 0.5 * cheek_w))
        ry1 = int(round(0.47 * size - 0.5 * cheek_h))
        rx2 = rx1 + cheek_w
        ry2 = ry1 + cheek_h
        cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 1.0, thickness=-1)

        # Gentle soften before later postprocess blur.
        k = max(7, int(size * 0.06))
        if k % 2 == 0:
            k += 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

        return np.clip(mask, 0.0, 1.0)

    @staticmethod
    def _ensure_hwc_f32(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HWC image, got shape={tuple(arr.shape)}")
        arr = arr.astype(np.float32, copy=False)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.ascontiguousarray(np.clip(arr, 0.0, 1.0))

    @staticmethod
    def _f32_to_u8(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return np.ascontiguousarray(np.clip(arr, 0.0, 255.0).round().astype(np.uint8))



    def _combine_masks(
        self,
        geom_mask_f32: Optional[np.ndarray],
        pred_src_mask_f32: Optional[np.ndarray],
        pred_dst_mask_f32: Optional[np.ndarray],
    ) -> np.ndarray:
        g = self._ensure_mask_f32(geom_mask_f32)

        shape_hw = None if g is None else (g.shape[0], g.shape[1])
        s = self._ensure_mask_f32(pred_src_mask_f32, shape_hw=shape_hw)
        d = self._ensure_mask_f32(pred_dst_mask_f32, shape_hw=shape_hw if shape_hw is not None else (s.shape[0], s.shape[1]) if s is not None else None)

        mode = str(self.cfg.mask_mode or "geom").strip().lower()

        def _pred_both() -> Optional[np.ndarray]:
            if s is None and d is None:
                return None
            if s is None:
                return d
            if d is None:
                return s
            return np.minimum(s, d)

        p = _pred_both()

        if mode in ("geom", "geometric"):
            return g if g is not None else (p if p is not None else s if s is not None else d)

        if mode in ("pred_src", "src", "predicted_src"):
            return s if s is not None else (g if g is not None else d)

        if mode in ("pred_dst", "dst", "predicted_dst"):
            return d if d is not None else (g if g is not None else s)

        if mode in ("pred_both", "pred_src_dst", "predicted_both"):
            return p if p is not None else (g if g is not None else s if s is not None else d)

        if mode in ("geom_pred_intersection", "intersection"):
            if g is None:
                return p if p is not None else s if s is not None else d
            q = p if p is not None else s if s is not None else d
            if q is None:
                return g
            return np.minimum(g, q)

        if mode in ("geom_pred_union", "union"):
            if g is None:
                return p if p is not None else s if s is not None else d
            q = p if p is not None else s if s is not None else d
            if q is None:
                return g
            return np.maximum(g, q)

        return g if g is not None else (p if p is not None else s if s is not None else d)

    def _postprocess_mask(self, mask_f32: np.ndarray) -> np.ndarray:
        m = np.clip(mask_f32.astype(np.float32, copy=False), 0.0, 1.0)

        if self.cfg.mask_erode > 0:
            k = int(self.cfg.mask_erode)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
            m = cv2.erode(m, kernel, iterations=1)

        if self.cfg.mask_dilate > 0:
            k = int(self.cfg.mask_dilate)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
            m = cv2.dilate(m, kernel, iterations=1)

        if self.cfg.mask_blur > 0:
            k = int(self.cfg.mask_blur)
            if k % 2 == 0:
                k += 1
            m = cv2.GaussianBlur(m, (k, k), 0)

        return np.clip(m, 0.0, 1.0)

    @staticmethod
    def _warp_aligned_face_to_roi(
        aligned_face_bgr_u8: np.ndarray,
        aligned_to_roi: np.ndarray,
        roi_w: int,
        roi_h: int,
    ) -> np.ndarray:
        return cv2.warpAffine(
            aligned_face_bgr_u8,
            aligned_to_roi,
            (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    @staticmethod
    def _warp_aligned_mask_to_roi(
        aligned_mask_f32: np.ndarray,
        aligned_to_roi: np.ndarray,
        roi_w: int,
        roi_h: int,
    ) -> np.ndarray:
        return cv2.warpAffine(
            aligned_mask_f32.astype(np.float32, copy=False),
            aligned_to_roi,
            (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    def _maybe_color_transfer(
        self,
        aligned_target_bgr_u8: Optional[np.ndarray],
        aligned_swapped_bgr_u8: np.ndarray,
        aligned_mask_f32: np.ndarray,
    ) -> np.ndarray:
        mode = str(self.cfg.color_transfer or "none").strip().lower()
        if mode in ("", "none", "off"):
            return aligned_swapped_bgr_u8
        return aligned_swapped_bgr_u8

    @staticmethod
    def _blend_alpha(
        base_bgr_u8: np.ndarray,
        over_bgr_u8: np.ndarray,
        alpha_f32: np.ndarray,
    ) -> np.ndarray:
        alpha = np.clip(alpha_f32.astype(np.float32, copy=False), 0.0, 1.0)[..., None]
        base = base_bgr_u8.astype(np.float32)
        over = over_bgr_u8.astype(np.float32)
        out = base * (1.0 - alpha) + over * alpha
        return np.clip(out, 0.0, 255.0).round().astype(np.uint8)

    def compose_debug(
        self,
        *,
        original_roi_bgr_u8: np.ndarray,
        target_face_meta: FaceMetadata,
        backend_result: FaceSwapBackendResult,
        occlusion_keep_mask_f32: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        roi_u8 = self._ensure_hwc_u8(original_roi_bgr_u8)
        roi_h, roi_w = int(roi_u8.shape[0]), int(roi_u8.shape[1])

        swapped_f32 = self._ensure_hwc_f32(backend_result.swapped_face_f32)
        aligned_size = int(backend_result.aligned_size or swapped_f32.shape[0])

        if swapped_f32.shape[0] != aligned_size or swapped_f32.shape[1] != aligned_size:
            swapped_f32 = cv2.resize(swapped_f32, (aligned_size, aligned_size), interpolation=cv2.INTER_LINEAR)

        geom_mask = self._build_geom_mask_aligned(
            target_face_meta,
            np.asarray(backend_result.roi_to_aligned, dtype=np.float32),
            aligned_size,
        )

        pred_src = self._ensure_mask_f32(backend_result.pred_src_mask_f32, shape_hw=(aligned_size, aligned_size))
        pred_dst = self._ensure_mask_f32(backend_result.pred_dst_mask_f32, shape_hw=(aligned_size, aligned_size))

        combined_mask = self._combine_masks(geom_mask, pred_src, pred_dst)
        combined_mask = self._postprocess_mask(combined_mask)

        aligned_target_u8 = None
        if backend_result.aligned_target_f32 is not None:
            aligned_target_u8 = self._f32_to_u8(self._ensure_hwc_f32(backend_result.aligned_target_f32))

        swapped_u8 = self._f32_to_u8(swapped_f32)
        swapped_u8 = self._maybe_color_transfer(aligned_target_u8, swapped_u8, combined_mask)

        warped_face = self._warp_aligned_face_to_roi(
            swapped_u8,
            np.asarray(backend_result.aligned_to_roi, dtype=np.float32),
            roi_w,
            roi_h,
        )
        warped_alpha = self._warp_aligned_mask_to_roi(
            combined_mask,
            np.asarray(backend_result.aligned_to_roi, dtype=np.float32),
            roi_w,
            roi_h,
        )

        keep_mask = None
        if occlusion_keep_mask_f32 is not None:
            keep_mask = self._ensure_mask_f32(occlusion_keep_mask_f32, shape_hw=(roi_h, roi_w))
            warped_alpha = warped_alpha * (1.0 - keep_mask)

        warped_alpha = np.clip(warped_alpha, 0.0, 1.0)
        out = self._blend_alpha(roi_u8, warped_face, warped_alpha)

        debug = {
            "aligned_geom_mask_f32": geom_mask,
            "aligned_pred_src_mask_f32": pred_src if pred_src is not None else np.zeros_like(combined_mask),
            "aligned_pred_dst_mask_f32": pred_dst if pred_dst is not None else np.zeros_like(combined_mask),
            "aligned_combined_mask_f32": combined_mask,
            "roi_warped_alpha_f32": warped_alpha,
            "roi_warped_face_bgr_u8": warped_face,
        }
        if keep_mask is not None:
            debug["roi_keep_mask_f32"] = keep_mask

        return np.ascontiguousarray(out), debug


    def compose(
        self,
        *,
        original_roi_bgr_u8: np.ndarray,
        target_face_meta: FaceMetadata,
        backend_result: FaceSwapBackendResult,
        occlusion_keep_mask_f32: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        out, _ = self.compose_debug(
            original_roi_bgr_u8=original_roi_bgr_u8,
            target_face_meta=target_face_meta,
            backend_result=backend_result,
            occlusion_keep_mask_f32=occlusion_keep_mask_f32,
        )
        return out


__all__ = ["FaceCompositor"]