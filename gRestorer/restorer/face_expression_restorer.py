from __future__ import annotations

"""FaceFusion-style LivePortrait expression restoration for face-swap ROIs.

This module intentionally mirrors the FaceFusion Expression Restorer flow first:
  - warp original target ROI and swapped/temp ROI using the target 5-point landmarks
  - extract feature volume from swapped/temp crop
  - extract target expression from original target crop
  - extract pose/motion/expression from swapped/temp crop
  - restrict expression areas and blend factor 0..100 -> 0..1.2
  - generate a corrected crop and paste it back to the swapped/temp ROI

It is ROI-local and optional. Swappers still own their native swap/paste semantics.
"""

from typing import List, Optional, Sequence, Tuple
import threading

import cv2
import numpy as np
import torch

from gRestorer.detector.core import FaceMetadata


# Direct constants from FaceFusion LivePortrait expression limiting.
EXPRESSION_MIN = np.array(
    [
        [
            [-2.88067125e-02, -8.12731311e-02, -1.70541159e-03],
            [-4.88598682e-02, -3.32196616e-02, -1.67431499e-04],
            [-6.75425082e-02, -4.28681746e-02, -1.98950816e-04],
            [-7.23103955e-02, -3.28503326e-02, -7.31324719e-04],
            [-3.87073644e-02, -6.01546466e-02, -5.50269964e-04],
            [-6.38048723e-02, -2.23840728e-01, -7.13261834e-04],
            [-3.02710701e-02, -3.93195450e-02, -8.24086510e-06],
            [-2.95799859e-02, -5.39318882e-02, -1.74219604e-04],
            [-2.92359516e-02, -1.53050944e-02, -6.30460854e-05],
            [-5.56493877e-03, -2.34344602e-02, -1.26858242e-04],
            [-4.37593013e-02, -2.77768299e-02, -2.70503685e-02],
            [-1.76926646e-02, -1.91676542e-02, -1.15090821e-04],
            [-8.34268332e-03, -3.99775570e-03, -3.27481248e-05],
            [-3.40162888e-02, -2.81868968e-02, -1.96679524e-04],
            [-2.91855410e-02, -3.97511162e-02, -2.81230678e-05],
            [-1.50395725e-02, -2.49494594e-02, -9.42573533e-05],
            [-1.67938769e-02, -2.00953931e-02, -4.00750607e-04],
            [-1.86435618e-02, -2.48535164e-02, -2.74416432e-02],
            [-4.61211195e-03, -1.21660791e-02, -2.93173041e-04],
            [-4.10017073e-02, -7.43824020e-02, -4.42762971e-02],
            [-1.90370996e-02, -3.74363363e-02, -1.34740388e-02],
        ]
    ]
).astype(np.float32)

EXPRESSION_MAX = np.array(
    [
        [
            [4.46682945e-02, 7.08772913e-02, 4.08344204e-04],
            [2.14308221e-02, 6.15894832e-02, 4.85319615e-05],
            [3.02363783e-02, 4.45043296e-02, 1.28298725e-05],
            [3.05869691e-02, 3.79812494e-02, 6.57040102e-04],
            [4.45670523e-02, 3.97259220e-02, 7.10966764e-04],
            [9.43699256e-02, 9.85926315e-02, 2.02551950e-04],
            [1.61131397e-02, 2.92906128e-02, 3.44733417e-06],
            [5.23825921e-02, 1.07065082e-01, 6.61510974e-04],
            [2.85718683e-03, 8.32320191e-03, 2.39314613e-04],
            [2.57947259e-02, 1.60935968e-02, 2.41853559e-05],
            [4.90833223e-02, 3.43903080e-02, 3.22353356e-02],
            [1.44766076e-02, 3.39248963e-02, 1.42291479e-04],
            [8.75749043e-04, 6.82212645e-03, 2.76097053e-05],
            [1.86958015e-02, 3.84016186e-02, 7.33085908e-05],
            [2.01714113e-02, 4.90544215e-02, 2.34028921e-05],
            [2.46518422e-02, 3.29151377e-02, 3.48571630e-05],
            [2.22457591e-02, 1.21796541e-02, 1.56396593e-04],
            [1.72109623e-02, 3.01626958e-02, 1.36556877e-02],
            [1.83460284e-02, 1.61141958e-02, 2.87440169e-04],
            [3.57594155e-02, 1.80554688e-01, 2.75554154e-02],
            [2.17450950e-02, 8.66811201e-02, 3.34241726e-02],
        ]
    ]
).astype(np.float32)


class FaceExpressionRestorer:
    """LivePortrait expression restoration stage for already swapped ROIs."""

    WARP_TEMPLATE_ARCFACE_128 = np.array(
        [
            [0.36167656, 0.40387734],
            [0.63696719, 0.40235469],
            [0.50019687, 0.56044219],
            [0.38710391, 0.72160547],
            [0.61507734, 0.72034453],
        ],
        dtype=np.float32,
    )

    UPPER_FACE_INDICES = [1, 2, 6, 10, 11, 12, 13, 15, 16]
    LOWER_FACE_INDICES = [3, 7, 14, 17, 18, 19, 20]
    ALWAYS_KEEP_TEMP_INDICES = [0, 4, 5, 8, 9]

    def __init__(
        self,
        device: torch.device,
        *,
        feature_extractor_path: str,
        motion_extractor_path: str,
        generator_path: str,
        provider: str = "auto",
        model: str = "live_portrait",
        factor: int = 80,
        areas: Optional[Sequence[str]] = None,
        mask_blur: float = 0.3,
    ) -> None:
        self.device = device
        self.model = str(model or "live_portrait").lower()
        if self.model != "live_portrait":
            raise ValueError(f"Unsupported expression restorer model: {model!r}")

        self.feature_extractor_path = str(feature_extractor_path)
        self.motion_extractor_path = str(motion_extractor_path)
        self.generator_path = str(generator_path)
        self.provider = str(provider or "auto").lower()
        self.factor = int(max(0, min(100, int(factor))))
        self.expression_factor = float(np.interp(float(self.factor), [0.0, 100.0], [0.0, 1.2]))
        self.areas = self._normalize_areas(areas)
        self.mask_blur = float(max(0.0, mask_blur))
        self.model_size = (512, 512)
        self._lock = threading.Lock()

        try:
            import onnxruntime as ort
        except Exception as e:
            raise ImportError("FaceExpressionRestorer requires onnxruntime / onnxruntime-gpu.") from e

        providers = self._providers_for(self.provider, self.device)
        self._feature_session = ort.InferenceSession(self.feature_extractor_path, providers=providers)
        self._motion_session = ort.InferenceSession(self.motion_extractor_path, providers=providers)
        self._generator_session = ort.InferenceSession(self.generator_path, providers=providers)

        self._feature_input_name = self._feature_session.get_inputs()[0].name
        self._motion_input_name = self._motion_session.get_inputs()[0].name
        self._generator_inputs = [inp.name for inp in self._generator_session.get_inputs()]

        print(
            f"[ExpressionRestorer] enabled model={self.model} provider={self.provider} "
            f"factor={self.factor} mapped_factor={self.expression_factor:.3f} "
            f"areas={','.join(self.areas)} mask_blur={self.mask_blur}"
        )

    @staticmethod
    def _providers_for(provider: str, device: torch.device) -> List[str]:
        p = str(provider or "auto").lower()
        if p == "cpu":
            return ["CPUExecutionProvider"]
        if p == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.type == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def _normalize_areas(areas: Optional[Sequence[str]]) -> List[str]:
        if areas is None:
            return ["upper-face", "lower-face"]
        if isinstance(areas, str):
            raw = [p.strip() for p in areas.replace(",", " ").split() if p.strip()]
        else:
            raw = [str(p).strip() for p in areas if str(p).strip()]
        out: List[str] = []
        for area in raw:
            area_l = area.lower()
            if area_l not in ("upper-face", "lower-face"):
                raise ValueError(f"Unsupported expression restorer area: {area!r}")
            if area_l not in out:
                out.append(area_l)
        return out or ["upper-face", "lower-face"]

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
        return np.stack(
            [
                np.mean(pts68[36:42], axis=0),
                np.mean(pts68[42:48], axis=0),
                pts68[30],
                pts68[48],
                pts68[54],
            ],
            axis=0,
        ).astype(np.float32)

    @classmethod
    def _five_points(cls, face_meta: FaceMetadata) -> np.ndarray:
        if face_meta.kps is not None:
            kps = face_meta.kps.detach().cpu().numpy().astype(np.float32, copy=True)
            if kps.ndim == 2 and kps.shape == (5, 2):
                return kps
            if kps.ndim == 2 and kps.shape[0] >= 68 and kps.shape[1] == 2:
                return cls._landmarks68_to_five_points(kps[:68])
        return cls._bbox_to_five_points(face_meta)

    @classmethod
    def _estimate_affine(cls, face_landmark_5: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        warp_template = cls.WARP_TEMPLATE_ARCFACE_128 * np.asarray(crop_size, dtype=np.float32)
        affine_matrix = cv2.estimateAffinePartial2D(
            face_landmark_5.astype(np.float32),
            warp_template.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=100,
        )[0]
        if affine_matrix is None:
            raise RuntimeError("Failed to estimate expression-restorer affine transform.")
        return affine_matrix.astype(np.float32)

    @classmethod
    def _warp_face_by_landmark_5(
        cls,
        frame_bgr_u8: np.ndarray,
        face_landmark_5: np.ndarray,
        crop_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        affine_matrix = cls._estimate_affine(face_landmark_5, crop_size)
        crop = cv2.warpAffine(
            frame_bgr_u8,
            affine_matrix,
            crop_size,
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_AREA,
        )
        return crop, affine_matrix

    @staticmethod
    def _create_box_mask(crop_bgr_u8: np.ndarray, face_mask_blur: float, padding: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> np.ndarray:
        crop_size = crop_bgr_u8.shape[:2][::-1]
        blur_amount = int(crop_size[0] * 0.5 * float(face_mask_blur))
        blur_area = max(blur_amount // 2, 1)
        box_mask = np.ones(crop_size, dtype=np.float32)
        box_mask[:max(blur_area, int(crop_size[1] * padding[0] / 100)), :] = 0.0
        box_mask[-max(blur_area, int(crop_size[1] * padding[2] / 100)):, :] = 0.0
        box_mask[:, :max(blur_area, int(crop_size[0] * padding[3] / 100))] = 0.0
        box_mask[:, -max(blur_area, int(crop_size[0] * padding[1] / 100)):] = 0.0
        if blur_amount > 0:
            box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
        return box_mask.astype(np.float32, copy=False)

    @staticmethod
    def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        pts = cv2.transform(pts, matrix).reshape(-1, 2)
        return pts

    @staticmethod
    def _calculate_paste_area(frame_bgr_u8: np.ndarray, crop_bgr_u8: np.ndarray, affine_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        frame_h, frame_w = frame_bgr_u8.shape[:2]
        crop_h, crop_w = crop_bgr_u8.shape[:2]
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        crop_points = np.array([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]], dtype=np.float32)
        paste_region_points = FaceExpressionRestorer._transform_points(crop_points, inverse_matrix)
        paste_region_min = np.floor(paste_region_points.min(axis=0)).astype(int)
        paste_region_max = np.ceil(paste_region_points.max(axis=0)).astype(int)
        x1, y1 = np.clip(paste_region_min, 0, [frame_w, frame_h])
        x2, y2 = np.clip(paste_region_max, 0, [frame_w, frame_h])
        paste_matrix = inverse_matrix.copy()
        paste_matrix[0, 2] -= x1
        paste_matrix[1, 2] -= y1
        return np.array([x1, y1, x2, y2], dtype=np.int32), paste_matrix.astype(np.float32)

    @staticmethod
    def _paste_back(frame_bgr_u8: np.ndarray, crop_bgr_u8: np.ndarray, crop_mask: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        paste_box, paste_matrix = FaceExpressionRestorer._calculate_paste_area(frame_bgr_u8, crop_bgr_u8, affine_matrix)
        x1, y1, x2, y2 = [int(v) for v in paste_box]
        paste_w = x2 - x1
        paste_h = y2 - y1
        if paste_w <= 0 or paste_h <= 0:
            return frame_bgr_u8
        inverse_mask = cv2.warpAffine(crop_mask, paste_matrix, (paste_w, paste_h)).clip(0.0, 1.0)
        inverse_mask = inverse_mask[..., None]
        inverse_frame = cv2.warpAffine(
            crop_bgr_u8,
            paste_matrix,
            (paste_w, paste_h),
            borderMode=cv2.BORDER_REPLICATE,
        )
        out = frame_bgr_u8.copy()
        paste_region = out[y1:y2, x1:x2].astype(np.float32)
        blended = paste_region * (1.0 - inverse_mask) + inverse_frame.astype(np.float32) * inverse_mask
        out[y1:y2, x1:x2] = blended.clip(0.0, 255.0).astype(out.dtype)
        return out

    @staticmethod
    def _prepare_crop(crop_bgr_u8: np.ndarray) -> np.ndarray:
        # FaceFusion: resize 512 aligned crop to 256 before LivePortrait ONNX input.
        prep_size = (256, 256)
        crop = cv2.resize(crop_bgr_u8, prep_size, interpolation=cv2.INTER_AREA)
        crop = crop[:, :, ::-1] / 255.0
        crop = np.expand_dims(crop.transpose(2, 0, 1), axis=0).astype(np.float32)
        return crop

    @staticmethod
    def _normalize_crop(crop: np.ndarray) -> np.ndarray:
        arr = np.asarray(crop)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.ndim != 3:
            raise RuntimeError(f"Unsupported LivePortrait generator output shape: {tuple(np.asarray(crop).shape)}")
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        arr = arr.astype(np.float32).clip(0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)[:, :, ::-1]
        return np.ascontiguousarray(arr)

    @staticmethod
    def _limit_expression(expression: np.ndarray) -> np.ndarray:
        return np.clip(expression, EXPRESSION_MIN, EXPRESSION_MAX)

    @staticmethod
    def _create_rotation_numpy(pitch: float, yaw: float, roll: float) -> np.ndarray:
        # Match scipy.spatial.transform.Rotation.from_euler('xyz', ..., degrees=True).as_matrix().
        try:
            from scipy.spatial.transform import Rotation
            return Rotation.from_euler("xyz", [float(pitch), float(yaw), float(roll)], degrees=True).as_matrix().astype(np.float32)
        except Exception:
            px, py, pz = np.deg2rad([float(pitch), float(yaw), float(roll)])
            cx, sx = np.cos(px), np.sin(px)
            cy, sy = np.cos(py), np.sin(py)
            cz, sz = np.cos(pz), np.sin(pz)
            rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
            ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
            rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
            return (rz @ ry @ rx).astype(np.float32)

    def _forward_extract_feature(self, crop: np.ndarray) -> np.ndarray:
        return self._feature_session.run(None, {self._feature_input_name: crop})[0]

    def _forward_extract_motion(self, crop: np.ndarray):
        outs = self._motion_session.run(None, {self._motion_input_name: crop})
        if len(outs) < 7:
            raise RuntimeError(f"LivePortrait motion extractor returned {len(outs)} outputs; expected 7.")
        return outs[:7]

    def _forward_generate_frame(self, feature_volume: np.ndarray, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        input_names = self._generator_inputs
        if {"feature_volume", "source", "target"}.issubset(set(input_names)):
            feed = {
                "feature_volume": feature_volume,
                "source": source_points,
                "target": target_points,
            }
        else:
            if len(input_names) < 3:
                raise RuntimeError("LivePortrait generator must expose at least three inputs.")
            feed = {
                input_names[0]: feature_volume,
                input_names[1]: source_points,
                input_names[2]: target_points,
            }
        return self._generator_session.run(None, feed)[0]

    def _restrict_expression_areas(self, temp_expression: np.ndarray, target_expression: np.ndarray) -> np.ndarray:
        target = np.asarray(target_expression, dtype=np.float32).copy()
        temp = np.asarray(temp_expression, dtype=np.float32)
        if "upper-face" not in self.areas:
            target[:, self.UPPER_FACE_INDICES] = temp[:, self.UPPER_FACE_INDICES]
        if "lower-face" not in self.areas:
            target[:, self.LOWER_FACE_INDICES] = temp[:, self.LOWER_FACE_INDICES]
        target[:, self.ALWAYS_KEEP_TEMP_INDICES] = temp[:, self.ALWAYS_KEEP_TEMP_INDICES]
        return target

    def _apply_restore(self, target_crop: np.ndarray, temp_crop: np.ndarray) -> np.ndarray:
        feature_volume = self._forward_extract_feature(temp_crop)
        target_expression = self._forward_extract_motion(target_crop)[5]
        pitch, yaw, roll, scale, translation, temp_expression, motion_points = self._forward_extract_motion(temp_crop)

        rotation = self._create_rotation_numpy(float(np.asarray(pitch).squeeze()), float(np.asarray(yaw).squeeze()), float(np.asarray(roll).squeeze()))
        target_expression = self._restrict_expression_areas(temp_expression, target_expression)
        target_expression = target_expression * self.expression_factor + temp_expression * (1.0 - self.expression_factor)
        target_expression = self._limit_expression(target_expression)

        target_motion_points = scale * (motion_points @ rotation.T + target_expression) + translation
        temp_motion_points = scale * (motion_points @ rotation.T + temp_expression) + translation
        generated = self._forward_generate_frame(feature_volume, target_motion_points, temp_motion_points)
        return generated

    def restore(self, original_roi_bgr_u8: np.ndarray, swapped_roi_bgr_u8: np.ndarray, face_meta: FaceMetadata) -> Optional[np.ndarray]:
        if original_roi_bgr_u8 is None or swapped_roi_bgr_u8 is None or face_meta is None:
            return None
        if self.factor <= 0:
            return swapped_roi_bgr_u8

        face_landmark_5 = self._five_points(face_meta)
        target_crop, _ = self._warp_face_by_landmark_5(original_roi_bgr_u8, face_landmark_5, self.model_size)
        temp_crop, affine_matrix = self._warp_face_by_landmark_5(swapped_roi_bgr_u8, face_landmark_5, self.model_size)
        crop_mask = self._create_box_mask(temp_crop, self.mask_blur, (0, 0, 0, 0)).clip(0.0, 1.0)

        target_input = self._prepare_crop(target_crop)
        temp_input = self._prepare_crop(temp_crop)

        with self._lock:
            generated = self._apply_restore(target_input, temp_input)

        generated_crop = self._normalize_crop(generated)
        out = self._paste_back(swapped_roi_bgr_u8, generated_crop, crop_mask, affine_matrix)
        return np.ascontiguousarray(out)


__all__ = ["FaceExpressionRestorer"]
