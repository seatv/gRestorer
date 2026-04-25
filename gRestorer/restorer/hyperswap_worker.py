from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from gRestorer.detector.core import FaceMetadata


class HyperSwapWorker:
    """
    Native HyperSwap worker with optional FaceFusion-style pixel boost.

    Design goal:
    - follow the FaceFusion-style native flow for HyperSwap
    - return a fully pasted ROI from swap()
    - do native warp -> infer -> mask -> paste-back in swap()
    - keep backend-specific paste-back inside HyperSwap
    - optionally apply FaceFusion-style pixel boost using interleaved
      implode/explode helpers around the native model size
    """

    ARC_TEMPLATE_128 = np.array(
        [
            [0.36167656, 0.40387734],
            [0.63696719, 0.40235469],
            [0.50019687, 0.56044219],
            [0.38710391, 0.72160547],
            [0.61507734, 0.72034453],
        ],
        dtype=np.float32,
    )

    BOX_MASK_BLUR = 0.30
    BOX_MASK_PADDING = (0, 0, 0, 0)
    USE_AREA_MASK_IF_68 = True
    AREA_MASK_SIGMA = 5.0

    def __init__(
        self,
        device: torch.device,
        source_face_path: str,
        swap_model_path: str,
        *,
        swap_input_size: int = 256,
        provider: str = "auto",
        pixel_boost: str | int | tuple[int, int] | None = None,
    ) -> None:
        self.device = device
        self.source_face_path = str(source_face_path)
        self.swap_model_path = str(swap_model_path)
        self.provider = str(provider or "auto").lower()
        self.requested_swap_input_size = int(swap_input_size)

        try:
            from insightface.app import FaceAnalysis
        except Exception as e:
            raise ImportError(
                "HyperSwapWorker requires `insightface` in the gRestorer environment."
            ) from e

        try:
            import onnxruntime as ort
        except Exception as e:
            raise ImportError(
                "HyperSwapWorker requires `onnxruntime` / `onnxruntime-gpu` in the gRestorer environment."
            ) from e

        providers = self._providers_for(self.provider, self.device)
        self._providers = providers

        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        ctx_id = 0 if self.device.type == "cuda" else -1
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        src = cv2.imread(self.source_face_path, cv2.IMREAD_COLOR)
        if src is None:
            raise FileNotFoundError(f"Failed to read source face image: {self.source_face_path}")

        src_faces = self._app.get(src)
        if not src_faces:
            raise RuntimeError(f"No face detected in source image: {self.source_face_path}")

        self._src_face = max(
            src_faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        )

        self._session = ort.InferenceSession(self.swap_model_path, providers=providers)

        self._source_input_name: str = ""
        self._target_input_name: str = ""
        self._image_output_name: Optional[str] = None
        self._mask_output_name: Optional[str] = None
        self._target_input_layout: str = "nchw"
        self._target_size: int = self.requested_swap_input_size

        self._init_model_contract()
        self._src_embedding = self._prepare_source_embedding()

        self._pixel_boost_size = self._resolve_pixel_boost_size(pixel_boost)
        self._pixel_boost_total = max(1, self._pixel_boost_size // self._target_size)

        print(
            f"[HyperSwapWorker] provider={self.provider} "
            f"size={self._target_size} layout={self._target_input_layout} "
            f"source_input={self._source_input_name} target_input={self._target_input_name} "
            f"image_output={self._image_output_name or '-'} pixel_boost={self._pixel_boost_size} "
            f"scale={self._pixel_boost_total}"
        )

    @staticmethod
    def _providers_for(provider: str, device: torch.device) -> List[str]:
        p = str(provider or "auto").lower()
        if p == "cpu":
            return ["CPUExecutionProvider"]
        if p == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if p == "xpu":
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        if device.type == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _init_model_contract(self) -> None:
        inputs = list(self._session.get_inputs())
        outputs = list(self._session.get_outputs())

        if not inputs or not outputs:
            raise RuntimeError("HyperSwap ONNX session has no inputs or outputs.")

        source_input = None
        target_input = None

        for inp in inputs:
            nm = str(inp.name).lower()
            shape = tuple(inp.shape)
            if source_input is None and ("source" in nm or "embed" in nm or "id" in nm):
                source_input = inp
            if target_input is None and ("target" in nm or "image" in nm or "input" in nm):
                if len(shape) == 4:
                    target_input = inp

        if source_input is None:
            for inp in inputs:
                if len(tuple(inp.shape)) == 2:
                    source_input = inp
                    break

        if target_input is None:
            for inp in inputs:
                if len(tuple(inp.shape)) == 4:
                    target_input = inp
                    break

        if source_input is None or target_input is None:
            raise RuntimeError(
                f"Could not identify HyperSwap inputs: {[(i.name, tuple(i.shape)) for i in inputs]}"
            )

        self._source_input_name = source_input.name
        self._target_input_name = target_input.name

        tshape = tuple(target_input.shape)
        if len(tshape) != 4:
            raise RuntimeError(f"Unexpected HyperSwap target input shape: {tshape}")

        if isinstance(tshape[1], int) and int(tshape[1]) == 3:
            self._target_input_layout = "nchw"
            if isinstance(tshape[-1], int) and int(tshape[-1]) > 0:
                self._target_size = int(tshape[-1])
        elif isinstance(tshape[-1], int) and int(tshape[-1]) == 3:
            self._target_input_layout = "nhwc"
            if isinstance(tshape[1], int) and int(tshape[1]) > 0:
                self._target_size = int(tshape[1])
        else:
            self._target_input_layout = "nchw"

        out_img = None
        out_mask = None
        for out in outputs:
            nm = str(out.name).lower()
            shape = tuple(out.shape)
            if out_img is None and ("output" in nm or "image" in nm or "face" in nm):
                if len(shape) == 4:
                    out_img = out
            if out_mask is None and ("mask" in nm or "alpha" in nm):
                out_mask = out

        if out_img is None:
            for out in outputs:
                shape = tuple(out.shape)
                if len(shape) != 4:
                    continue
                c = None
                if isinstance(shape[1], int):
                    c = int(shape[1])
                elif isinstance(shape[-1], int):
                    c = int(shape[-1])
                if c == 3:
                    out_img = out
                    break

        if out_img is None:
            for out in outputs:
                if len(tuple(out.shape)) == 4:
                    out_img = out
                    break

        if out_img is None:
            raise RuntimeError(
                f"Could not identify HyperSwap image output: {[(o.name, tuple(o.shape)) for o in outputs]}"
            )

        self._image_output_name = out_img.name
        self._mask_output_name = out_mask.name if out_mask is not None else None

    def _prepare_source_embedding(self) -> np.ndarray:
        emb = getattr(self._src_face, "normed_embedding", None)
        if emb is None:
            emb = getattr(self._src_face, "embedding_norm", None)
        if emb is None:
            emb = getattr(self._src_face, "embedding", None)
        if emb is None:
            raise RuntimeError("Source face embedding missing from insightface result.")

        emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        if emb.shape[1] != 512:
            raise RuntimeError(f"Unexpected HyperSwap source embedding shape: {tuple(emb.shape)}")
        norm = float(np.linalg.norm(emb))
        if norm > 0.0:
            emb = emb / norm
        return np.ascontiguousarray(emb.astype(np.float32, copy=False))

    @staticmethod
    def _extract_target_kps5(face_meta: FaceMetadata) -> np.ndarray:
        if face_meta.kps is not None:
            kps = np.asarray(face_meta.kps.detach().cpu().numpy(), dtype=np.float32)
            if kps.ndim == 2 and kps.shape[0] >= 5:
                return np.ascontiguousarray(kps[:5].astype(np.float32, copy=True))

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
    def _extract_target_kps68(face_meta: FaceMetadata) -> Optional[np.ndarray]:
        if face_meta.kps is None:
            return None
        kps = np.asarray(face_meta.kps.detach().cpu().numpy(), dtype=np.float32)
        if kps.ndim == 2 and kps.shape[0] >= 68:
            return np.ascontiguousarray(kps[:68].astype(np.float32, copy=True))
        return None

    def _resolve_pixel_boost_size(self, pixel_boost: str | int | tuple[int, int] | None) -> int:
        if pixel_boost is None or pixel_boost == "":
            return int(self._target_size)

        if isinstance(pixel_boost, tuple):
            if len(pixel_boost) != 2 or int(pixel_boost[0]) != int(pixel_boost[1]):
                raise ValueError(f"Pixel boost must be square, got {pixel_boost!r}")
            size = int(pixel_boost[0])
        elif isinstance(pixel_boost, int):
            size = int(pixel_boost)
        else:
            text = str(pixel_boost).strip().lower().strip("\"'")
            if "x" in text:
                a, b = text.split("x", 1)
                if int(a) != int(b):
                    raise ValueError(f"Pixel boost must be square, got {pixel_boost!r}")
                size = int(a)
            else:
                size = int(text)

        if size < int(self._target_size):
            size = int(self._target_size)
        if size % int(self._target_size) != 0:
            raise ValueError(
                f"Pixel boost size must be a multiple of native model size {self._target_size}, got {size}"
            )
        return size

    @staticmethod
    def _warp_face_by_landmark_5(
        roi_bgr_u8: np.ndarray,
        kps5: np.ndarray,
        aligned_size: int,
        template_norm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dst = template_norm * float(aligned_size)
        M = cv2.estimateAffinePartial2D(
            kps5.astype(np.float32),
            dst.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=100.0,
        )[0]
        if M is None:
            raise RuntimeError("Failed to estimate affine transform for HyperSwap alignment.")
        crop = cv2.warpAffine(
            roi_bgr_u8,
            M,
            (aligned_size, aligned_size),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return np.ascontiguousarray(crop), np.ascontiguousarray(M.astype(np.float32))

    def _prepare_target_tensor(self, aligned_bgr_u8: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(aligned_bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        if self._target_input_layout == "nchw":
            rgb = np.transpose(rgb, (2, 0, 1))[None, ...]
        else:
            rgb = rgb[None, ...]
        return np.ascontiguousarray(rgb.astype(np.float32, copy=False))

    @staticmethod
    def _decode_output_bgr_u8(raw_out: np.ndarray, size: int) -> np.ndarray:
        arr = np.asarray(raw_out)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            raise RuntimeError(f"Unexpected HyperSwap output shape: {tuple(np.asarray(raw_out).shape)}")
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        arr = arr.astype(np.float32, copy=False)
        arr = arr * 0.5 + 0.5
        arr = np.clip(arr, 0.0, 1.0)
        if arr.shape[0] != size or arr.shape[1] != size:
            arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
            arr = np.clip(arr, 0.0, 1.0)
        rgb_u8 = (arr * 255.0).round().astype(np.uint8)
        bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(bgr_u8)

    def _run_model_image_only(self, target_tensor: np.ndarray) -> np.ndarray:
        feeds = {
            self._source_input_name: self._src_embedding,
            self._target_input_name: target_tensor,
        }
        outs = self._session.run(None, feeds)
        if not outs:
            raise RuntimeError("HyperSwap ONNX session returned no outputs.")
        return outs[0]

    def _run_single_tile(self, tile_bgr_u8: np.ndarray) -> np.ndarray:
        target_tensor = self._prepare_target_tensor(tile_bgr_u8)
        raw_img = self._run_model_image_only(target_tensor)
        return self._decode_output_bgr_u8(raw_img, self._target_size)

    def _implode_pixel_boost(self, crop_bgr_u8: np.ndarray) -> List[np.ndarray]:
        total = int(self._pixel_boost_total)
        size = int(self._target_size)
        pixel_boost = crop_bgr_u8.reshape(size, total, size, total, 3)
        pixel_boost = pixel_boost.transpose(1, 3, 0, 2, 4).reshape(total ** 2, size, size, 3)
        return [np.ascontiguousarray(tile) for tile in pixel_boost]

    def _explode_pixel_boost(self, temp_frames: List[np.ndarray]) -> np.ndarray:
        total = int(self._pixel_boost_total)
        size = int(self._target_size)
        boost = int(self._pixel_boost_size)
        crop = np.stack(temp_frames).reshape(total, total, size, size, 3)
        crop = crop.transpose(2, 0, 3, 1, 4).reshape(boost, boost, 3)
        return np.ascontiguousarray(crop)

    def _run_pixel_boost(self, boosted_crop_bgr_u8: np.ndarray) -> np.ndarray:
        if self._pixel_boost_total <= 1:
            return self._run_single_tile(boosted_crop_bgr_u8)
        tiles_in = self._implode_pixel_boost(boosted_crop_bgr_u8)
        tiles_out = [self._run_single_tile(tile) for tile in tiles_in]
        return self._explode_pixel_boost(tiles_out)

    @staticmethod
    def _create_box_mask(crop_bgr_u8: np.ndarray, face_mask_blur: float, face_mask_padding: Tuple[int, int, int, int]) -> np.ndarray:
        crop_w, crop_h = crop_bgr_u8.shape[:2][::-1]
        blur_amount = int(crop_w * 0.5 * float(face_mask_blur))
        blur_area = max(blur_amount // 2, 1)
        box_mask = np.ones((crop_h, crop_w), dtype=np.float32)
        top, right, bottom, left = [int(v) for v in face_mask_padding]
        box_mask[:max(blur_area, int(crop_h * top / 100.0)), :] = 0.0
        box_mask[-max(blur_area, int(crop_h * bottom / 100.0)) :, :] = 0.0
        box_mask[:, :max(blur_area, int(crop_w * left / 100.0))] = 0.0
        box_mask[:, -max(blur_area, int(crop_w * right / 100.0)) :] = 0.0
        if blur_amount > 0:
            box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
        return np.ascontiguousarray(np.clip(box_mask, 0.0, 1.0))

    @staticmethod
    def _create_area_mask_from_landmarks68(crop_size: int, landmarks68_aligned: np.ndarray, sigma: float = 5.0) -> np.ndarray:
        mask = np.zeros((crop_size, crop_size), dtype=np.float32)
        hull = cv2.convexHull(landmarks68_aligned.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
        mask = cv2.GaussianBlur(mask.clip(0.0, 1.0), (0, 0), sigma).clip(0.5, 1.0)
        mask = (mask - 0.5) * 2.0
        return np.ascontiguousarray(np.clip(mask, 0.0, 1.0))

    @staticmethod
    def _transform_points(pts_xy: np.ndarray, M_2x3: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.transform(pts, M_2x3).reshape(-1, 2)

    @staticmethod
    def _calculate_paste_area(roi_bgr_u8: np.ndarray, crop_bgr_u8: np.ndarray, affine_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        roi_h, roi_w = roi_bgr_u8.shape[:2]
        crop_h, crop_w = crop_bgr_u8.shape[:2]
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        crop_points = np.array([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]], dtype=np.float32)
        paste_region_points = cv2.transform(crop_points.reshape(-1, 1, 2), inverse_matrix).reshape(-1, 2)
        paste_min = np.floor(paste_region_points.min(axis=0)).astype(int)
        paste_max = np.ceil(paste_region_points.max(axis=0)).astype(int)
        x1, y1 = np.clip(paste_min, 0, [roi_w, roi_h])
        x2, y2 = np.clip(paste_max, 0, [roi_w, roi_h])
        paste_box = np.array([x1, y1, x2, y2], dtype=np.int32)
        paste_matrix = inverse_matrix.copy()
        paste_matrix[0, 2] -= x1
        paste_matrix[1, 2] -= y1
        return paste_box, paste_matrix

    @classmethod
    def _paste_back(cls, roi_bgr_u8: np.ndarray, crop_bgr_u8: np.ndarray, crop_mask_f32: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        paste_box, paste_matrix = cls._calculate_paste_area(roi_bgr_u8, crop_bgr_u8, affine_matrix)
        x1, y1, x2, y2 = [int(v) for v in paste_box]
        paste_w = max(0, x2 - x1)
        paste_h = max(0, y2 - y1)
        if paste_w <= 0 or paste_h <= 0:
            return np.ascontiguousarray(roi_bgr_u8.copy())
        inverse_mask = cv2.warpAffine(
            crop_mask_f32.astype(np.float32, copy=False),
            paste_matrix,
            (paste_w, paste_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ).clip(0.0, 1.0)
        inverse_mask = inverse_mask[..., None]
        inverse_crop = cv2.warpAffine(
            crop_bgr_u8,
            paste_matrix,
            (paste_w, paste_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        out = roi_bgr_u8.copy()
        paste_region = out[y1:y2, x1:x2].astype(np.float32)
        inverse_crop_f32 = inverse_crop.astype(np.float32)
        blended = paste_region * (1.0 - inverse_mask) + inverse_crop_f32 * inverse_mask
        out[y1:y2, x1:x2] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        return np.ascontiguousarray(out)

    def swap(self, roi_bgr_u8: np.ndarray, target_face_meta: FaceMetadata) -> Optional[np.ndarray]:
        if roi_bgr_u8 is None:
            return None
        aligned_size = int(self._pixel_boost_size)
        kps5 = self._extract_target_kps5(target_face_meta)
        crop_bgr_u8, affine_matrix = self._warp_face_by_landmark_5(
            roi_bgr_u8=roi_bgr_u8,
            kps5=kps5,
            aligned_size=aligned_size,
            template_norm=self.ARC_TEMPLATE_128,
        )
        swapped_crop_bgr_u8 = self._run_pixel_boost(crop_bgr_u8)
        crop_masks: List[np.ndarray] = []
        crop_masks.append(self._create_box_mask(swapped_crop_bgr_u8, self.BOX_MASK_BLUR, self.BOX_MASK_PADDING))
        if self.USE_AREA_MASK_IF_68:
            kps68 = self._extract_target_kps68(target_face_meta)
            if kps68 is not None:
                kps68_aligned = self._transform_points(kps68, affine_matrix)
                crop_masks.append(self._create_area_mask_from_landmarks68(aligned_size, kps68_aligned, self.AREA_MASK_SIGMA))
        crop_mask = np.minimum.reduce(crop_masks).clip(0.0, 1.0) if crop_masks else np.ones((aligned_size, aligned_size), dtype=np.float32)
        return self._paste_back(roi_bgr_u8, swapped_crop_bgr_u8, crop_mask, affine_matrix)


__all__ = ["HyperSwapWorker"]
