
from __future__ import annotations

from types import SimpleNamespace
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from gRestorer.detector.core import FaceMetadata


class SimSwapWorker:
    """
    Native SimSwap worker, cleaned up for the stable SimSwap 256 path.

    Design goals:
    - keep the FaceFusion-style native SimSwap flow
    - keep the debug dump path
    - explicitly reject unsupported SimSwap 512 unofficial models
    - keep the worker simple and reliable for SimSwap 256
    """

    ARC_DST_5_V1 = np.array(
        [
            [39.7300, 51.1380],
            [72.2700, 51.1380],
            [56.0000, 68.4930],
            [42.4630, 87.0100],
            [69.5370, 87.0100],
        ],
        dtype=np.float32,
    )

    BOX_MASK_BLUR = 0.30
    BOX_MASK_PADDING = (0, 0, 0, 0)  # top, right, bottom, left in percent
    USE_AREA_MASK_IF_68 = False
    AREA_MASK_SIGMA = 5.0

    def __init__(
        self,
        device: torch.device,
        source_face_path: str,
        swap_model_path: str,
        *,
        provider: str = "auto",
        embedding_converter_path: str | None = None,
        face_swapper_weight: float = 1.0,
    ) -> None:
        self.device = device
        self.source_face_path = str(source_face_path)
        self.swap_model_path = str(swap_model_path)
        self.provider = str(provider or "auto").lower()
        self.face_swapper_weight = float(max(0.0, min(1.0, float(face_swapper_weight))))
        self.last_swap_metrics: dict = {}

        try:
            from insightface.app import FaceAnalysis
        except Exception as e:
            raise ImportError(
                "SimSwapWorker requires `insightface` in the gRestorer environment."
            ) from e

        try:
            import onnxruntime as ort
        except Exception as e:
            raise ImportError(
                "SimSwapWorker requires onnxruntime / onnxruntime-gpu in the gRestorer environment."
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
        self._output_name: str | None = None
        self._image_input_name: str | None = None
        self._embedding_input_name: str | None = None
        self._image_input_layout = "nchw"
        self._image_size = 256
        self._model_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._model_std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self._init_simswap_contract()

        self._embedding_converter_path = self._resolve_embedding_converter_path(embedding_converter_path)
        self._embedding_converter = None
        if self._embedding_converter_path:
            self._embedding_converter = ort.InferenceSession(self._embedding_converter_path, providers=providers)

        self._src_embedding_raw = self._prepare_source_embedding_raw()
        self._src_embedding = self._convert_simswap_embedding(self._src_embedding_raw)

        self._debug_dir = str(os.environ.get("GR_SIMSWAP_DEBUG_DIR", "") or "").strip()
        self._debug_swap_index = int(str(os.environ.get("GR_SIMSWAP_DEBUG_SWAP_INDEX", "0") or "0"))
        self._debug_dump_all = str(os.environ.get("GR_SIMSWAP_DEBUG_DUMP_ALL", "0") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._swap_counter = 0

        print(
            f"[SimSwapWorker] provider={self.provider} image_size={self._image_size} "
            f"layout={self._image_input_layout} image_input={self._image_input_name} "
            f"embedding_input={self._embedding_input_name} "
            f"converter={'yes' if self._embedding_converter is not None else 'no'} "
            f"face_swapper_weight={self.face_swapper_weight:.3f} "
            f"mask_mode={'box_only' if not self.USE_AREA_MASK_IF_68 else 'box_plus_area68'} "
            f"debug_dir={self._debug_dir if self._debug_dir else '-'} "
            f"debug_swap_index={self._debug_swap_index if self._debug_swap_index > 0 else '-'} "
            f"debug_dump_all={self._debug_dump_all}"
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

    def _resolve_embedding_converter_path(self, explicit: str | None) -> str:
        if explicit:
            p = Path(explicit)
            return str(p) if p.exists() else ""

        model_path = Path(self.swap_model_path)
        candidates = [
            model_path.with_name("crossface_simswap.onnx"),
            model_path.with_name("arcface_converter_simswap.onnx"),
            model_path.parent / "crossface_simswap.onnx",
            model_path.parent / "arcface_converter_simswap.onnx",
            model_path.parent / ".assets" / "models" / "crossface_simswap.onnx",
            model_path.parent / ".assets" / "models" / "arcface_converter_simswap.onnx",
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand)
        return ""

    def _init_simswap_contract(self) -> None:
        inputs = list(self._session.get_inputs())
        outputs = list(self._session.get_outputs())
        if not inputs or not outputs:
            raise RuntimeError("SimSwap ONNX session has no inputs or outputs.")

        self._output_name = outputs[0].name

        image_input = None
        emb_input = None
        for inp in inputs:
            shape = tuple(inp.shape)
            if len(shape) == 4 and image_input is None:
                image_input = inp
            elif len(shape) == 2 and emb_input is None:
                emb_input = inp

        if image_input is None or emb_input is None:
            for inp in inputs:
                nm = str(inp.name).lower()
                if image_input is None and any(tok in nm for tok in ("target", "input", "img", "image")):
                    image_input = inp
                if emb_input is None and any(tok in nm for tok in ("source", "id", "latent", "embed")):
                    emb_input = inp

        if image_input is None or emb_input is None:
            raise RuntimeError(
                f"Could not identify SimSwap ONNX inputs: {[(i.name, tuple(i.shape)) for i in inputs]}"
            )

        self._image_input_name = image_input.name
        self._embedding_input_name = emb_input.name

        ishape = tuple(image_input.shape)
        if len(ishape) != 4:
            raise RuntimeError(f"Unexpected SimSwap image input shape: {ishape}")

        if isinstance(ishape[1], int) and int(ishape[1]) == 3:
            self._image_input_layout = "nchw"
            self._image_size = int(ishape[-1]) if isinstance(ishape[-1], int) else 256
        elif isinstance(ishape[-1], int) and int(ishape[-1]) == 3:
            self._image_input_layout = "nhwc"
            self._image_size = int(ishape[1]) if isinstance(ishape[1], int) else 256
        else:
            dims = [int(v) for v in ishape if isinstance(v, int) and v > 0]
            self._image_input_layout = "nchw"
            self._image_size = dims[-1] if dims else 256

        name = Path(self.swap_model_path).name.lower()
        if "unofficial_512" in name or "512_unofficial" in name or self._image_size >= 512:
            raise SystemExit(
                "❌ Unsupported configuration: SimSwap 512 unofficial is not supported in gRestorer. "
                "Please select the stable SimSwap 256 model instead."
            )

        self._model_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._model_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


    @staticmethod
    def _normalize_embedding(emb: np.ndarray) -> np.ndarray:
        arr = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(arr))
        if norm > 0.0:
            arr = arr / norm
        return np.ascontiguousarray(arr.astype(np.float32, copy=False))

    @staticmethod
    def _face_embedding(face) -> Optional[np.ndarray]:
        for attr in ("normed_embedding", "embedding_norm", "embedding"):
            emb = getattr(face, attr, None)
            if emb is not None:
                return SimSwapWorker._normalize_embedding(np.asarray(emb, dtype=np.float32))
        return None

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0.0 else 0.0

    def _select_target_face_candidate(self, roi_bgr_u8: np.ndarray, target_face_meta: FaceMetadata):
        try:
            candidates = self._app.get(roi_bgr_u8)
        except Exception:
            return None
        if not candidates:
            return None
        target_bbox = np.asarray(target_face_meta.bbox_xyxy, dtype=np.float32)
        return max(
            candidates,
            key=lambda f: (
                self._bbox_iou(np.asarray(getattr(f, "bbox", target_bbox), dtype=np.float32), target_bbox),
                float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if hasattr(f, "bbox") else 0.0,
            ),
        )

    def _target_face_from_meta(self, face_meta: FaceMetadata):
        return SimpleNamespace(
            bbox=np.asarray(face_meta.bbox_xyxy, dtype=np.float32),
            kps=self._extract_target_kps5(face_meta),
            det_score=float(face_meta.det_score) if face_meta.det_score is not None else 1.0,
        )

    def _recognize_target_face_from_meta(self, roi_bgr_u8: np.ndarray, target_face_meta: FaceMetadata):
        """Run ArcFace recognition using pipeline-selected face metadata instead of re-detecting the face."""
        face = self._target_face_from_meta(target_face_meta)
        models = getattr(self._app, "models", {}) or {}
        model_values = list(models.values()) if isinstance(models, dict) else list(models)
        recognizers = [
            m for m in model_values
            if str(getattr(m, "taskname", "")).lower() == "recognition"
        ]
        recognizers.extend([m for m in model_values if m not in recognizers and hasattr(m, "get")])

        for model in recognizers:
            try:
                model.get(roi_bgr_u8, face)
                if self._face_embedding(face) is not None:
                    return face
            except Exception:
                continue
        return None

    @staticmethod
    def _embedding_cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
        if a is None or b is None:
            return None
        aa = np.asarray(a, dtype=np.float32).reshape(-1)
        bb = np.asarray(b, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        if denom <= 0.0:
            return None
        return float(np.dot(aa, bb) / denom)

    def _set_last_embedding_metrics(
        self,
        *,
        source_embedding: Optional[np.ndarray],
        target_embedding: Optional[np.ndarray],
        mixed_embedding: Optional[np.ndarray],
        target_candidate_found: bool,
        mode: str,
        target_embedding_source: str = "none",
    ) -> None:
        target_used = target_embedding is not None
        self.last_swap_metrics = {
            "embedding_mode": mode,
            "face_swapper_weight": float(self.face_swapper_weight),
            "target_candidate_found": bool(target_candidate_found),
            "target_embedding_source": str(target_embedding_source),
            "target_embedding_used": bool(target_used),
            "target_embedding_missing": bool(not target_used and self.face_swapper_weight < 0.999),
            "mixed_embedding_used": bool(str(mode).startswith("mixed") and target_used and self.face_swapper_weight < 0.999),
            "source_to_target_cos": self._embedding_cosine(source_embedding, target_embedding),
            "source_to_mixed_cos": self._embedding_cosine(source_embedding, mixed_embedding),
            "target_to_mixed_cos": self._embedding_cosine(target_embedding, mixed_embedding),
        }

    @staticmethod
    def _mix_embeddings(source_embedding: np.ndarray, target_embedding: Optional[np.ndarray], weight: float) -> np.ndarray:
        if target_embedding is None:
            return source_embedding
        w = float(max(0.0, min(1.0, weight)))
        mixed = source_embedding.astype(np.float32) * w + target_embedding.astype(np.float32) * (1.0 - w)
        return SimSwapWorker._normalize_embedding(mixed)

    def _prepare_source_embedding_raw(self) -> np.ndarray:
        emb = self._face_embedding(self._src_face)
        if emb is None:
            raise RuntimeError("Source face embedding missing from insightface result.")
        return emb

    def _convert_simswap_embedding(self, emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        if self._embedding_converter is not None:
            conv_inputs = self._embedding_converter.get_inputs()
            conv_name = conv_inputs[0].name if conv_inputs else "input"
            emb = self._embedding_converter.run(None, {conv_name: emb})[0]
        return self._normalize_embedding(emb)

    def _runtime_simswap_embedding(self, roi_bgr_u8: np.ndarray, target_face_meta: FaceMetadata) -> np.ndarray:
        if self.face_swapper_weight >= 0.999:
            self._set_last_embedding_metrics(
                source_embedding=self._src_embedding_raw,
                target_embedding=None,
                mixed_embedding=self._src_embedding_raw,
                target_candidate_found=False,
                mode="source_only",
                target_embedding_source="none",
            )
            return self._src_embedding
        target_embedding_source = "detector_candidate"
        target_candidate = self._select_target_face_candidate(roi_bgr_u8, target_face_meta)
        target_embedding = self._face_embedding(target_candidate) if target_candidate is not None else None

        if target_embedding is None:
            target_embedding_source = "metadata_recognition"
            target_candidate = self._recognize_target_face_from_meta(roi_bgr_u8, target_face_meta)
            target_embedding = self._face_embedding(target_candidate) if target_candidate is not None else None

        if target_embedding is None:
            target_embedding_source = "failed"

        mixed_raw = self._mix_embeddings(self._src_embedding_raw, target_embedding, self.face_swapper_weight)
        self._set_last_embedding_metrics(
            source_embedding=self._src_embedding_raw,
            target_embedding=target_embedding,
            mixed_embedding=mixed_raw,
            target_candidate_found=target_candidate is not None,
            mode="mixed" if target_embedding is not None else "target_lookup_failed",
            target_embedding_source=target_embedding_source,
        )
        return self._convert_simswap_embedding(mixed_raw)

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

    def _align_face(self, roi_bgr_u8: np.ndarray, kps5: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dst = self.ARC_DST_5_V1.copy()
        if self._image_size != 112:
            dst *= float(self._image_size) / 112.0

        M, _ = cv2.estimateAffinePartial2D(
            kps5.astype(np.float32),
            dst.astype(np.float32),
            method=cv2.LMEDS,
        )
        if M is None:
            raise RuntimeError("Failed to estimate affine transform for SimSwap alignment.")

        aligned = cv2.warpAffine(
            roi_bgr_u8,
            M,
            (self._image_size, self._image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return np.ascontiguousarray(aligned), np.ascontiguousarray(M.astype(np.float32))

    def _prepare_simswap_image(self, aligned_bgr_u8: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(aligned_bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._model_mean) / self._model_std
        if self._image_input_layout == "nchw":
            rgb = np.transpose(rgb, (2, 0, 1))[None, ...]
        else:
            rgb = rgb[None, ...]
        return np.ascontiguousarray(rgb.astype(np.float32, copy=False))

    @staticmethod
    def _decode_simswap_output(out: np.ndarray) -> np.ndarray:
        arr = np.asarray(out)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            raise RuntimeError(f"Unexpected SimSwap output shape: {tuple(np.asarray(out).shape)}")
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        y = np.clip(arr.astype(np.float32), 0.0, 1.0)
        y = (y * 255.0).round().astype(np.uint8)
        y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(y)

    def _run_simswap_once(self, img_in: np.ndarray, emb_in: np.ndarray) -> np.ndarray:
        return self._session.run(
            [self._output_name],
            {
                self._image_input_name: img_in,
                self._embedding_input_name: emb_in,
            },
        )[0]

    @staticmethod
    def _transform_points(pts_xy: np.ndarray, M_2x3: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.transform(pts, M_2x3).reshape(-1, 2)

    @staticmethod
    def _create_box_mask(
        crop_bgr_u8: np.ndarray,
        face_mask_blur: float,
        face_mask_padding: Tuple[int, int, int, int],
    ) -> np.ndarray:
        crop_w, crop_h = crop_bgr_u8.shape[:2][::-1]
        blur_amount = int(crop_w * 0.5 * float(face_mask_blur))

        box_mask = np.ones((crop_h, crop_w), dtype=np.float32)
        top, right, bottom, left = [int(v) for v in face_mask_padding]

        if top == 0 and right == 0 and bottom == 0 and left == 0:
            edge_x = max(1, int(round(crop_w * 0.02)))
            edge_y = max(1, int(round(crop_h * 0.02)))
            top_px, right_px, bottom_px, left_px = edge_y, edge_x, edge_y, edge_x
        else:
            top_px = int(round(crop_h * top / 100.0))
            right_px = int(round(crop_w * right / 100.0))
            bottom_px = int(round(crop_h * bottom / 100.0))
            left_px = int(round(crop_w * left / 100.0))

        if top_px > 0:
            box_mask[:top_px, :] = 0.0
        if bottom_px > 0:
            box_mask[-bottom_px:, :] = 0.0
        if left_px > 0:
            box_mask[:, :left_px] = 0.0
        if right_px > 0:
            box_mask[:, -right_px:] = 0.0

        if blur_amount > 0:
            box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)

        return np.ascontiguousarray(np.clip(box_mask, 0.0, 1.0))

    @staticmethod
    def _create_area_mask_from_landmarks68(
        crop_size: int,
        landmarks68_aligned: np.ndarray,
        sigma: float = 5.0,
    ) -> np.ndarray:
        mask = np.zeros((crop_size, crop_size), dtype=np.float32)
        hull = cv2.convexHull(landmarks68_aligned.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
        mask = cv2.GaussianBlur(mask.clip(0.0, 1.0), (0, 0), sigma).clip(0.5, 1.0)
        mask = (mask - 0.5) * 2.0
        return np.ascontiguousarray(np.clip(mask, 0.0, 1.0))

    @staticmethod
    def _calculate_paste_area(
        roi_bgr_u8: np.ndarray,
        crop_bgr_u8: np.ndarray,
        affine_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        roi_h, roi_w = roi_bgr_u8.shape[:2]
        crop_h, crop_w = crop_bgr_u8.shape[:2]

        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        crop_points = np.array(
            [[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]],
            dtype=np.float32,
        )
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
    def _paste_back(
        cls,
        roi_bgr_u8: np.ndarray,
        crop_bgr_u8: np.ndarray,
        crop_mask_f32: np.ndarray,
        affine_matrix: np.ndarray,
    ) -> np.ndarray:
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

    def _get_aligned_target_kps5(self, original_target_kps5: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        return self._transform_points(original_target_kps5, affine_matrix).astype(np.float32, copy=False)

    def _maybe_correct_swapped_geometry(
        self,
        swapped_crop_bgr_u8: np.ndarray,
        crop_mask_f32: np.ndarray,
        target_kps5_aligned: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        # 256-only cleaned worker: keep a no-op here so the debug contract
        # remains consistent, but do not attempt any 512-oriented geometry fix.
        return swapped_crop_bgr_u8, crop_mask_f32, {
            "attempted": False,
            "applied": False,
            "reason": "disabled_256_only_worker",
        }

    def _should_dump_debug(self, swap_index: int) -> bool:
        if not self._debug_dir:
            return False
        if self._debug_dump_all:
            return True
        return self._debug_swap_index > 0 and swap_index == self._debug_swap_index

    @staticmethod
    def _mask_to_u8(mask_f32: np.ndarray) -> np.ndarray:
        arr = np.asarray(mask_f32, dtype=np.float32)
        return np.ascontiguousarray(np.clip(arr * 255.0, 0.0, 255.0).round().astype(np.uint8))

    def _dump_debug_artifacts(
        self,
        swap_index: int,
        aligned_target_bgr_u8: np.ndarray,
        swapped_crop_raw_bgr_u8: np.ndarray,
        swapped_crop_final_bgr_u8: np.ndarray,
        crop_mask_f32: np.ndarray,
        pasted_roi_bgr_u8: np.ndarray,
        correction_info: dict,
    ) -> None:
        if not self._should_dump_debug(swap_index):
            return

        out_dir = Path(self._debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"simswap_s{swap_index:06d}"

        cv2.imwrite(str(out_dir / f"{prefix}_01_aligned_target.png"), aligned_target_bgr_u8)
        cv2.imwrite(str(out_dir / f"{prefix}_02_swapped_aligned_raw.png"), swapped_crop_raw_bgr_u8)
        if correction_info.get("applied"):
            cv2.imwrite(str(out_dir / f"{prefix}_02b_swapped_aligned_corrected.png"), swapped_crop_final_bgr_u8)
        else:
            cv2.imwrite(str(out_dir / f"{prefix}_02b_swapped_aligned_corrected.png"), swapped_crop_final_bgr_u8)
        cv2.imwrite(str(out_dir / f"{prefix}_03_crop_mask.png"), self._mask_to_u8(crop_mask_f32))
        cv2.imwrite(str(out_dir / f"{prefix}_04_pasted_roi.png"), pasted_roi_bgr_u8)

        meta_path = out_dir / f"{prefix}_05_debug_meta.txt"
        lines = [
            "worker_mode=256_only",
            f"correction_attempted={bool(correction_info.get('attempted', False))}",
            f"correction_applied={bool(correction_info.get('applied', False))}",
            f"correction_reason={correction_info.get('reason', '-')}",
        ]
        meta_path.write_text("\n".join(str(x) for x in lines), encoding="utf-8")

    def swap(
        self,
        roi_bgr_u8: np.ndarray,
        target_face_meta: FaceMetadata,
    ) -> Optional[np.ndarray]:
        if roi_bgr_u8 is None:
            return None

        self._swap_counter += 1
        swap_index = self._swap_counter

        kps5 = self._extract_target_kps5(target_face_meta)
        aligned_bgr_u8, affine_matrix = self._align_face(roi_bgr_u8, kps5)

        img_in = self._prepare_simswap_image(aligned_bgr_u8)
        emb_in = self._runtime_simswap_embedding(roi_bgr_u8, target_face_meta)
        raw = self._run_simswap_once(img_in, emb_in)
        swapped_crop_raw_bgr_u8 = self._decode_simswap_output(raw)

        crop_masks: List[np.ndarray] = []
        crop_masks.append(
            self._create_box_mask(
                swapped_crop_raw_bgr_u8,
                face_mask_blur=self.BOX_MASK_BLUR,
                face_mask_padding=self.BOX_MASK_PADDING,
            )
        )

        if self.USE_AREA_MASK_IF_68:
            kps68 = self._extract_target_kps68(target_face_meta)
            if kps68 is not None:
                kps68_aligned = self._transform_points(kps68, affine_matrix)
                crop_masks.append(
                    self._create_area_mask_from_landmarks68(
                        crop_size=int(swapped_crop_raw_bgr_u8.shape[0]),
                        landmarks68_aligned=kps68_aligned,
                        sigma=self.AREA_MASK_SIGMA,
                    )
                )

        crop_mask = np.minimum.reduce(crop_masks).clip(0.0, 1.0)

        swapped_crop_final_bgr_u8, crop_mask_final, correction_info = self._maybe_correct_swapped_geometry(
            swapped_crop_bgr_u8=swapped_crop_raw_bgr_u8,
            crop_mask_f32=crop_mask,
            target_kps5_aligned=kps5,
        )

        pasted_roi = self._paste_back(
            roi_bgr_u8=roi_bgr_u8,
            crop_bgr_u8=swapped_crop_final_bgr_u8,
            crop_mask_f32=crop_mask_final,
            affine_matrix=affine_matrix,
        )

        self._dump_debug_artifacts(
            swap_index=swap_index,
            aligned_target_bgr_u8=aligned_bgr_u8,
            swapped_crop_raw_bgr_u8=swapped_crop_raw_bgr_u8,
            swapped_crop_final_bgr_u8=swapped_crop_final_bgr_u8,
            crop_mask_f32=crop_mask_final,
            pasted_roi_bgr_u8=pasted_roi,
            correction_info=correction_info,
        )
        return pasted_roi


__all__ = ["SimSwapWorker"]
