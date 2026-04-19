from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from gRestorer.detector.core import FaceMetadata
from gRestorer.restorer.face_types import FaceSwapBackendResult



class SimSwapWorker:
    """Pure SimSwap worker.

    This class owns only the SimSwap-specific contract:
      - source embedding preparation
      - target face alignment
      - ONNX inference
      - output decode
      - backend-result construction

    It knows nothing about clip-level policy, target selection, or adjacent frames.
    """

    # FaceFusion's SimSwap models use arcface_112_v1 geometry.
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

    @staticmethod
    def _u8_to_f32(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.dtype == np.uint8:
            return np.ascontiguousarray(arr.astype(np.float32) / 255.0)
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

    def __init__(
        self,
        device: torch.device,
        source_face_path: str,
        swap_model_path: str,
        *,
        provider: str = "auto",
        embedding_converter_path: str | None = None,
    ) -> None:
        self.device = device
        self.source_face_path = str(source_face_path)
        self.swap_model_path = str(swap_model_path)
        self.provider = str(provider or "auto").lower()

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
        self._output_name = None
        self._image_input_name = None
        self._embedding_input_name = None
        self._image_input_layout = "nchw"
        self._image_size = 256
        self._model_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._model_std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self._init_simswap_contract()

        # FaceFusion uses a separate arcface_converter_simswap ONNX for SimSwap source embeddings.
        self._embedding_converter_path = self._resolve_embedding_converter_path(embedding_converter_path)
        self._embedding_converter = None
        if self._embedding_converter_path:
            self._embedding_converter = ort.InferenceSession(self._embedding_converter_path, providers=providers)

        self._src_embedding = self._prepare_simswap_embedding()

        print(
            f"[SimSwapWorker] provider={self.provider} image_size={self._image_size} "
            f"layout={self._image_input_layout} image_input={self._image_input_name} "
            f"embedding_input={self._embedding_input_name} converter={'yes' if self._embedding_converter is not None else 'no'}"
        )

    @staticmethod
    def _u8_to_f32(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.dtype == np.uint8:
            return np.ascontiguousarray(arr.astype(np.float32) / 255.0)
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

    def _resolve_embedding_converter_path(self, explicit: str | None) -> str:
        if explicit:
            p = Path(explicit)
            return str(p) if p.exists() else ""
        model_path = Path(self.swap_model_path)
        candidates = [
            model_path.with_name("arcface_converter_simswap.onnx"),
            model_path.parent / "arcface_converter_simswap.onnx",
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

        # FaceFusion model metadata uses ImageNet normalization for simswap_256 and 0/1 for unofficial_512.
        name = Path(self.swap_model_path).name.lower()
        if "unofficial_512" in name or "512_unofficial" in name:
            self._model_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self._model_std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            self._model_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self._model_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _extract_target_kps5(face_meta: FaceMetadata) -> np.ndarray:
        if face_meta.kps is not None:
            kps = np.asarray(face_meta.kps.detach().cpu().numpy(), dtype=np.float32)
            if kps.shape[0] >= 5:
                return kps[:5].astype(np.float32, copy=True)
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

    def _align_face(self, roi_bgr_u8: np.ndarray, kps5: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # FaceFusion uses arcface_112_v1 for both SimSwap models and scales that template to the crop size.
        dst = self.ARC_DST_5_V1.copy()
        if self._image_size != 112:
            dst *= float(self._image_size) / 112.0
        M, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), dst.astype(np.float32), method=cv2.LMEDS)
        if M is None:
            raise RuntimeError("Failed to estimate affine transform for SimSwap alignment.")
        aligned = cv2.warpAffine(
            roi_bgr_u8,
            M,
            (self._image_size, self._image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return aligned, M

    def _prepare_simswap_image(self, aligned_bgr_u8: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(aligned_bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._model_mean) / self._model_std
        if self._image_input_layout == "nchw":
            rgb = np.transpose(rgb, (2, 0, 1))[None, ...]
        else:
            rgb = rgb[None, ...]
        return np.ascontiguousarray(rgb.astype(np.float32, copy=False))

    def _prepare_simswap_embedding(self) -> np.ndarray:
        emb = getattr(self._src_face, "embedding", None)
        if emb is None:
            emb = getattr(self._src_face, "normed_embedding", None)
        if emb is None:
            raise RuntimeError("Source face embedding missing from insightface result.")
        emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        if self._embedding_converter is not None:
            conv_inputs = self._embedding_converter.get_inputs()
            conv_name = conv_inputs[0].name if conv_inputs else "input"
            emb = self._embedding_converter.run(None, {conv_name: emb})[0]
        emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(emb))
        if norm > 0.0:
            emb = emb / norm
        return np.ascontiguousarray(emb.astype(np.float32, copy=False))

    @staticmethod
    def _decode_simswap_output(out: np.ndarray) -> np.ndarray:
        arr = np.asarray(out)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            raise RuntimeError(f"Unexpected SimSwap output shape: {tuple(out.shape)}")
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
    def _make_face_mask(size: int) -> np.ndarray:
        mask = np.zeros((size, size), dtype=np.float32)
        center = (size // 2, int(round(size * 0.54)))
        axes = (int(round(size * 0.23)), int(round(size * 0.31)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, size / 48.0), sigmaY=max(1.0, size / 48.0))
        return np.clip(mask, 0.0, 1.0)

    def _paste_aligned_face_back(
        self,
        roi_bgr_u8: np.ndarray,
        swapped_aligned_bgr_u8: np.ndarray,
        M: np.ndarray,
    ) -> np.ndarray:
        size = int(swapped_aligned_bgr_u8.shape[0])
        mask = self._make_face_mask(size)
        h, w = roi_bgr_u8.shape[:2]
        inv_M = cv2.invertAffineTransform(M)
        warped = cv2.warpAffine(
            swapped_aligned_bgr_u8,
            inv_M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask_warp = cv2.warpAffine(
            mask,
            inv_M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        mask_warp = np.clip(mask_warp, 0.0, 1.0)[..., None]
        out = roi_bgr_u8.astype(np.float32) * (1.0 - mask_warp) + warped.astype(np.float32) * mask_warp
        return np.clip(out, 0.0, 255.0).round().astype(np.uint8)

    def swap_result(
        self,
        roi_bgr_u8: np.ndarray,
        target_face_meta: FaceMetadata,
    ) -> Optional[FaceSwapBackendResult]:
        if roi_bgr_u8 is None:
            return None

        kps5 = self._extract_target_kps5(target_face_meta)
        aligned, M = self._align_face(roi_bgr_u8, kps5)
        img_in = self._prepare_simswap_image(aligned)
        emb_in = self._src_embedding
        raw = self._run_simswap_once(img_in, emb_in)
        swapped_aligned = self._decode_simswap_output(raw)

        return FaceSwapBackendResult(
            aligned_swapped_bgr_u8=swapped_aligned,
            aligned_target_bgr_u8=aligned,
            aligned_backend_mask_f32=self._make_face_mask(int(swapped_aligned.shape[0])),
            roi_to_aligned=np.asarray(M, dtype=np.float32),
            aligned_to_roi=np.asarray(cv2.invertAffineTransform(M), dtype=np.float32),
            aligned_size=int(swapped_aligned.shape[0]),
            debug={
                "backend": "simswap",
                "image_size": int(self._image_size),
                "provider": self.provider,
            },
        )

    def swap_result(
        self,
        roi_bgr_u8: np.ndarray,
        target_face_meta: FaceMetadata,
    ) -> Optional[FaceSwapBackendResult]:
        if roi_bgr_u8 is None:
            return None

        kps5 = self._extract_target_kps5(target_face_meta)
        aligned, M = self._align_face(roi_bgr_u8, kps5)
        img_in = self._prepare_simswap_image(aligned)
        emb_in = self._src_embedding
        raw = self._run_simswap_once(img_in, emb_in)
        swapped_aligned = self._decode_simswap_output(raw)

        target_landmarks_aligned = cv2.transform(
            kps5.reshape(-1, 1, 2).astype(np.float32),
            M.astype(np.float32),
        ).reshape(-1, 2)

        pred_src = self._make_face_mask(int(swapped_aligned.shape[0]))
        pred_dst = pred_src.copy()

        return FaceSwapBackendResult(
            swapped_face_f32=self._u8_to_f32(swapped_aligned),
            pred_src_mask_f32=np.ascontiguousarray(pred_src.astype(np.float32)),
            pred_dst_mask_f32=np.ascontiguousarray(pred_dst.astype(np.float32)),
            aligned_target_f32=self._u8_to_f32(aligned),
            target_landmarks_aligned=np.ascontiguousarray(target_landmarks_aligned.astype(np.float32)),
            source_landmarks_aligned=None,
            roi_to_aligned=np.asarray(M, dtype=np.float32),
            aligned_to_roi=np.asarray(cv2.invertAffineTransform(M), dtype=np.float32),
            aligned_size=int(swapped_aligned.shape[0]),
            quality=None,
            backend="simswap",
            debug={
                "backend": "simswap",
                "image_size": int(self._image_size),
                "provider": self.provider,
            },
        )

    def swap(self, roi_bgr_u8: np.ndarray, target_face_meta: FaceMetadata) -> Optional[np.ndarray]:
        result = self.swap_result(roi_bgr_u8, target_face_meta)
        if result is None:
            return None
        return self._paste_aligned_face_back(
            roi_bgr_u8,
            self._f32_to_u8(result.swapped_face_f32),
            result.roi_to_aligned,
        )
__all__ = ["SimSwapWorker"]
