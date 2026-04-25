from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch

from gRestorer.detector.core import FaceMetadata
from .face_types import FaceOcclusionMaskResult

class FaceOccluder:
    """Optional post-swap occluder-preserve stage.

    The occluder predicts an occlusion mask from the *original* ROI and then preserves
    those regions by blending pixels from the original ROI back over the swapped/enhanced ROI.

    This keeps hands / microphones / glasses / similar occluders from being rewritten by
    the swapper, without touching the shared compositor.
    """

    def __init__(
        self,
        device: torch.device,
        occluder_model_path: str,
        *,
        provider: str = "auto",
        threshold: float = 0.5,
        blur: int = 5,
        blend: int = 100,
        invert: bool = False,
    ) -> None:
        self.device = device
        self.occluder_model_path = str(occluder_model_path)
        self.provider = str(provider or "auto").lower()
        self.threshold = float(max(0.0, min(1.0, threshold)))
        self.blur = int(max(0, blur))
        self.blend = int(max(0, min(100, blend)))
        self.invert = bool(invert)

        try:
            import onnxruntime as ort
        except Exception as e:
            raise ImportError("FaceOccluder requires onnxruntime / onnxruntime-gpu in the gRestorer environment.") from e

        providers = self._providers_for(self.provider, self.device)
        self._session = ort.InferenceSession(self.occluder_model_path, providers=providers)
        self._input = self._session.get_inputs()[0]
        self._output = self._session.get_outputs()[0]
        self._input_name = self._input.name
        self._output_name = self._output.name
        self.input_size = self._infer_input_size(self._input.shape)
        self.input_layout = self._infer_input_layout(self._input.shape)

        print(
            f"[FaceOccluder] provider={self.provider} input_size={self.input_size} "
            f"layout={self.input_layout} input={self._input_name} output={self._output_name}"
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
    def _infer_input_size(shape) -> int:
        # NCHW: [N,3,H,W] -> size is W/H
        # NHWC: [N,H,W,3] -> size is H/W, not the trailing channel dim.
        if len(shape) == 4:
            if isinstance(shape[1], int) and int(shape[1]) == 3:
                hw = [int(v) for v in shape[2:] if isinstance(v, int) and int(v) > 0]
                if hw:
                    return int(hw[-1])
            if isinstance(shape[-1], int) and int(shape[-1]) == 3:
                hw = [int(v) for v in shape[1:3] if isinstance(v, int) and int(v) > 0]
                if hw:
                    return int(hw[-1])
        dims = [int(v) for v in shape if isinstance(v, int) and int(v) > 0]
        if not dims:
            return 256
        for v in reversed(dims):
            if v > 4:
                return int(v)
        return int(dims[-1])

    @staticmethod
    def _infer_input_layout(shape) -> str:
        if len(shape) == 4:
            if isinstance(shape[1], int) and int(shape[1]) == 3:
                return "nchw"
            if isinstance(shape[-1], int) and int(shape[-1]) == 3:
                return "nhwc"
        return "nchw"

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
            kps = face_meta.kps.detach().cpu().numpy().astype(np.float32, copy=True)
            if kps.ndim == 2 and kps.shape == (5, 2):
                return kps
        return FaceOccluder._bbox_to_five_points(face_meta)

    def _estimate_affine(self, src_pts: np.ndarray) -> np.ndarray:
        dst_pts = self._arcface_template(self.input_size)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        if M is None:
            raise RuntimeError("Failed to estimate affine transform for face occlusion.")
        return M.astype(np.float32)

    def _prepare_input(self, aligned_bgr_u8: np.ndarray) -> np.ndarray:
        # XSeg-style occluder models expect the resized crop scaled to [0,1].
        # Keep channel order as BGR; do not convert to RGB here.
        x = aligned_bgr_u8.astype(np.float32) / 255.0
        if self.input_layout == "nchw":
            return np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
        return x[None, ...].astype(np.float32, copy=False)

    def _run_model(self, aligned_bgr_u8: np.ndarray) -> np.ndarray:
        inp = self._prepare_input(aligned_bgr_u8)
        out = self._session.run([self._output_name], {self._input_name: inp})[0]
        arr = np.asarray(out)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3:
            if arr.shape[0] in (1, 2, 3):
                arr = arr[0] if arr.shape[0] == 1 else np.max(arr, axis=0)
            elif arr.shape[-1] in (1, 2, 3):
                arr = arr[..., 0] if arr.shape[-1] == 1 else np.max(arr, axis=-1)
        if arr.ndim != 2:
            raise RuntimeError(f"Unsupported occluder output shape: {tuple(np.asarray(arr).shape)}")

        arr = arr.astype(np.float32)
        # Common output conventions: logits, [-1,1], [0,255], or [0,1]
        if float(arr.min()) < -0.25 and float(arr.max()) <= 1.25:
            arr = (arr + 1.0) * 0.5
        elif float(arr.max()) > 1.25:
            # Prefer sigmoid for logits-like outputs; fall back to /255 for mask-like outputs.
            if float(arr.max()) > 20.0 or float(arr.min()) < -2.0:
                arr = 1.0 / (1.0 + np.exp(-arr))
            else:
                arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        return arr

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.invert:
            mask = 1.0 - mask
        mask = (mask >= self.threshold).astype(np.float32)
        if self.blur > 0:
            k = int(self.blur)
            if k % 2 == 0:
                k += 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return np.clip(mask, 0.0, 1.0)

    def preserve(
            self,
            original_roi_bgr_u8: np.ndarray,
            modified_roi_bgr_u8: np.ndarray,
            face_meta: FaceMetadata,
    ) -> Optional[np.ndarray]:
        if original_roi_bgr_u8 is None or modified_roi_bgr_u8 is None or face_meta is None:
            return None

        result = self.build_keep_mask(original_roi_bgr_u8, face_meta)
        if result is None:
            return modified_roi_bgr_u8

        keep = result.keep_mask_f32[..., None]
        orig = original_roi_bgr_u8.astype(np.float32)
        mod = modified_roi_bgr_u8.astype(np.float32)
        out = mod * (1.0 - keep) + orig * keep
        return np.clip(out, 0.0, 255.0).round().astype(np.uint8)

    def build_keep_mask(
            self,
            original_roi_bgr_u8: np.ndarray,
            face_meta: FaceMetadata,
    ) -> Optional[FaceOcclusionMaskResult]:
        if original_roi_bgr_u8 is None or face_meta is None:
            return None
        if self.blend <= 0:
            h, w = original_roi_bgr_u8.shape[:2]
            return FaceOcclusionMaskResult(
                keep_mask_f32=np.zeros((h, w), dtype=np.float32),
                aligned_input_bgr_u8=None,
                aligned_raw_mask_f32=None,
                aligned_keep_mask_f32=None,
            )

        h, w = original_roi_bgr_u8.shape[:2]
        src_pts = self._five_points(face_meta)
        M = self._estimate_affine(src_pts)

        aligned = cv2.warpAffine(
            original_roi_bgr_u8,
            M,
            (self.input_size, self.input_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        raw_mask = self._run_model(aligned)
        keep_mask_aligned = self._postprocess_mask(raw_mask)

        M_inv = cv2.invertAffineTransform(M)
        keep_mask_roi = cv2.warpAffine(
            keep_mask_aligned,
            M_inv,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ).astype(np.float32)

        keep_mask_roi = np.clip(keep_mask_roi, 0.0, 1.0)
        keep_mask_roi *= float(self.blend) / 100.0

        return FaceOcclusionMaskResult(
            keep_mask_f32=keep_mask_roi,
            aligned_input_bgr_u8=aligned,
            aligned_raw_mask_f32=np.clip(raw_mask.astype(np.float32), 0.0, 1.0),
            aligned_keep_mask_f32=np.clip(keep_mask_aligned.astype(np.float32), 0.0, 1.0),
        )
__all__ = ["FaceOccluder"]
