from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch

from gRestorer.detector.core import FaceMetadata


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
        dims = [int(v) for v in shape if isinstance(v, int) and v > 0]
        if not dims:
            return 256
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
        rgb = cv2.cvtColor(aligned_bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.input_layout == "nchw":
            return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
        return rgb[None, ...].astype(np.float32, copy=False)

    def _run_model(self, aligned_bgr_u8: np.ndarray) -> np.ndarray:
        inp = self._prepare_input(aligned_bgr_u8)
        out = self._session.run([self._output_name], {self._input_name: inp})[0]
        out = np.asarray(out)
        if out.ndim == 4:
            out = out[0]
        if out.ndim == 3:
            if out.shape[0] in (1, 2, 3):
                out = out[0] if out.shape[0] == 1 else np.max(out, axis=0)
            elif out.shape[-1] in (1, 2, 3):
                out = out[..., 0] if out.shape[-1] == 1 else np.max(out, axis=-1)
        if out.ndim != 2:
            raise RuntimeError(f"Unsupported occluder output shape: {tuple(np.asarray(out).shape)}")

        out = out.astype(np.float32)
        if float(out.min()) < -0.25:
            out = (out + 1.0) * 0.5
        elif float(out.max()) > 1.25:
            out = out / 255.0
        out = np.clip(out, 0.0, 1.0)
        return out

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

    def preserve(self, original_roi_bgr_u8: np.ndarray, modified_roi_bgr_u8: np.ndarray, face_meta: FaceMetadata) -> Optional[np.ndarray]:
        if original_roi_bgr_u8 is None or modified_roi_bgr_u8 is None or face_meta is None:
            return None
        if self.blend <= 0:
            return modified_roi_bgr_u8

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
        mask = self._run_model(aligned)
        mask = self._postprocess_mask(mask)

        M_inv = cv2.invertAffineTransform(M)
        warped_mask = cv2.warpAffine(
            mask,
            M_inv,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        warped_mask = np.clip(warped_mask, 0.0, 1.0)[..., None]
        warped_mask *= float(self.blend) / 100.0

        orig = original_roi_bgr_u8.astype(np.float32)
        mod = modified_roi_bgr_u8.astype(np.float32)
        out = mod * (1.0 - warped_mask) + orig * warped_mask
        return np.clip(out, 0.0, 255.0).round().astype(np.uint8)


__all__ = ["FaceOccluder"]
