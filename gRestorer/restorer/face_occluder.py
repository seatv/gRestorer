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
        video_memory_strategy: str = "strict",
        gpu_mem_limit_mb: int = 256,
        cuda_use_max_workspace: bool = False,
    ) -> None:
        self.device = device
        self.occluder_model_path = str(occluder_model_path)
        self.provider = str(provider or "auto").lower()
        self.threshold = float(max(0.0, min(1.0, threshold)))
        self.blur = int(max(0, blur))
        self.blend = int(max(0, min(100, blend)))
        self.invert = bool(invert)
        self.video_memory_strategy = str(video_memory_strategy or "strict").lower()
        self.gpu_mem_limit_mb = int(gpu_mem_limit_mb)
        self.cuda_use_max_workspace = bool(cuda_use_max_workspace)

        try:
            import onnxruntime as ort
        except Exception as e:
            raise ImportError("FaceOccluder requires onnxruntime / onnxruntime-gpu in the gRestorer environment.") from e

        strat_limit_mb, strat_workspace = self._memory_strategy_defaults(self.video_memory_strategy)
        if self.gpu_mem_limit_mb <= 0:
            self.gpu_mem_limit_mb = strat_limit_mb
        if not self.cuda_use_max_workspace:
            self.cuda_use_max_workspace = strat_workspace

        providers, provider_options = self._provider_config(
            self.provider,
            self.device,
            gpu_mem_limit_mb=self.gpu_mem_limit_mb,
            use_max_workspace=self.cuda_use_max_workspace,
        )

        self._providers = providers
        self._provider_options = provider_options

        self._session = ort.InferenceSession(
            self.occluder_model_path,
            providers=providers,
            provider_options=provider_options,
        )
        self._input = self._session.get_inputs()[0]
        self._output = self._session.get_outputs()[0]
        self._input_name = self._input.name
        self._output_name = self._output.name
        self.input_size = self._infer_input_size(self._input.shape)
        self.input_layout = self._infer_input_layout(self._input.shape)

        print(
            f"[FaceOccluder] provider={self.provider} input_size={self.input_size} "
            f"layout={self.input_layout} input={self._input_name} output={self._output_name} "
            f"memory_strategy={self.video_memory_strategy} gpu_mem_limit_mb={self.gpu_mem_limit_mb} "
            f"max_workspace={self.cuda_use_max_workspace}"
        )

    @staticmethod
    def _memory_strategy_defaults(strategy: str) -> tuple[int, bool]:
        s = str(strategy or "strict").lower()
        if s == "tolerant":
            return 512, True
        if s == "moderate":
            return 384, False
        return 256, False

    @staticmethod
    def _provider_config(
        provider: str,
        device: torch.device,
        *,
        gpu_mem_limit_mb: int,
        use_max_workspace: bool,
    ) -> tuple[List[str], List[dict]]:
        p = str(provider or "auto").lower()
        if p == "cpu" or device.type != "cuda":
            return ["CPUExecutionProvider"], [{}]

        device_id = 0 if device.index is None else int(device.index)
        limit_bytes = int(max(0, gpu_mem_limit_mb)) * 1024 * 1024

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [
            {
                "device_id": str(device_id),
                "gpu_mem_limit": str(limit_bytes),
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "DEFAULT",
                "cudnn_conv_use_max_workspace": "1" if use_max_workspace else "0",
                "do_copy_in_default_stream": "1",
            },
            {},
        ]
        return providers, provider_options

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
