from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from gRestorer.detector.core import FaceMetadata


class FaceEnhancer:
    """Optional post-swap face enhancer.

    Designed for single-input/single-output ONNX face enhancers such as GFPGAN-like
    models that accept a square RGB tensor and return an enhanced square RGB tensor.

    The enhancer is intentionally local and stateless:
      - no clip policy
      - no temporal smoothing
      - no full-frame awareness
    """

    def __init__(
        self,
        device: torch.device,
        enhancer_model_path: str,
        *,
        provider: str = 'auto',
        blend: int = 80,
        video_memory_strategy: str = 'strict',
        gpu_mem_limit_mb: int = 256,
        cuda_use_max_workspace: bool = False,
    ) -> None:
        self.device = device
        self.enhancer_model_path = str(enhancer_model_path)
        self.provider = str(provider or 'auto').lower()
        self.blend = int(max(0, min(100, blend)))
        self.video_memory_strategy = str(video_memory_strategy or 'strict').lower()
        self.gpu_mem_limit_mb = int(gpu_mem_limit_mb)
        self.cuda_use_max_workspace = bool(cuda_use_max_workspace)

        try:
            import onnxruntime as ort
        except Exception as e:
            raise ImportError('FaceEnhancer requires onnxruntime / onnxruntime-gpu in the gRestorer environment.') from e

        self._ort = ort

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
            self.enhancer_model_path,
            providers=providers,
            provider_options=provider_options,
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self.input_size = self._infer_input_size(self._session)

        print(
            f"[FaceEnhancer] provider={self.provider} input_size={self.input_size} "
            f"blend={self.blend} memory_strategy={self.video_memory_strategy} "
            f"gpu_mem_limit_mb={self.gpu_mem_limit_mb} max_workspace={self.cuda_use_max_workspace}"
        )

    @staticmethod
    def _memory_strategy_defaults(strategy: str) -> tuple[int, bool]:
        s = str(strategy or 'strict').lower()
        if s == 'tolerant':
            return 512, True
        if s == 'moderate':
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
        p = str(provider or 'auto').lower()
        if p == 'cpu' or device.type != 'cuda':
            return ['CPUExecutionProvider'], [{}]

        device_id = 0 if device.index is None else int(device.index)
        limit_bytes = int(max(0, gpu_mem_limit_mb)) * 1024 * 1024

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = [
            {
                'device_id': str(device_id),
                'gpu_mem_limit': str(limit_bytes),
                'arena_extend_strategy': 'kSameAsRequested',
                'cudnn_conv_algo_search': 'DEFAULT',
                'cudnn_conv_use_max_workspace': '1' if use_max_workspace else '0',
                'do_copy_in_default_stream': '1',
            },
            {},
        ]
        return providers, provider_options

    @staticmethod
    def _infer_input_size(session) -> int:
        shape = list(session.get_inputs()[0].shape)
        for v in shape[::-1]:
            if isinstance(v, int) and v > 0:
                return int(v)
        return 512

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
        return FaceEnhancer._bbox_to_five_points(face_meta)

    def _estimate_affine(self, src_pts: np.ndarray) -> np.ndarray:
        dst_pts = self._arcface_template(self.input_size)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        if M is None:
            raise RuntimeError('Failed to estimate affine transform for face enhancement.')
        return M.astype(np.float32)

    def _run_model(self, aligned_bgr_u8: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(aligned_bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32)
        inp = (rgb / 127.5) - 1.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
        out = self._session.run([self._output_name], {self._input_name: inp})[0]
        out = np.asarray(out)
        if out.ndim == 4:
            out = out[0]
        if out.ndim == 3 and out.shape[0] in (1, 3):
            out = np.transpose(out, (1, 2, 0))
        if out.ndim != 3 or out.shape[2] not in (1, 3):
            raise RuntimeError(f'Unsupported enhancer output shape: {tuple(np.asarray(out).shape)}')
        if out.shape[2] == 1:
            out = np.repeat(out, 3, axis=2)
        out = out.astype(np.float32)
        # Most face enhancers emit [-1, 1]; allow [0, 1] or [0, 255] too.
        if float(out.min()) < -0.25:
            out = (out + 1.0) * 127.5
        elif float(out.max()) <= 1.25:
            out = out * 255.0
        out = np.clip(out, 0.0, 255.0).round().astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _soft_mask(size: int) -> np.ndarray:
        m = np.ones((size, size), dtype=np.float32)
        border = max(8, size // 16)
        m[:border, :] = 0.0
        m[-border:, :] = 0.0
        m[:, :border] = 0.0
        m[:, -border:] = 0.0
        blur = max(5, (border // 2) * 2 + 1)
        return cv2.GaussianBlur(m, (blur, blur), 0)

    def enhance(self, roi_bgr_u8: np.ndarray, face_meta: FaceMetadata) -> Optional[np.ndarray]:
        if roi_bgr_u8 is None or face_meta is None:
            return None
        if self.blend <= 0:
            return roi_bgr_u8

        h, w = roi_bgr_u8.shape[:2]
        src_pts = self._five_points(face_meta)
        M = self._estimate_affine(src_pts)

        aligned = cv2.warpAffine(
            roi_bgr_u8,
            M,
            (self.input_size, self.input_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        enhanced = self._run_model(aligned)

        M_inv = cv2.invertAffineTransform(M)
        warped = cv2.warpAffine(
            enhanced,
            M_inv,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        mask = self._soft_mask(self.input_size)
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

        base = roi_bgr_u8.astype(np.float32)
        over = warped.astype(np.float32)
        out = base * (1.0 - warped_mask) + over * warped_mask
        return np.clip(out, 0.0, 255.0).round().astype(np.uint8)


__all__ = ['FaceEnhancer']
