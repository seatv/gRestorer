from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch

from gRestorer.detector.core import FaceMetadata


class FaceLandmarker:
    """Optional ONNX landmark refinement stage for one detector-selected face.

    This class does *not* decide which face to use. It only refines landmarks for the
    single target face metadata that the wrapper hands to it.
    """

    def __init__(
        self,
        device: torch.device,
        model_name: str,
        model_path: str,
        *,
        provider: str = "auto",
        score: float = 0.5,
        video_memory_strategy: str = "strict",
        gpu_mem_limit_mb: int = 256,
        cuda_use_max_workspace: bool = False,
    ) -> None:
        self.device = device
        self.model_name = str(model_name or "2dfan4").lower()
        self.model_path = str(model_path)
        self.provider = str(provider or "auto").lower()
        self.score = float(max(0.0, min(1.0, score)))
        self.video_memory_strategy = str(video_memory_strategy or "strict").lower()
        self.gpu_mem_limit_mb = int(gpu_mem_limit_mb)
        self.cuda_use_max_workspace = bool(cuda_use_max_workspace)

        try:
            import onnxruntime as ort
        except Exception as e:
            raise ImportError("FaceLandmarker requires onnxruntime / onnxruntime-gpu in the gRestorer environment.") from e

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
            self.model_path,
            providers=providers,
            provider_options=provider_options,
        )
        self._input = self._session.get_inputs()[0]
        self._input_name = self._input.name
        self._input_shape = tuple(self._input.shape)
        self.input_size = self._infer_input_size(self._input_shape)
        self.input_layout = self._infer_input_layout(self._input_shape)
        self.crop_scale = 1.8

        print(
            f"[FaceLandmarker] enabled model={self.model_name} provider={self.provider} "
            f"input_size={self.input_size} layout={self.input_layout} score={self.score:.2f} "
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
        dims = [int(v) for v in shape if isinstance(v, int) and v > 0]
        if len(dims) >= 2:
            if dims[-1] >= 32:
                return int(dims[-1])
            if dims[-2] >= 32:
                return int(dims[-2])
        return 256

    @staticmethod
    def _infer_input_layout(shape) -> str:
        if len(shape) == 4:
            if isinstance(shape[1], int) and int(shape[1]) == 3:
                return "nchw"
            if isinstance(shape[-1], int) and int(shape[-1]) == 3:
                return "nhwc"
        return "nchw"

    @staticmethod
    def _clamp_points(pts: np.ndarray, w: int, h: int) -> np.ndarray:
        out = np.asarray(pts, dtype=np.float32).copy()
        if out.size == 0:
            return out
        out[:, 0] = np.clip(out[:, 0], 0.0, float(max(0, w - 1)))
        out[:, 1] = np.clip(out[:, 1], 0.0, float(max(0, h - 1)))
        return out

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
        left_eye = np.mean(pts68[36:42], axis=0)
        right_eye = np.mean(pts68[42:48], axis=0)
        nose = pts68[30]
        mouth_left = pts68[48]
        mouth_right = pts68[54]
        return np.stack([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0).astype(np.float32)

    def _prepare_crop(self, roi_bgr_u8: np.ndarray, face_meta: FaceMetadata) -> tuple[np.ndarray, tuple[int, int, int]]:
        h, w = roi_bgr_u8.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in face_meta.bbox_xyxy]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        side = max(32, int(round(max(bw, bh) * self.crop_scale)))
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        crop_x1 = int(round(cx - side / 2.0))
        crop_y1 = int(round(cy - side / 2.0))
        crop_x2 = crop_x1 + side
        crop_y2 = crop_y1 + side

        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(w, crop_x2)
        src_y2 = min(h, crop_y2)

        crop = roi_bgr_u8[src_y1:src_y2, src_x1:src_x2]
        if crop.size == 0:
            raise RuntimeError("Invalid face crop for landmark refinement.")

        pad_left = max(0, -crop_x1)
        pad_top = max(0, -crop_y1)
        pad_right = max(0, crop_x2 - w)
        pad_bottom = max(0, crop_y2 - h)
        if pad_left or pad_top or pad_right or pad_bottom:
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_REFLECT_101,
            )

        crop = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        return crop, (crop_x1, crop_y1, side)

    def _prepare_input(self, crop_bgr_u8: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.input_layout == "nchw":
            return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
        return rgb[None, ...].astype(np.float32, copy=False)

    @staticmethod
    def _pick_output(outputs: List[np.ndarray]) -> np.ndarray:
        best = None
        best_rank = (-1, -1)
        for out in outputs:
            arr = np.asarray(out)
            shape = tuple(int(s) for s in arr.shape if isinstance(s, (int, np.integer)))
            score = (arr.ndim, int(np.prod(shape)) if shape else int(arr.size))
            if best is None or score > best_rank:
                best = arr
                best_rank = score
        if best is None:
            raise RuntimeError("Landmarker produced no outputs.")
        return np.asarray(best)

    def _decode_heatmaps(self, arr: np.ndarray, crop_origin: tuple[int, int, int], roi_w: int, roi_h: int) -> tuple[np.ndarray, Optional[float]]:
        hm = np.asarray(arr, dtype=np.float32)
        if hm.ndim == 4:
            hm = hm[0]
        if hm.ndim != 3:
            raise RuntimeError(f"Unsupported heatmap output shape: {tuple(arr.shape)}")
        n_pts, hm_h, hm_w = hm.shape
        pts = np.zeros((n_pts, 2), dtype=np.float32)
        max_vals = []
        crop_x1, crop_y1, side = crop_origin
        sx = float(side) / float(max(1, hm_w))
        sy = float(side) / float(max(1, hm_h))
        for i in range(n_pts):
            ch = hm[i]
            idx = int(np.argmax(ch))
            yy = idx // hm_w
            xx = idx % hm_w
            max_val = float(ch[yy, xx])
            if max_val < 0.0 or max_val > 1.0:
                max_val = 1.0 / (1.0 + np.exp(-max_val))
            max_vals.append(max_val)
            pts[i, 0] = float(crop_x1) + (float(xx) + 0.5) * sx
            pts[i, 1] = float(crop_y1) + (float(yy) + 0.5) * sy
        pts = self._clamp_points(pts, roi_w, roi_h)
        conf = float(np.mean(max_vals)) if max_vals else None
        return pts, conf

    def _decode_coordinates(self, arr: np.ndarray, crop_origin: tuple[int, int, int], roi_w: int, roi_h: int) -> tuple[np.ndarray, Optional[float]]:
        pts = np.asarray(arr, dtype=np.float32)
        if pts.ndim == 3 and pts.shape[0] == 1:
            pts = pts[0]
        if pts.ndim == 1:
            if pts.size % 2 != 0:
                raise RuntimeError(f"Unsupported landmark coordinate shape: {tuple(arr.shape)}")
            pts = pts.reshape(-1, 2)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise RuntimeError(f"Unsupported landmark coordinate shape: {tuple(arr.shape)}")

        crop_x1, crop_y1, side = crop_origin
        pts = pts.copy()
        vmax = float(np.max(pts)) if pts.size else 0.0
        vmin = float(np.min(pts)) if pts.size else 0.0
        if -1.5 <= vmin <= 1.5 and -1.5 <= vmax <= 1.5:
            if vmin < 0.0:
                pts = (pts + 1.0) * 0.5 * float(self.input_size)
            else:
                pts = pts * float(self.input_size)
        scale = float(side) / float(max(1, self.input_size))
        pts[:, 0] = float(crop_x1) + pts[:, 0] * scale
        pts[:, 1] = float(crop_y1) + pts[:, 1] * scale
        pts = self._clamp_points(pts, roi_w, roi_h)
        return pts, None

    def _decode_output(self, arr: np.ndarray, crop_origin: tuple[int, int, int], roi_w: int, roi_h: int) -> tuple[np.ndarray, Optional[float]]:
        a = np.asarray(arr)
        if a.ndim >= 3 and ((a.ndim == 4 and a.shape[0] == 1) or a.ndim == 3):
            shape = a.shape[1:] if a.ndim == 4 else a.shape
            if len(shape) == 3 and int(shape[1]) >= 8 and int(shape[2]) >= 8:
                return self._decode_heatmaps(a, crop_origin, roi_w, roi_h)
        return self._decode_coordinates(a, crop_origin, roi_w, roi_h)

    def _to_swapper_five_points(self, pts: np.ndarray, face_meta: FaceMetadata) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape[0] == 5:
            return pts
        if pts.shape[0] >= 68 and pts.shape[0] < 100:
            return self._landmarks68_to_five_points(pts[:68])
        return self._bbox_to_five_points(face_meta)

    def refine(self, roi_bgr_u8: np.ndarray, face_meta: FaceMetadata) -> Optional[FaceMetadata]:
        if roi_bgr_u8 is None or face_meta is None:
            return None

        h, w = roi_bgr_u8.shape[:2]
        crop, crop_origin = self._prepare_crop(roi_bgr_u8, face_meta)
        inp = self._prepare_input(crop)
        outputs = self._session.run(None, {self._input_name: inp})
        raw = self._pick_output(outputs)
        pts, conf = self._decode_output(raw, crop_origin, w, h)
        if conf is not None and conf < self.score:
            return None

        five = self._to_swapper_five_points(pts, face_meta)
        kps = torch.from_numpy(np.ascontiguousarray(five)).to(torch.float32)
        return FaceMetadata(
            bbox_xyxy=tuple(float(v) for v in face_meta.bbox_xyxy),
            kps=kps,
            det_score=face_meta.det_score,
        )


__all__ = ["FaceLandmarker"]
