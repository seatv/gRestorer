from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import cv2
import numpy as np
import torch

from gRestorer.detector.core import FaceMetadata


class InSwapperWorker:
    """Pure face-swap worker.

    Contract:
      - Knows nothing about whole-frame policy, scene tracking, adjacent frames,
        clip-level anchor selection, or whether a frame should be swapped.
      - Given one ROI crop + one target-face metadata object + one enrolled
        source face, returns its best swap result for that ROI.
    """

    def __init__(
        self,
        device: torch.device,
        source_face_path: str,
        swap_model_path: str,
        *,
        provider: str = "auto",
    ) -> None:
        self.device = device
        self.source_face_path = str(source_face_path)
        self.swap_model_path = str(swap_model_path)
        self.provider = str(provider or "auto").lower()

        try:
            from insightface.app import FaceAnalysis
            from insightface.model_zoo import get_model
        except Exception as e:
            raise ImportError(
                "FaceSwapWorker requires `insightface` in the gRestorer environment."
            ) from e

        providers = self._providers_for(self.provider, self.device)
        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        ctx_id = 0 if self.device.type == "cuda" else -1
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self._swapper = get_model(self.swap_model_path, providers=providers)

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
    def _target_face_from_meta(face_meta: FaceMetadata):
        bbox = np.asarray(face_meta.bbox_xyxy, dtype=np.float32)
        kps = None
        if face_meta.kps is not None:
            kps = face_meta.kps.detach().cpu().numpy().astype(np.float32, copy=True)
        return SimpleNamespace(
            bbox=bbox,
            kps=kps,
            det_score=float(face_meta.det_score) if face_meta.det_score is not None else 1.0,
        )

    def swap(self, roi_bgr_u8: np.ndarray, target_face_meta: FaceMetadata) -> Optional[np.ndarray]:
        if roi_bgr_u8 is None:
            return None
        target_face = self._target_face_from_meta(target_face_meta)
        out = self._swapper.get(roi_bgr_u8, target_face, self._src_face, paste_back=True)
        return out


__all__ = ["InSwapperWorker"]