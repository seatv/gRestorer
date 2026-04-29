from __future__ import annotations

from types import SimpleNamespace
import copy
from typing import List, Optional

import cv2
import numpy as np
import torch

from gRestorer.detector.core import FaceMetadata


class InSwapperWorker:
    """Pure face-swap worker.

    This intentionally stays on the known-good legacy path:
      - no swap_result()
      - no shared aligned compositor path
      - let insightface InSwapper do paste_back=True itself

    This matches the known-good behavior from the attached gRestorer-main.zip.
    """

    def __init__(
        self,
        device: torch.device,
        source_face_path: str,
        swap_model_path: str,
        *,
        provider: str = "auto",
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
        self._src_embedding_for_mix = self._face_embedding(self._src_face)

        print(f"[InSwapperWorker] provider={self.provider} mode=legacy_paste_back_true face_swapper_weight={self.face_swapper_weight:.3f}")

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
                return InSwapperWorker._normalize_embedding(np.asarray(emb, dtype=np.float32))
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
        return InSwapperWorker._normalize_embedding(mixed)

    @staticmethod
    def _target_kps5_from_meta(face_meta: FaceMetadata) -> np.ndarray:
        if face_meta.kps is not None:
            kps = face_meta.kps.detach().cpu().numpy().astype(np.float32, copy=True)
            if kps.ndim == 2:
                if kps.shape == (5, 2):
                    return np.ascontiguousarray(kps)
                if kps.shape[0] >= 68:
                    left_eye = np.mean(kps[36:42], axis=0)
                    right_eye = np.mean(kps[42:48], axis=0)
                    nose = kps[30]
                    mouth_left = kps[48]
                    mouth_right = kps[54]
                    return np.ascontiguousarray(
                        np.stack([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0).astype(np.float32)
                    )
                if kps.shape[0] >= 5:
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
    def _target_face_from_meta(face_meta: FaceMetadata):
        bbox = np.asarray(face_meta.bbox_xyxy, dtype=np.float32)
        kps = InSwapperWorker._target_kps5_from_meta(face_meta)
        return SimpleNamespace(
            bbox=bbox,
            kps=kps,
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

    def _source_face_for_swap(self, roi_bgr_u8: np.ndarray, target_face_meta: FaceMetadata):
        if self.face_swapper_weight >= 0.999 or self._src_embedding_for_mix is None:
            self._set_last_embedding_metrics(
                source_embedding=self._src_embedding_for_mix,
                target_embedding=None,
                mixed_embedding=self._src_embedding_for_mix,
                target_candidate_found=False,
                mode="source_only",
                target_embedding_source="none",
            )
            return self._src_face
        target_embedding_source = "detector_candidate"
        target_candidate = self._select_target_face_candidate(roi_bgr_u8, target_face_meta)
        target_embedding = self._face_embedding(target_candidate) if target_candidate is not None else None

        if target_embedding is None:
            target_embedding_source = "metadata_recognition"
            target_candidate = self._recognize_target_face_from_meta(roi_bgr_u8, target_face_meta)
            target_embedding = self._face_embedding(target_candidate) if target_candidate is not None else None

        if target_embedding is None:
            target_embedding_source = "failed"

        mixed_arr = self._mix_embeddings(self._src_embedding_for_mix, target_embedding, self.face_swapper_weight)
        self._set_last_embedding_metrics(
            source_embedding=self._src_embedding_for_mix,
            target_embedding=target_embedding,
            mixed_embedding=mixed_arr,
            target_candidate_found=target_candidate is not None,
            mode="mixed" if target_embedding is not None else "target_lookup_failed",
            target_embedding_source=target_embedding_source,
        )
        mixed = mixed_arr.reshape(-1)
        try:
            source_face = copy.copy(self._src_face)
            setattr(source_face, "normed_embedding", mixed)
            setattr(source_face, "embedding", mixed)
            return source_face
        except Exception:
            return SimpleNamespace(normed_embedding=mixed, embedding=mixed)

    def swap(self, roi_bgr_u8: np.ndarray, target_face_meta: FaceMetadata) -> Optional[np.ndarray]:
        if roi_bgr_u8 is None:
            return None
        target_face = self._target_face_from_meta(target_face_meta)
        source_face = self._source_face_for_swap(roi_bgr_u8, target_face_meta)
        out = self._swapper.get(roi_bgr_u8, target_face, source_face, paste_back=True)
        return out


__all__ = ["InSwapperWorker"]