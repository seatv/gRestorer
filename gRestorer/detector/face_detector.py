from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from gRestorer.detector.core import Detection, FaceMetadata


@dataclass
class _FaceDetections:
    boxes_xyxy: Optional[torch.Tensor]
    scores: Optional[torch.Tensor]
    classes: Optional[torch.Tensor]
    masks: Optional[torch.Tensor]
    face_metas: Optional[List[Optional[FaceMetadata]]]


class FaceDetector:
    """
    InsightFace/SCRFD-based face detector wrapper for the gRestorer pipeline.

    Important geometry contract:
      - Detection.boxes = expanded ROI boxes used by tracker/cropper
      - FaceMetadata.bbox_xyxy = original tight face box from detector
      - FaceMetadata.kps = original tight landmark coordinates from detector

    The tracker follows the expanded ROI, but the swapper must receive the
    tight face geometry.
    """

    def __init__(
        self,
        model_path: str,
        device: str | torch.device = "cuda:0",
        imgsz: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[Sequence[int]] = None,
        fp16: bool = True,
        suppress_onnx_warnings: bool = False,
        angles: Optional[Sequence[int]] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.imgsz = int(imgsz)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.classes = classes
        self.fp16 = bool(fp16)
        self.suppress_onnx_warnings = bool(suppress_onnx_warnings)
        self.angles = self._normalize_angles(angles)

        # ROI expansion factors used only for tracker/crop boxes.
        self.top_expand = 0.05
        self.bottom_expand = 0.10
        self.side_expand = 0.15

        if self.suppress_onnx_warnings:
            try:
                import onnxruntime as ort
                ort.set_default_logger_severity(3)  # errors only
            except Exception:
                pass

        try:
            from insightface.model_zoo import get_model
        except Exception as e:
            raise ImportError(
                "Face detector backend requires `insightface`. "
                "Install it in the gRestorer environment before using --det-type face."
            ) from e

        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        if self.device.type == "cuda":
            ctx_id = 0 if self.device.index is None else int(self.device.index)
        else:
            ctx_id = -1

        self.model = get_model(self.model_path)
        self.model.prepare(ctx_id=ctx_id)

        print(
            f"[FaceDetector] provider_device={self.device} imgsz={self.imgsz} "
            f"conf={self.conf_thres:.3f} iou={self.iou_thres:.3f} "
            f"angles={','.join(str(a) for a in self.angles)}"
        )

    @staticmethod
    def _normalize_angles(angles: Optional[Sequence[int]]) -> Tuple[int, ...]:
        if angles is None:
            return (0,)
        out: list[int] = []
        for raw in angles:
            try:
                a = int(raw) % 360
            except Exception:
                continue
            if a not in (0, 90, 180, 270):
                raise ValueError(f"Unsupported face detector angle: {raw!r}. Allowed: 0, 90, 180, 270")
            if a not in out:
                out.append(a)
        return tuple(out or [0])

    @staticmethod
    def _rotate_image_for_angle(img: np.ndarray, angle: int) -> np.ndarray:
        a = int(angle) % 360
        if a == 0:
            return np.ascontiguousarray(img)
        if a == 90:
            return np.ascontiguousarray(np.rot90(img, k=3))
        if a == 180:
            return np.ascontiguousarray(np.rot90(img, k=2))
        if a == 270:
            return np.ascontiguousarray(np.rot90(img, k=1))
        raise ValueError(f"Unsupported angle: {angle}")

    @staticmethod
    def _map_points_from_rotated(points_xy: np.ndarray, *, angle: int, orig_w: int, orig_h: int) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float32).copy()
        a = int(angle) % 360
        if a == 0:
            return pts
        xr = pts[..., 0].copy()
        yr = pts[..., 1].copy()
        if a == 90:
            pts[..., 0] = yr
            pts[..., 1] = float(orig_h - 1) - xr
        elif a == 180:
            pts[..., 0] = float(orig_w - 1) - xr
            pts[..., 1] = float(orig_h - 1) - yr
        elif a == 270:
            pts[..., 0] = float(orig_w - 1) - yr
            pts[..., 1] = xr
        else:
            raise ValueError(f"Unsupported angle: {angle}")
        pts[..., 0] = np.clip(pts[..., 0], 0.0, max(0.0, float(orig_w - 1)))
        pts[..., 1] = np.clip(pts[..., 1], 0.0, max(0.0, float(orig_h - 1)))
        return pts.astype(np.float32, copy=False)

    @classmethod
    def _map_boxes_from_rotated(cls, boxes_xyxy: np.ndarray, *, angle: int, orig_w: int, orig_h: int) -> np.ndarray:
        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
        if boxes.size == 0:
            return boxes.reshape(0, 4).astype(np.float32)
        corners = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1)
        mapped = cls._map_points_from_rotated(corners, angle=angle, orig_w=orig_w, orig_h=orig_h)
        x1 = mapped[..., 0].min(axis=1)
        y1 = mapped[..., 1].min(axis=1)
        x2 = mapped[..., 0].max(axis=1)
        y2 = mapped[..., 1].max(axis=1)
        return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32, copy=False)

    @staticmethod
    def _nms_indices(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if boxes.size == 0:
            return np.empty((0,), dtype=np.int64)
        if float(iou_threshold) <= 0.0:
            return np.argsort(-scores).astype(np.int64)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = np.argsort(-scores)
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            union = areas[i] + areas[rest] - inter
            iou = inter / np.maximum(union, 1e-6)
            order = rest[iou <= float(iou_threshold)]
        return np.asarray(keep, dtype=np.int64)

    @staticmethod
    def _to_numpy_bgr_u8(frame: torch.Tensor) -> np.ndarray:
        if not isinstance(frame, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(frame)!r}")

        x = frame.detach()
        if x.device.type != "cpu":
            x = x.cpu()

        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected HWC with 3 channels, got shape={tuple(x.shape)}")

        if x.dtype == torch.uint8:
            return x.contiguous().numpy()

        x = x.to(torch.float32)
        if float(x.max()) <= 1.5:
            x = x * 255.0

        x = x.round().clamp(0.0, 255.0).to(torch.uint8)
        return x.contiguous().numpy()

    @staticmethod
    def _empty_detection() -> _FaceDetections:
        return _FaceDetections(
            boxes_xyxy=torch.empty((0, 4), dtype=torch.float32),
            scores=torch.empty((0,), dtype=torch.float32),
            classes=torch.empty((0,), dtype=torch.int64),
            masks=None,
            face_metas=[],
        )

    def _expand_and_clip_boxes(
        self,
        boxes_xyxy: np.ndarray,
        *,
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        if boxes_xyxy.size == 0:
            return boxes_xyxy

        out = boxes_xyxy.astype(np.float32, copy=True)

        widths = out[:, 2] - out[:, 0]
        heights = out[:, 3] - out[:, 1]

        dx = widths * self.side_expand
        dy_top = heights * self.top_expand
        dy_bottom = heights * self.bottom_expand

        out[:, 0] = np.clip(out[:, 0] - dx, 0.0, max(0.0, float(frame_w - 1)))
        out[:, 1] = np.clip(out[:, 1] - dy_top, 0.0, max(0.0, float(frame_h - 1)))
        out[:, 2] = np.clip(out[:, 2] + dx, 0.0, max(0.0, float(frame_w - 1)))
        out[:, 3] = np.clip(out[:, 3] + dy_bottom, 0.0, max(0.0, float(frame_h - 1)))

        out[:, 2] = np.maximum(out[:, 2], out[:, 0])
        out[:, 3] = np.maximum(out[:, 3], out[:, 1])
        return out

    @staticmethod
    def _build_face_metas(
        tight_boxes_xyxy: np.ndarray,
        scores_np: np.ndarray,
        kpss: Optional[np.ndarray],
    ) -> List[Optional[FaceMetadata]]:
        metas: List[Optional[FaceMetadata]] = []
        for i in range(tight_boxes_xyxy.shape[0]):
            bbox = tight_boxes_xyxy[i]
            kps_t: Optional[torch.Tensor] = None
            if kpss is not None and i < len(kpss) and kpss[i] is not None:
                kps_arr = np.asarray(kpss[i], dtype=np.float32)
                if kps_arr.ndim == 2 and kps_arr.shape[-1] == 2:
                    kps_t = torch.from_numpy(kps_arr.copy()).to(dtype=torch.float32)
            metas.append(
                FaceMetadata(
                    bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    kps=kps_t,
                    det_score=float(scores_np[i]),
                )
            )
        return metas

    def _detect_at_angle(self, img: np.ndarray, angle: int) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        frame_h, frame_w = img.shape[:2]
        work = self._rotate_image_for_angle(img, angle)
        bboxes, kpss = self.model.detect(work, input_size=(self.imgsz, self.imgsz), max_num=0)
        if bboxes is None or len(bboxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), None
        bboxes = np.asarray(bboxes, dtype=np.float32)
        scores = bboxes[:, 4].astype(np.float32, copy=False)
        keep = scores >= self.conf_thres
        if not np.any(keep):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), None
        boxes_rot = bboxes[keep, :4].astype(np.float32, copy=False)
        scores = scores[keep].astype(np.float32, copy=False)
        boxes = self._map_boxes_from_rotated(boxes_rot, angle=angle, orig_w=frame_w, orig_h=frame_h)
        kpss_out = None
        if kpss is not None:
            kpss_arr = np.asarray(kpss, dtype=np.float32)[keep]
            if kpss_arr.ndim == 3 and kpss_arr.shape[-1] == 2:
                kpss_out = self._map_points_from_rotated(kpss_arr, angle=angle, orig_w=frame_w, orig_h=frame_h)
        return boxes, scores, kpss_out

    def _detect_one(self, frame: torch.Tensor) -> _FaceDetections:
        img = self._to_numpy_bgr_u8(frame)
        frame_h, frame_w = img.shape[:2]
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_kps: list[np.ndarray] = []
        have_any_kps = False
        for angle in self.angles:
            boxes_a, scores_a, kps_a = self._detect_at_angle(img, int(angle))
            if boxes_a.size == 0:
                continue
            all_boxes.append(boxes_a)
            all_scores.append(scores_a)
            if kps_a is not None:
                all_kps.append(kps_a)
                have_any_kps = True
            else:
                all_kps.append(np.full((boxes_a.shape[0], 5, 2), np.nan, dtype=np.float32))
        if not all_boxes:
            return self._empty_detection()
        tight_boxes_np = np.concatenate(all_boxes, axis=0).astype(np.float32, copy=False)
        scores_np = np.concatenate(all_scores, axis=0).astype(np.float32, copy=False)
        kpss_np = np.concatenate(all_kps, axis=0).astype(np.float32, copy=False) if have_any_kps else None
        keep_idx = self._nms_indices(tight_boxes_np, scores_np, self.iou_thres)
        if keep_idx.size == 0:
            return self._empty_detection()
        tight_boxes_np = tight_boxes_np[keep_idx]
        scores_np = scores_np[keep_idx]
        if kpss_np is not None:
            kpss_np = kpss_np[keep_idx]
            if not np.isfinite(kpss_np).any():
                kpss_np = None
        roi_boxes_np = self._expand_and_clip_boxes(tight_boxes_np, frame_w=frame_w, frame_h=frame_h)
        boxes_xyxy = torch.from_numpy(roi_boxes_np).to(dtype=torch.float32)
        scores = torch.from_numpy(scores_np).to(dtype=torch.float32)
        classes = torch.zeros((boxes_xyxy.shape[0],), dtype=torch.int64)
        face_metas = self._build_face_metas(tight_boxes_np, scores_np, kpss_np)
        return _FaceDetections(boxes_xyxy=boxes_xyxy, scores=scores, classes=classes, masks=None, face_metas=face_metas)

    def detect_batch(self, frames: List[torch.Tensor]) -> List[Detection]:
        if not frames:
            return []

        detections: List[Detection] = []
        for frame in frames:
            fr = self._detect_one(frame)
            detections.append(
                Detection(
                    boxes=fr.boxes_xyxy,
                    scores=fr.scores,
                    classes=fr.classes,
                    masks=fr.masks,
                    face_metas=fr.face_metas,
                )
            )
        return detections


__all__ = ["FaceDetector"]
