from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

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
    ) -> None:
        self.model_path = str(model_path)
        self.imgsz = int(imgsz)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.classes = classes
        self.fp16 = bool(fp16)

        # ROI expansion factors used only for tracker/crop boxes.
        self.top_expand = 0.05
        self.bottom_expand = 0.10
        self.side_expand = 0.15

        try:
            import onnxruntime as ort
            ort.set_default_logger_severity(3)  # suppress warning spam from some RetinaFace ONNX exports
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

    def _detect_one(self, frame: torch.Tensor) -> _FaceDetections:
        img = self._to_numpy_bgr_u8(frame)
        frame_h, frame_w = img.shape[:2]

        bboxes, kpss = self.model.detect(
            img,
            input_size=(self.imgsz, self.imgsz),
            max_num=0,
        )

        if bboxes is None or len(bboxes) == 0:
            return self._empty_detection()

        bboxes = np.asarray(bboxes, dtype=np.float32)
        scores_np = bboxes[:, 4]

        keep = scores_np >= self.conf_thres
        if not np.any(keep):
            return self._empty_detection()

        tight_boxes_np = bboxes[keep, :4]
        scores_np = scores_np[keep]
        kpss_np = None
        if kpss is not None:
            kpss_np = np.asarray(kpss, dtype=np.float32)[keep]

        roi_boxes_np = self._expand_and_clip_boxes(
            tight_boxes_np,
            frame_w=frame_w,
            frame_h=frame_h,
        )

        boxes_xyxy = torch.from_numpy(roi_boxes_np).to(dtype=torch.float32)
        scores = torch.from_numpy(scores_np).to(dtype=torch.float32)
        classes = torch.zeros((boxes_xyxy.shape[0],), dtype=torch.int64)
        face_metas = self._build_face_metas(tight_boxes_np, scores_np, kpss_np)

        return _FaceDetections(
            boxes_xyxy=boxes_xyxy,
            scores=scores,
            classes=classes,
            masks=None,
            face_metas=face_metas,
        )

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
