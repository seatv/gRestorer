from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import os
import torch

from .yolo import MosaicDetectionModel, FrameDetections


@dataclass
class FaceMetadata:
    """Detector-owned face metadata aligned with one detection box.

    Coordinates are in the same space as Detection.boxes:
      - bbox_xyxy: full-frame xyxy float coordinates
      - kps:       optional landmark array in full-frame coordinates, shape [K,2]
    """
    bbox_xyxy: Tuple[float, float, float, float]
    kps: Optional[torch.Tensor] = None
    det_score: Optional[float] = None


@dataclass
class Detection:
    """
    Public detection object used by the pipeline.

    boxes:      [N,4] float32 xyxy in original coordinates, or None
    scores:     [N]   float32, or None
    classes:    [N]   int64,   or None
    masks:      [N,H,W] uint8, or None
    face_metas: list[FaceMetadata] aligned 1:1 with boxes, or None
    """
    boxes: Optional[torch.Tensor]
    scores: Optional[torch.Tensor]
    classes: Optional[torch.Tensor]
    masks: Optional[torch.Tensor]
    face_metas: Optional[List[Optional[FaceMetadata]]] = None


class Detector:
    """
    High-level detector wrapper used by the pipeline.

    IMPORTANT TODO (later): the YOLO wrapper currently does a CPU hop for
    letterbox/preprocess. We'll fix that in a later optimization step.

    Input frames:
      - List of BGR frames, either:
          * uint8 [H,W,3]  (preferred for baseline)
          * float16/float32 [H,W,3] in [0,1]
        on any device.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        imgsz: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[Sequence[int]] = None,
        fp16: bool = True,
    ) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Detector model not found: {model_path}")

        self.model = MosaicDetectionModel(
            model_path=model_path,
            device=device,
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_thres,
            classes=classes,
            fp16=fp16,
        )

    def detect_batch(self, frames: List[torch.Tensor]) -> List[Detection]:
        if not frames:
            return []

        frame_results: List[FrameDetections] = self.model.infer_batch(frames)

        detections: List[Detection] = []
        for fr in frame_results:
            detections.append(
                Detection(
                    boxes=fr.boxes_xyxy,
                    scores=fr.scores,
                    classes=fr.classes,
                    masks=fr.masks,
                    face_metas=None,
                )
            )
        return detections


# Keep the Lada-ish name available (your pipeline previously imported MosaicDetector)
MosaicDetector = Detector

__all__ = ["FaceMetadata", "Detection", "Detector", "MosaicDetector"]
