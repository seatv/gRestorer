# gRestorer/detector/yolo.py
# Lada-style YOLO segmentation detector with proper letterbox/unpad handling.


# TODO(GPU): Current preprocess path forces CPU:
#  - frames are converted to CPU uint8
#  - letterbox uses NumPy
#  - outputs (boxes/masks) are CPU tensors
# Later: implement torch-only letterbox on GPU + keep outputs on GPU to remove the CPU hop.


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.results import Results
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, nms, ops
from ultralytics.utils.checks import check_imgsz
from ultralytics.data.augment import LetterBox


@dataclass
class FrameDetections:
    """
    Per-frame detection results from the mosaic detector.

    All tensors are on CPU.

    boxes_xyxy: [N, 4] float32 (x1, y1, x2, y2) in original image coordinates.
    scores:     [N]     float32 confidence scores.
    classes:    [N]     int64 class indices.
    masks:      [N, H, W] uint8 binary masks in original HxW (0 or 255).
    """
    boxes_xyxy: Optional[torch.Tensor]
    scores: Optional[torch.Tensor]
    classes: Optional[torch.Tensor]
    masks: Optional[torch.Tensor]


def _scale_and_unpad_image(masks: torch.Tensor, im0_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Port of Lada's scale_and_unpad_image (torch version).

    masks: [H1, W1, C] uint8/float tensor in letterboxed coordinates
    im0_shape: original image shape (H0, W0, C0)
    """
    h0, w0 = im0_shape[:2]
    h1, w1, _ = masks.shape

    if h1 == h0 and w1 == w0:
        return masks

    g = min(h1 / h0, w1 / w0)
    pw, ph = (w1 - w0 * g) / 2, (h1 - h0 * g) / 2

    # same rounding trick as Lada
    t = round(ph - 0.1)
    l = round(pw - 0.1)
    b = h1 - round(ph + 0.1)
    r = w1 - round(pw + 0.1)

    # crop out the letterboxed region
    x = masks[t:b, l:r].permute(2, 0, 1).unsqueeze(0).float()
    y = F.interpolate(x, size=(h0, w0), mode="bilinear", align_corners=False)
    return (
        y.squeeze(0)
        .permute(1, 2, 0)
        .round_()
        .clamp_(0, 255)
        .to(masks.dtype)
    )


def _to_mask_img_tensor(masks: torch.Tensor, class_val: int = 0, pixel_val: int = 255) -> torch.Tensor:
    """
    Port of Lada's _to_mask_img_tensor for a single Ultralytics Masks.data tensor.

    masks: [1, H, W] or [H, W] tensor
    """
    if masks.ndim == 2:
        masks_tensor = masks
    else:
        # [1, H, W] -> [H, W]
        masks_tensor = masks[0]
    masks_tensor = torch.where(masks_tensor != class_val, pixel_val, 0).to(torch.uint8)
    return masks_tensor


def _convert_yolo_mask_tensor(yolo_mask, img_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Port of Lada's convert_yolo_mask_tensor (torch version),
    adapted to always return a [H, W, 1] uint8 tensor on CPU.

    yolo_mask: ultralytics.engine.results.Masks (single instance via __getitem__)
    img_shape: original image shape (H, W, C)
    """
    # yolo_mask.data is typically [1, h1, w1]
    mask_img = _to_mask_img_tensor(yolo_mask.data)
    if mask_img.ndim == 2:
        mask_img = mask_img.unsqueeze(-1)  # [H1, W1, 1]

    mask_img = _scale_and_unpad_image(mask_img, img_shape)
    mask_img = torch.where(mask_img > 127, 255, 0).to(torch.uint8)
    assert mask_img.ndim == 3 and mask_img.shape[2] == 1
    return mask_img  # [H, W, 1]


def _convert_yolo_box_xyxy(yolo_box, img_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """
    Lada-style box clipping, but we return XYXY instead of (t,l,b,r).

    yolo_box: an element of res.boxes (Ultralytics Boxes)
    img_shape: (H, W, C)
    """
    h, w = img_shape[0], img_shape[1]
    _box = yolo_box  # tensor [4]

    x1 = int(torch.clip(_box[0], 0, w).item())
    y1 = int(torch.clip(_box[1], 0, h).item())
    x2 = int(torch.clip(_box[2], 0, w).item())
    y2 = int(torch.clip(_box[3], 0, h).item())
    return x1, y1, x2, y2


class YoloSegmentationModel:
    """
    Port of Lada's Yolo11SegmentationModel, CPU-letterbox + AutoBackend.

    We always do letterboxing on CPU (like Lada's default VideoReader path),
    but run the network itself on the requested device (CUDA or CPU).
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        imgsz: int = 640,
        fp16: bool = False,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[Sequence[int]] = None,
        agnostic_nms: bool = False,
        max_det: int = 300,
    ) -> None:
        yolo_model = YOLO(model_path)
        assert yolo_model.task == "segment", f"Model '{model_path}' is not a segmentation model"

        self.stride = 32
        self.imgsz = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        self.letterbox = LetterBox(self.imgsz, auto=True, stride=self.stride)

        custom = {
            "conf": conf,
            "iou": iou,
            "classes": classes,
            "agnostic_nms": agnostic_nms,
            "max_det": max_det,
            "batch": 1,
            "save": False,
            "mode": "predict",
            "device": device,
            "half": fp16,
        }
        args = {**yolo_model.overrides, **custom}
        self.args = get_cfg(DEFAULT_CFG, args)

        self.device: torch.device = torch.device(device)
        self.model = AutoBackend(
            model=yolo_model.model,
            device=self.device,
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            fuse=True,
            verbose=False,
        )
        self.args.half = self.model.fp16
        self.model.eval()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        self.dtype = torch.float16 if fp16 else torch.float32

        print(
            f"[YOLO] Loaded seg model on {self.device}, "
            f"imgsz={self.imgsz}, conf={conf}, iou={iou}, fp16={fp16}"
        )

    def preprocess(self, imgs: List[torch.Tensor]) -> torch.Tensor:
        """
        imgs: list of [H, W, 3] uint8 CPU tensors, BGR order.

        Returns: [B, 3, H_lb, W_lb] uint8 letterboxed tensor on CPU.
        """
        # LetterBox expects numpy HWC uint8
        im = np.stack([self.letterbox(image=x.numpy()) for x in imgs])
        im = im.transpose((0, 3, 1, 2))  # BHWC -> BCHW
        im = np.ascontiguousarray(im)
        return torch.from_numpy(im)

    def inference_and_postprocess(
        self,
        imgs: torch.Tensor,
        orig_imgs: List[torch.Tensor],
    ) -> List[Results]:
        """
        imgs: [B, 3, H_lb, W_lb] uint8 CPU
        orig_imgs: list of [H, W, 3] uint8 CPU tensors (BGR)
        """
        with torch.inference_mode():
            x = imgs.to(device=self.device).to(dtype=self.dtype).div_(255.0)
            preds = self.model(x, augment=False, visualize=False, embed=None)
            return self._postprocess(preds, x, orig_imgs)

    def _postprocess(
        self,
        preds,
        img: torch.Tensor,
        orig_imgs: List[torch.Tensor],
    ) -> List[Results]:
        # protos: segmentation prototypes
        protos = preds[1][-1]

        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
        )

        return [
            self._construct_result(pred, img, orig_img, proto)
            for pred, orig_img, proto in zip(preds, orig_imgs, protos)
        ]

    def _construct_result(
        self,
        preds: torch.Tensor,
        img: torch.Tensor,
        orig_img: torch.Tensor,
        proto: torch.Tensor,
    ) -> Results:
        """
        preds: [N, 6 + num_masks] after NMS
        img:   network input (BCHW) tensor
        orig_img: [H, W, 3] uint8 CPU tensor (BGR)
        proto: prototypes tensor
        """
        if not len(preds):  # no detections
            masks = None
        else:
            # HWC masks in letterboxed coordinates
            masks = ops.process_mask(proto, preds[:, 6:], preds[:, :4], img.shape[2:], upsample=True)
            # scale boxes back to original image shape
            preds[:, :4] = ops.scale_boxes(img.shape[2:], preds[:, :4], orig_img.shape)

        if masks is not None:
            # Drop predictions whose masks are entirely zero
            keep = masks.sum((-2, -1)) > 0
            preds, masks = preds[keep], masks[keep]

        # Results will wrap boxes into ultralytics Boxes, masks into Masks, and keep orig_shape.
        return Results(
            orig_img,
            path="",
            names=self.model.names,
            boxes=preds[:, :6].cpu(),
            masks=masks,
        )


class MosaicDetectionModel:
    """
    gRestorer-facing detector that wraps the Lada-style YoloSegmentationModel.

    Contract:
    - Input frames: [H, W, 3] RGB float32 [0,1], typically on CUDA.
    - Output: FrameDetections with boxes/masks in original frame coordinates on CPU.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[Sequence[int]] = None,
        fp16: bool = True,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.conf = float(conf)
        self.iou = float(iou)
        self.classes = list(classes) if classes is not None else None
        self.fp16 = bool(fp16)

        print(f"[Detector] Initializing YOLO model...")
        self.seg = YoloSegmentationModel(
            model_path=model_path,
            device=device,
            imgsz=imgsz,
            fp16=fp16,
            conf=conf,
            iou=iou,
            classes=self.classes,
        )
        print("[Detector] Ready!")

    @staticmethod
    def _to_uint8_cpu(frame: torch.Tensor) -> torch.Tensor:
        """
        Convert [H, W, 3] BGR tensor (float [0,1] or uint8) on any device
        to a CPU uint8 BGR tensor.
        
        NOTE: Expects BGR input (no color conversion).
        """
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected [H, W, 3] frame, got shape={tuple(frame.shape)}")

        if frame.dtype in (torch.float16, torch.float32):
            f = frame.clamp(0.0, 1.0).mul(255.0).round()
            frame_u8 = f.to(torch.uint8)
        elif frame.dtype == torch.uint8:
            frame_u8 = frame
        else:
            raise TypeError(f"Unsupported frame dtype for detector: {frame.dtype}")

        # Move to CPU
        if frame_u8.device.type != "cpu":
            frame_u8 = frame_u8.cpu()

        return frame_u8

    @torch.no_grad()
    def infer_batch(self, frames: List[torch.Tensor]) -> List[FrameDetections]:
        """
        frames: list of [H, W, 3] BGR float32 [0,1] tensors (typically on CUDA).

        Returns: list of FrameDetections, one per frame.
        """
        if not frames:
            return []

        # Prepare CPU BGR uint8 inputs
        imgs_bgr: List[torch.Tensor] = []
        for f in frames:
            img_bgr = self._to_uint8_cpu(f)
            imgs_bgr.append(img_bgr)

        # Preprocess (letterbox) on CPU
        im_batch = self.seg.preprocess(imgs_bgr)

        # Run network on requested device
        results: List[Results] = self.seg.inference_and_postprocess(im_batch, imgs_bgr)

        outputs: List[FrameDetections] = []

        for idx, res in enumerate(results):
            # No detections for this frame
            if res.boxes is None or res.boxes.data.numel() == 0:
                outputs.append(
                    FrameDetections(
                        boxes_xyxy=None,
                        scores=None,
                        classes=None,
                        masks=None,
                    )
                )
                continue

            H0, W0 = res.orig_shape[:2]

            boxes_xyxy_list: List[Tuple[int, int, int, int]] = []
            scores_list: List[float] = []
            classes_list: List[int] = []
            masks_list: List[torch.Tensor] = []

            num_boxes = len(res.boxes)
            for i in range(num_boxes):
                # Box with Lada's clipping logic
                xyxy = _convert_yolo_box_xyxy(res.boxes[i].xyxy[0], (H0, W0, 3))
                boxes_xyxy_list.append(xyxy)

                # Conf / class from Boxes
                scores_list.append(float(res.boxes.conf[i].item()))
                classes_list.append(int(res.boxes.cls[i].item()))

                # Full-res mask via Lada's scale_and_unpad_image
                if res.masks is not None:
                    mask_img = _convert_yolo_mask_tensor(res.masks[i], (H0, W0, 3))
                    # [H, W, 1] -> [H, W]
                    masks_list.append(mask_img[..., 0])

            boxes_xyxy = torch.tensor(boxes_xyxy_list, dtype=torch.float32)
            scores = torch.tensor(scores_list, dtype=torch.float32)
            classes = torch.tensor(classes_list, dtype=torch.int64)

            masks_tensor: Optional[torch.Tensor]
            if masks_list:
                masks_tensor = torch.stack(masks_list, dim=0)  # [N, H, W] uint8
            else:
                masks_tensor = None

            outputs.append(
                FrameDetections(
                    boxes_xyxy=boxes_xyxy,
                    scores=scores,
                    classes=classes,
                    masks=masks_tensor,
                )
            )

        return outputs

# ---------------------------------------------------------------------------
# Compatibility wrapper (pipeline expects this symbol)
# ---------------------------------------------------------------------------

class YoloMosaicDetector:
    """
    Backwards/compat wrapper so older/newer pipeline variants can do:

        from gRestorer.detector.yolo import YoloMosaicDetector

    Internally we delegate to gRestorer.detector.core.Detector.
    Local import avoids circular-import issues (core.py imports yolo.py).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        debug: bool = False,
        imgsz: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes=None,
        fp16: bool = True,
        **kwargs,
    ) -> None:
        from .core import Detector  # local import to avoid import cycles

        self._impl = Detector(
            model_path=model_path,
            device=device,
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            fp16=fp16,
        )
        self.debug = debug  # kept only for signature compatibility

    def detect_batch(self, frames):
        return self._impl.detect_batch(frames)
