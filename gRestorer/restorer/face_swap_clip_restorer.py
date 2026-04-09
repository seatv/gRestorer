from __future__ import annotations

from typing import List
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from gRestorer.core.scene import Clip
from gRestorer.restorer.clip_restorer import BaseClipRestorer


class FaceSwapClipRestorer(BaseClipRestorer):
    """
    ROI-authoritative face-swap restorer.

    Key behavior:
      - The pipeline-selected ROI clip is authoritative.
      - FaceAnalysis is only an alignment helper.
      - We do a clip-level prepass so target selection is stable.
      - We anchor one target face for the whole clip.
      - We forward-fill AND backward-fill target-face state so early frames
        are not left unswapped just because the first successful detection
        happens late in the clip.
    """

    def __init__(
        self,
        device: torch.device,
        source_face_path: str,
        swap_model_path: str,
        *,
        swap_input_size: int = 128,
        provider: str = "auto",
    ) -> None:
        super().__init__(device=device)

        self.source_face_path = str(source_face_path)
        self.swap_model_path = str(swap_model_path)
        self.swap_input_size = int(swap_input_size)
        self.provider = str(provider or "auto").lower()

        self.debug_enabled = str(os.getenv("GR_FS_DEBUG", "0")).strip().lower() not in ("", "0", "false", "no")
        self.debug_dir = Path(os.getenv("GR_FS_DEBUG_DIR", "fs_debug"))
        self.debug_start = int(os.getenv("GR_FS_DEBUG_START", "-1"))
        self.debug_end = int(os.getenv("GR_FS_DEBUG_END", "-1"))
        if self.debug_enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        try:
            from insightface.app import FaceAnalysis
            from insightface.model_zoo import get_model
        except Exception as e:
            raise ImportError(
                "FaceSwapClipRestorer requires `insightface` in the gRestorer environment."
            ) from e

        providers = self._providers_for(self.provider, self.device)

        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        ctx_id = 0 if self.device.type == "cuda" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        self.swapper = get_model(self.swap_model_path, providers=providers)

        src = cv2.imread(self.source_face_path, cv2.IMREAD_COLOR)
        if src is None:
            raise FileNotFoundError(f"Failed to read source face image: {self.source_face_path}")

        src_faces = self.app.get(src)
        if not src_faces:
            raise RuntimeError(f"No face detected in source image: {self.source_face_path}")

        self.src_face = max(
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
    def _tensor_hwc_float_to_numpy_bgr_u8(x: torch.Tensor) -> np.ndarray:
        y = x.detach()
        if y.device.type != "cpu":
            y = y.cpu()
        y = y.to(torch.float32).clamp(0.0, 1.0)
        y = (y * 255.0).round().to(torch.uint8).contiguous()
        return np.ascontiguousarray(y.numpy())

    def _numpy_bgr_u8_to_tensor_hwc_float(self, x: np.ndarray) -> torch.Tensor:
        y = torch.from_numpy(np.ascontiguousarray(x)).to(torch.float32) / 255.0
        if self.device.type != "cpu":
            y = y.to(self.device, non_blocking=True)
        return y.contiguous()

    @staticmethod
    def _unpad_hwc_numpy(x: np.ndarray, pad: tuple[int, int, int, int]) -> np.ndarray:
        pt, pb, pl, pr = [int(v) for v in pad]
        h, w = int(x.shape[0]), int(x.shape[1])
        y0 = pt
        y1 = h - pb if pb > 0 else h
        x0 = pl
        x1 = w - pr if pr > 0 else w
        return np.ascontiguousarray(x[y0:y1, x0:x1, :])

    @staticmethod
    def _pad_hwc_numpy(x: np.ndarray, pad: tuple[int, int, int, int], clip_size: int) -> np.ndarray:
        pt, pb, pl, pr = [int(v) for v in pad]
        out = np.zeros((int(clip_size), int(clip_size), 3), dtype=x.dtype)
        h, w = int(x.shape[0]), int(x.shape[1])
        out[pt:pt + h, pl:pl + w, :] = x
        return out

    def _debug_this_frame(self, frame_num: int) -> bool:
        if not self.debug_enabled:
            return False
        if self.debug_start >= 0 and frame_num < self.debug_start:
            return False
        if self.debug_end >= 0 and frame_num > self.debug_end:
            return False
        return True

    def _save_debug_image(self, frame_num: int, name: str, img: np.ndarray) -> None:
        if not self._debug_this_frame(frame_num):
            return
        out = self.debug_dir / f"f{frame_num:06d}_{name}.png"
        cv2.imwrite(str(out), img)

    def _save_debug_text(self, frame_num: int, text: str) -> None:
        if not self._debug_this_frame(frame_num):
            return
        out = self.debug_dir / f"f{frame_num:06d}.txt"
        with open(out, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    @staticmethod
    def _draw_faces(img: np.ndarray, faces: list) -> np.ndarray:
        vis = img.copy()
        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, str(idx), (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if getattr(face, "kps", None) is not None:
                for p in face.kps:
                    px, py = int(p[0]), int(p[1])
                    cv2.circle(vis, (px, py), 2, (0, 0, 255), -1)
        return vis

    @staticmethod
    def _diff_image(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return cv2.absdiff(a, b)

    @staticmethod
    def _bbox_to_tuple(face) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(v) for v in face.bbox]
        return x1, y1, x2, y2

    @staticmethod
    def _face_area(face) -> float:
        x1, y1, x2, y2 = FaceSwapClipRestorer._bbox_to_tuple(face)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0

        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = a_area + b_area - inter
        if denom <= 0.0:
            return 0.0
        return inter / denom

    @staticmethod
    def _bbox_center_distance_sq(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        acx = 0.5 * (ax1 + ax2)
        acy = 0.5 * (ay1 + ay2)
        bcx = 0.5 * (bx1 + bx2)
        bcy = 0.5 * (by1 + by2)
        dx = acx - bcx
        dy = acy - bcy
        return dx * dx + dy * dy

    @staticmethod
    def _bbox_center_distance_sq_to_crop_center(face, crop_shape: tuple[int, int, int]) -> float:
        h, w = int(crop_shape[0]), int(crop_shape[1])
        cx = 0.5 * w
        cy = 0.5 * h
        x1, y1, x2, y2 = FaceSwapClipRestorer._bbox_to_tuple(face)
        fx = 0.5 * (x1 + x2)
        fy = 0.5 * (y1 + y2)
        dx = fx - cx
        dy = fy - cy
        return dx * dx + dy * dy

    def _pick_anchor(self, all_faces: list[list], crops: list[np.ndarray]):
        """
        Pick ONE target face for the whole clip.

        We prefer:
          - face nearest crop center (ROI-authoritative)
          - larger face as tiebreaker
          - earliest frame as final tiebreaker
        """
        best = None
        best_key = None
        for i, faces in enumerate(all_faces):
            if not faces:
                continue
            for face in faces:
                key = (
                    -self._bbox_center_distance_sq_to_crop_center(face, crops[i].shape),
                    self._face_area(face),
                    -i,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = (i, face)
        return best

    def _pick_matching_face(self, faces: list, ref_bbox: tuple[float, float, float, float]):
        if not faces:
            return None

        def score(face):
            bbox = self._bbox_to_tuple(face)
            iou = self._bbox_iou(bbox, ref_bbox)
            dist = self._bbox_center_distance_sq(bbox, ref_bbox)
            area = self._face_area(face)
            return (iou, -dist, area)

        return max(faces, key=score)

    @torch.inference_mode()
    def restore_clip(self, clip: Clip) -> List[torch.Tensor]:
        out_frames: List[torch.Tensor] = []

        # Phase 1: reconstruct all crops first
        crops: list[np.ndarray] = []
        crop_resized_shapes: list[tuple[int, int]] = []
        pads: list[tuple[int, int, int, int]] = []
        clip_sizes: list[int] = []
        frame_nums: list[int] = []

        for i, clip_frame in enumerate(clip.frames):
            frame_num = int(clip.frame_nums[i])
            clip_np = self._tensor_hwc_float_to_numpy_bgr_u8(clip_frame)
            crop_h, crop_w = clip.crop_shapes[i]
            pad = clip.pad_after_resizes[i]
            clip_size = int(clip.clip_size)

            crop_resized_np = self._unpad_hwc_numpy(clip_np, pad)

            if int(crop_resized_np.shape[0]) != int(crop_h) or int(crop_resized_np.shape[1]) != int(crop_w):
                crop_np = cv2.resize(
                    crop_resized_np,
                    (int(crop_w), int(crop_h)),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                crop_np = crop_resized_np

            crops.append(crop_np)
            crop_resized_shapes.append((int(crop_resized_np.shape[0]), int(crop_resized_np.shape[1])))
            pads.append(pad)
            clip_sizes.append(clip_size)
            frame_nums.append(frame_num)

            self._save_debug_text(
                frame_num,
                f"frame={frame_num} clip_id={clip.id} idx={i} crop_shape={crop_np.shape} clip_size={clip_size} pad={pad}",
            )
            self._save_debug_image(frame_num, "01_crop", crop_np)

        # Phase 2: detect all frames once
        all_faces: list[list] = []
        for crop_np, frame_num in zip(crops, frame_nums):
            faces = self.app.get(crop_np)
            all_faces.append(faces)
            self._save_debug_text(frame_num, f"faces_detected={len(faces)}")
            if faces:
                self._save_debug_image(frame_num, "02_faces", self._draw_faces(crop_np, faces))

        # Phase 3: choose one anchor for the whole clip
        anchor = self._pick_anchor(all_faces, crops)

        selected_faces: list = [None] * len(crops)

        if anchor is not None:
            anchor_idx, anchor_face = anchor
            anchor_bbox = self._bbox_to_tuple(anchor_face)
            selected_faces[anchor_idx] = anchor_face
            self._save_debug_text(frame_nums[anchor_idx], "target_face_source=anchor")

            # Forward fill from anchor
            ref_bbox = anchor_bbox
            for i in range(anchor_idx + 1, len(crops)):
                face = self._pick_matching_face(all_faces[i], ref_bbox)
                if face is None:
                    face = selected_faces[i - 1]
                selected_faces[i] = face
                if face is not None:
                    ref_bbox = self._bbox_to_tuple(face)

            # Backward fill from anchor
            ref_bbox = anchor_bbox
            for i in range(anchor_idx - 1, -1, -1):
                face = self._pick_matching_face(all_faces[i], ref_bbox)
                if face is None:
                    face = selected_faces[i + 1]
                selected_faces[i] = face
                if face is not None:
                    ref_bbox = self._bbox_to_tuple(face)

        # Phase 4: swap every frame using the selected face for that frame
        for i, crop_np in enumerate(crops):
            frame_num = frame_nums[i]
            face = selected_faces[i]

            swapped_np = crop_np
            if face is not None:
                x1, y1, x2, y2 = self._bbox_to_tuple(face)
                self._save_debug_text(frame_num, f"target_face_bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

                before = crop_np.copy()
                try:
                    out = self.swapper.get(crop_np, face, self.src_face, paste_back=True)
                except Exception as e:
                    self._save_debug_text(frame_num, f"swap_exception={e!r}")
                    out = None

                if out is None:
                    self._save_debug_text(frame_num, "swap_returned=None")
                else:
                    mad = float(np.mean(np.abs(out.astype(np.int16) - before.astype(np.int16))))
                    self._save_debug_text(frame_num, f"mean_abs_diff={mad:.4f}")
                    self._save_debug_image(frame_num, "03_swap", out)
                    self._save_debug_image(frame_num, "04_diff", self._diff_image(before, out))
                    swapped_np = out
            else:
                self._save_debug_text(frame_num, "target_face_source=none")

            target_h, target_w = crop_resized_shapes[i]
            if int(swapped_np.shape[0]) != target_h or int(swapped_np.shape[1]) != target_w:
                swapped_resized_np = cv2.resize(
                    swapped_np,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                swapped_resized_np = swapped_np

            swapped_clip_np = self._pad_hwc_numpy(swapped_resized_np, pads[i], clip_sizes[i])
            self._save_debug_image(frame_num, "05_clip_return", swapped_clip_np)

            out_frames.append(self._numpy_bgr_u8_to_tensor_hwc_float(swapped_clip_np))

        return out_frames


__all__ = ["FaceSwapClipRestorer"]
