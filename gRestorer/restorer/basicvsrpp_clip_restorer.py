from __future__ import annotations

from typing import List, Optional

import torch

from gRestorer.core.scene import Clip
from gRestorer.restorer.clip_restorer import BaseClipRestorer

from gRestorer.models.basicvsrpp.inference import load_model


class BasicVSRPPClipRestorer(BaseClipRestorer):
    """LADA-faithful BasicVSR++ clip restorer.

    - Operates on Clip.frames (HWC float32 in [0,1], BGR order) already on device.
    - Produces restored clip frames (HWC float32 in [0,1], BGR order) on device.
    - No CPU round-trips inside the hot path.
    """

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        *,
        fp16: bool = True,
        config: str | dict | None = None,
    ) -> None:
        super().__init__(device=device)
        self.checkpoint_path = str(checkpoint_path)
        self.fp16 = bool(fp16) and device.type == "cuda"

        # Build + load model (once).
        # Note: mmengine loads ckpt on CPU then moves to device; that cost is paid once.
        self.model = load_model(config, self.checkpoint_path, device=self.device, fp16=self.fp16)
        self.model.eval()

    @torch.inference_mode()
    def restore_clip(self, clip: Clip) -> List[torch.Tensor]:
        # Clip.frames: list[T] of HWC float in [0,1] on self.device.
        frames = clip.frames
        if not frames:
            return []

        # Stack to BTCHW without changing channel order (BGR is kept as-is, matching LADA bgr2rgb=False path).
        tchw = torch.stack([f.permute(2, 0, 1).contiguous() for f in frames], dim=0)  # TCHW
        btchw = tchw.unsqueeze(0)  # 1,T,C,H,W

        if self.fp16:
            btchw = btchw.to(dtype=torch.float16)
        else:
            btchw = btchw.to(dtype=torch.float32)

        out = self.model(inputs=btchw)  # -> BTCHW
        out_tchw = out.squeeze(0)       # -> TCHW

        # Back to list of HWC floats in [0,1]
        out_frames: List[torch.Tensor] = [x.permute(1, 2, 0).contiguous() for x in out_tchw]
        return out_frames


__all__ = ["BasicVSRPPClipRestorer"]
