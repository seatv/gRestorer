from __future__ import annotations

from typing import List

import torch

from gRestorer.core.scene import Clip
from gRestorer.restorer.clip_restorer import BaseClipRestorer
from gRestorer.models.basicvsrpp.inference import load_model


class BasicVSRPPClipRestorer(BaseClipRestorer):
    """
    Robust BasicVSR++ clip restorer.

    Accepts Clip.frames as:
      - HWC uint8 [0..255], or
      - HWC float [0..1], or
      - HWC float [0..255] (we'll auto-normalize)

    Produces restored frames as HWC float in [0..1] on device.
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
        self.fp16 = bool(fp16) and (device.type == "cuda")  # keep conservative
        self.model = load_model(config, self.checkpoint_path, device=self.device, fp16=self.fp16)
        self.model.eval()

    @staticmethod
    def _to_float01_hwc(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            return x.to(dtype=torch.float32) / 255.0
        xf = x.to(dtype=torch.float32)
        # If values look like 0..255, normalize.
        # (We treat >1.5 as "probably 0..255".)
        try:
            vmax = float(xf.max().item())
        except Exception:
            vmax = 0.0
        if vmax > 1.5:
            xf = xf / 255.0
        return xf

    @torch.inference_mode()
    def restore_clip(self, clip: Clip) -> List[torch.Tensor]:
        frames = clip.frames
        if not frames:
            return []

        # Normalize to float [0..1] HWC
        frames01 = [self._to_float01_hwc(f) for f in frames]

        # Stack to BTCHW (BGR order preserved)
        tchw = torch.stack([f.permute(2, 0, 1).contiguous() for f in frames01], dim=0)  # TCHW
        btchw = tchw.unsqueeze(0)  # 1,T,C,H,W

        btchw = btchw.to(dtype=torch.float16 if self.fp16 else torch.float32)

        out = self.model(inputs=btchw)          # BTCHW
        out_tchw = out.squeeze(0)               # TCHW

        out_frames: List[torch.Tensor] = []
        for x in out_tchw:
            hwc = x.permute(1, 2, 0).contiguous()
            # Keep it sane
            out_frames.append(hwc.clamp(0.0, 1.0))
        return out_frames


__all__ = ["BasicVSRPPClipRestorer"]
