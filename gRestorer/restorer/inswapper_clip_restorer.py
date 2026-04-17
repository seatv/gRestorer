from __future__ import annotations

import torch

from gRestorer.restorer.base_face_swap_clip_restorer import BaseFaceSwapClipRestorer
from gRestorer.restorer.inswapper_worker import InSwapperWorker


class InSwapperClipRestorer(BaseFaceSwapClipRestorer):
    """Face-swap clip restorer using InsightFace InSwapper as the concrete worker."""

    def _build_worker(self):
        return InSwapperWorker(
            device=self.device,
            source_face_path=self.source_face_path,
            swap_model_path=self.swap_model_path,
            provider=self.provider,
        )


__all__ = ["InSwapperClipRestorer"]
