# gRestorer/restorer/simswap_clip_restorer.py

from __future__ import annotations

from gRestorer.restorer.base_face_swap_clip_restorer import BaseFaceSwapClipRestorer
from gRestorer.restorer.simswap_worker import SimSwapWorker


class SimSwapClipRestorer(BaseFaceSwapClipRestorer):
    """Face-swap clip restorer using native SimSwap paste-back."""

    def _build_worker(self):
        return SimSwapWorker(
            device=self.device,
            source_face_path=self.source_face_path,
            swap_model_path=self.swap_model_path,
            provider=self.provider,
        )


__all__ = ["SimSwapClipRestorer"]