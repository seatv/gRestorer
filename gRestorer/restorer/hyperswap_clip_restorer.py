# gRestorer/restorer/hyperswap_clip_restorer.py

from __future__ import annotations

from gRestorer.restorer.base_face_swap_clip_restorer import BaseFaceSwapClipRestorer
from gRestorer.restorer.hyperswap_worker import HyperSwapWorker


class HyperSwapClipRestorer(BaseFaceSwapClipRestorer):
    """Face-swap clip restorer using native HyperSwap paste-back."""

    def _build_worker(self):
        return HyperSwapWorker(
            device=self.device,
            source_face_path=self.source_face_path,
            swap_model_path=self.swap_model_path,
            swap_input_size=self.swap_input_size,
            provider=self.provider,
        )


__all__ = ["HyperSwapClipRestorer"]