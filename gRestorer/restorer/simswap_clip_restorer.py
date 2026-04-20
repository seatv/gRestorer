from __future__ import annotations

import torch

from gRestorer.restorer.base_face_swap_clip_restorer import BaseFaceSwapClipRestorer
from gRestorer.restorer.simswap_worker import SimSwapWorker


class SimSwapClipRestorer(BaseFaceSwapClipRestorer):
    """Face-swap clip restorer using SimSwap ONNX as the concrete worker."""

    def _build_worker(self):
        return SimSwapWorker(
            device=self.device,
            source_face_path=self.source_face_path,
            swap_model_path=self.swap_model_path,
            provider=self.provider,
            video_memory_strategy=self.face_video_memory_strategy,
            simswap_gpu_mem_limit_mb=self.face_simswap_gpu_mem_limit_mb,
            aux_gpu_mem_limit_mb=self.face_aux_gpu_mem_limit_mb,
            cuda_use_max_workspace=self.face_cuda_use_max_workspace,
        )

__all__ = ["SimSwapClipRestorer"]
