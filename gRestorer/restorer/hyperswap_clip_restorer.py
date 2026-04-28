# gRestorer/restorer/hyperswap_clip_restorer.py

from __future__ import annotations

from gRestorer.restorer.base_face_swap_clip_restorer import BaseFaceSwapClipRestorer
from gRestorer.restorer.hyperswap_worker import HyperSwapWorker


class HyperSwapClipRestorer(BaseFaceSwapClipRestorer):
    """Face-swap clip restorer using native HyperSwap paste-back."""

    def __init__(
        self,
        *args,
        swap_pixel_boost: str = "",
        swap_mask_box_blur: float | None = None,
        swap_mask_box_padding=None,
        hyperswap_output_shift_x: float = 0.0,
        hyperswap_output_shift_y: float = 0.0,
        hyperswap_output_scale: float = 1.0,
        **kwargs,
    ):
        self.swap_pixel_boost = str(swap_pixel_boost or "").strip()
        self.hyperswap_output_shift_x = float(hyperswap_output_shift_x)
        self.hyperswap_output_shift_y = float(hyperswap_output_shift_y)
        self.hyperswap_output_scale = float(hyperswap_output_scale)
        super().__init__(
            *args,
            swap_mask_box_blur=swap_mask_box_blur,
            swap_mask_box_padding=swap_mask_box_padding,
            **kwargs,
        )

    def _build_worker(self):
        return HyperSwapWorker(
            device=self.device,
            source_face_path=self.source_face_path,
            swap_model_path=self.swap_model_path,
            swap_input_size=self.swap_input_size,
            provider=self.provider,
            pixel_boost=(self.swap_pixel_boost or None),
            mask_box_blur=self.swap_mask_box_blur,
            mask_box_padding=self.swap_mask_box_padding,
            output_shift_x=self.hyperswap_output_shift_x,
            output_shift_y=self.hyperswap_output_shift_y,
            output_scale=self.hyperswap_output_scale,
        )


__all__ = ["HyperSwapClipRestorer"]
