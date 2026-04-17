from __future__ import annotations

# Compatibility wrapper: existing imports of FaceSwapClipRestorer continue to
# resolve to the known-good InSwapper implementation.
from gRestorer.restorer.inswapper_clip_restorer import InSwapperClipRestorer as FaceSwapClipRestorer

__all__ = ["FaceSwapClipRestorer"]
