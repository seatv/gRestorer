from __future__ import annotations

# Compatibility wrapper: existing imports of FaceSwapWorker continue to
# resolve to the known-good InSwapper implementation.
from gRestorer.restorer.inswapper_worker import InSwapperWorker as FaceSwapWorker

__all__ = ["FaceSwapWorker"]
