"""
gRestorer - GPU-accelerated video mosaic removal pipeline.

Keep this module lightweight: do NOT import detector/restorer/etc at import-time.
This enables optional dependencies (like ultralytics) without breaking
basic usage (e.g., --restorer none).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"
__author__ = "GMan"

# Subpackages we want to expose as attributes (but only import on demand).
_LAZY_SUBMODULES = {
    "cli",
    "core",
    "detector",
    "restorer",
    "utils",
    "video",
    # "synthmosaic",  # removed (module deleted)
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        mod = import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES))


__all__ = ["__version__", "__author__", *_LAZY_SUBMODULES]
