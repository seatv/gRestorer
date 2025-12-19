"""
gRestorer - GPU-accelerated video mosaic removal pipeline

A high-performance video processing pipeline that uses NVDEC/NVENC hardware
acceleration and BasicVSR++ deep learning model for mosaic detection and removal.
"""

__version__ = "0.1.0"
__author__ = "Gman"

# Core imports
from . import core
from . import detector
from . import restorer
from . import utils
from . import video

__all__ = [
    'core',
    'detector',
    'restorer',
    'utils',
    'video',
]
