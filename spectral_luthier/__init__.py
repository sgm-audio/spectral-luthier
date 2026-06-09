"""
Spectral Neural Luthier - Autonomous Stem-Aware Spectral Mastering

This module provides cutting-edge spectral mastering using source separation
and stem-level rebalancing to fix mix issues without access to the original multitrack.
"""

from .stem_master import StemMaster
from .analyzer import StemAnalyzer
from .rebalancer import StemRebalancer
from .limiter import TransparentLimiter
from .profiles import TargetProfile

__version__ = "0.1.0"
__all__ = [
    "StemMaster",
    "StemAnalyzer",
    "StemRebalancer",
    "TransparentLimiter",
    "TargetProfile",
]
