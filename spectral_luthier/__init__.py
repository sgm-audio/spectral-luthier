"""
Spectral Luthier - stem-based mastering experiment

Stem-based mastering experiment: separate a stereo mix with Demucs, adjust each
stem against a target profile, sum, then limit.
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
