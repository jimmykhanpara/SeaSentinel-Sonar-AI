"""
Preprocessing and Ingestion Subpackage
======================================
Provides slant-range correction, acoustic filters (Adaptive Lee, TVG),
nadir detection, image/navigation ingestion, and tiling mechanisms.
"""

from .slant_range import SlantRangeCorrector
from .filters import (
    adaptive_lee_filter,
    enhanced_lee_filter,
    time_varying_gain_correction,
    detect_nadir_and_dropouts,
    normalize_sonar_image
)
from .ingest import (
    SonarImageReader,
    NavigationParser,
    SonarIngestEngine,
    TilingEngine
)

__all__ = [
    "SlantRangeCorrector",
    "adaptive_lee_filter",
    "enhanced_lee_filter",
    "time_varying_gain_correction",
    "detect_nadir_and_dropouts",
    "normalize_sonar_image",
    "SonarImageReader",
    "NavigationParser",
    "SonarIngestEngine",
    "TilingEngine"
]
