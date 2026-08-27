"""
Physics-Based Confidence Scoring & Noise Filtering
==================================================
Eliminates false positives from natural geological clutter (rocks, ripples, shadows)
using acoustic shadow/highlight geometry, directional nadir constraints, and NMS.
"""

from .shadow_analyzer import AcousticShadowAnalyzer
from .postprocessor import SonarPostProcessor

__all__ = [
    "AcousticShadowAnalyzer",
    "SonarPostProcessor"
]
