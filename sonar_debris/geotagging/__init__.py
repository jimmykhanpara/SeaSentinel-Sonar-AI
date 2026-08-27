"""
Geotagging & Reporting Subpackage
=================================
Maps sonar pixel detections to real-world WGS84 coordinates, estimates physical
dimensions from acoustic shadows, and exports JSON, CSV, and QGIS GeoJSON reports.
"""

from .georeferencer import SonarGeoreferencer
from .reporter import SonarReporter

__all__ = [
    "SonarGeoreferencer",
    "SonarReporter"
]
