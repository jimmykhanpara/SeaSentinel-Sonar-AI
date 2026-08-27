"""
Acoustic Georeferencer & Physical Dimension Estimator
=====================================================
Transforms acoustic waterfall pixel detections into geodetic WGS84 coordinates
(latitude, longitude) using vessel navigation, tow-fish attitude, heading,
and slant-to-ground range geometry.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from ..types import (
    BoundingBox,
    GeoPoint,
    PhysicalDimensions,
    NavigationPoint,
    ChannelType
)


class SonarGeoreferencer:
    """
    Geotagging and real-world spatial measurement engine for Side Scan Sonar.
    """
    EARTH_RADIUS_M = 6371000.0

    def __init__(
        self,
        max_slant_range_m: float = 50.0,
        across_track_res_m: Optional[float] = None,
        along_track_res_m: Optional[float] = None
    ):
        self.max_slant_range_m = float(max_slant_range_m)
        self.across_res = across_track_res_m
        self.along_res = along_track_res_m

    def geotag_detection(
        self,
        bbox: BoundingBox,
        image_shape: Tuple[int, int],
        nav_track: List[NavigationPoint],
        shadow_len_px: float = 0.0
    ) -> Tuple[GeoPoint, PhysicalDimensions]:
        """
        Calculates real-world GPS coordinates and physical dimensions for a detected bounding box.

        Args:
            bbox: Detection bounding box in global image coordinates.
            image_shape: (H_pings, W_samples)
            nav_track: List of NavigationPoints interpolated for every ping row.
            shadow_len_px: Estimated shadow length in pixels.

        Returns:
            Tuple of (GeoPoint, PhysicalDimensions)
        """
        h_pings, w_samples = image_shape[:2]
        center_x = w_samples / 2.0
        samples_per_channel = w_samples / 2.0

        # Center ping row of detection
        center_y = int(np.clip(bbox.center[1], 0, h_pings - 1))
        center_x_px = bbox.center[0]

        # Fetch navigation state at target ping row
        if nav_track and center_y < len(nav_track):
            nav = nav_track[center_y]
        elif nav_track:
            nav = nav_track[-1]
        else:
            nav = NavigationPoint(latitude=18.9220, longitude=72.8346, heading_deg=45.0, altitude_m=10.0)

        # Lateral pixel offset from nadir
        pixel_offset = center_x_px - center_x
        channel = ChannelType.PORT if pixel_offset < 0 else ChannelType.STARBOARD
        slant_range_ratio = abs(pixel_offset) / samples_per_channel
        slant_r = slant_range_ratio * self.max_slant_range_m

        # Ground range Rg = sqrt(max(0, Rs^2 - h^2))
        alt = max(0.5, nav.altitude_m)
        if slant_r > alt:
            ground_r = math.sqrt(slant_r**2 - alt**2)
        else:
            ground_r = 0.0

        # Calculate bearing perpendicular to vessel heading
        # Port is heading - 90 deg, Starboard is heading + 90 deg
        if channel == ChannelType.PORT:
            bearing_deg = (nav.heading_deg - 90.0) % 360.0
        else:
            bearing_deg = (nav.heading_deg + 90.0) % 360.0

        # Compute WGS84 coordinates using direct geodesic formula
        target_lat, target_lon = self._compute_destination_point(
            lat=nav.latitude,
            lon=nav.longitude,
            distance_m=ground_r,
            bearing_deg=bearing_deg
        )

        # Resolution calculations
        m_per_px_across = self.across_res or (self.max_slant_range_m / samples_per_channel)
        # Along track resolution from speed (knots to m/s) and ping rate (~10Hz) or default
        m_per_px_along = self.along_res or ((nav.speed_knots * 0.514444 * 0.1) if nav.speed_knots > 0 else 0.05)

        # Physical Dimensions
        length_m = max(0.2, bbox.height * m_per_px_along)
        width_m = max(0.2, bbox.width * m_per_px_across)
        area_m2 = length_m * width_m

        # Estimate Object Height above seafloor from acoustic shadow: Ho = (h * Ls) / (Rg + Ls)
        shadow_len_m = shadow_len_px * m_per_px_across
        if (ground_r + shadow_len_m) > 0 and shadow_len_m > 0:
            est_height_m = (alt * shadow_len_m) / (ground_r + shadow_len_m)
            est_height_m = float(np.clip(est_height_m, 0.05, 12.0))
        else:
            est_height_m = 0.5

        geo_point = GeoPoint(
            latitude=round(target_lat, 7),
            longitude=round(target_lon, 7),
            depth_m=round(nav.depth_m, 1),
            towfish_altitude_m=round(alt, 1),
            slant_range_m=round(slant_r, 2),
            ground_range_m=round(ground_r, 2)
        )

        dimensions = PhysicalDimensions(
            length_m=round(length_m, 2),
            width_m=round(width_m, 2),
            estimated_height_m=round(est_height_m, 2),
            area_m2=round(area_m2, 2)
        )

        return geo_point, dimensions

    def _compute_destination_point(
        self,
        lat: float,
        lon: float,
        distance_m: float,
        bearing_deg: float
    ) -> Tuple[float, float]:
        """
        Direct geodesic computation on a spherical Earth.
        """
        if distance_m <= 0.0:
            return lat, lon

        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        brng_rad = math.radians(bearing_deg)
        d_r = distance_m / self.EARTH_RADIUS_M

        target_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(d_r) +
            math.cos(lat_rad) * math.sin(d_r) * math.cos(brng_rad)
        )

        target_lon_rad = lon_rad + math.atan2(
            math.sin(brng_rad) * math.sin(d_r) * math.cos(lat_rad),
            math.cos(d_r) - math.sin(lat_rad) * math.sin(target_lat_rad)
        )

        return math.degrees(target_lat_rad), math.degrees(target_lon_rad)
