"""
Sonar Ingestion & Tiling Engine
===============================
Reads sonar imagery (TIFF, PNG, JPG), parses navigation metadata (CSV, NMEA),
interpolates vehicle navigation state per ping, and tiles large waterfall swaths.
"""

from __future__ import annotations
import os
import csv
import io
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image

from ..types import NavigationPoint, MissionMetadata, BoundingBox
from .filters import (
    adaptive_lee_filter,
    time_varying_gain_correction,
    normalize_sonar_image
)
from .slant_range import SlantRangeCorrector


class SonarImageReader:
    """Reads raw acoustic sonar images and normalizes to canonical 2D/3D float32 numpy arrays."""

    @staticmethod
    def read_image(file_path_or_bytes: Any) -> np.ndarray:
        if isinstance(file_path_or_bytes, np.ndarray):
            arr = file_path_or_bytes.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            return arr
        elif isinstance(file_path_or_bytes, Image.Image):
            img = file_path_or_bytes
        elif isinstance(file_path_or_bytes, (str, os.PathLike)):
            img = Image.open(file_path_or_bytes)
        elif isinstance(file_path_or_bytes, bytes):
            img = Image.open(io.BytesIO(file_path_or_bytes))
        else:
            raise ValueError(f"Unsupported input format for SonarImageReader: {type(file_path_or_bytes)}")

        # Convert to grayscale 2D array
        if img.mode != "L":
            img = img.convert("L")

        arr = np.array(img, dtype=np.float32)

        # Scale to [0, 1]
        if arr.max() > 1.0:
            arr = arr / 255.0

        return arr


class NavigationParser:
    """Parses navigation logs (CSV or NMEA) and maps ping rows to geolocated NavigationPoints."""

    @staticmethod
    def parse_csv(csv_content_or_path: Any) -> List[NavigationPoint]:
        if isinstance(csv_content_or_path, (str, os.PathLike)) and os.path.exists(str(csv_content_or_path)):
            with open(csv_content_or_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        elif isinstance(csv_content_or_path, str):
            lines = csv_content_or_path.strip().splitlines()
        elif isinstance(csv_content_or_path, bytes):
            lines = csv_content_or_path.decode("utf-8", errors="ignore").strip().splitlines()
        else:
            return []

        if not lines:
            return []

        reader = csv.DictReader(lines)
        points: List[NavigationPoint] = []

        # Find header mapping
        for i, row in enumerate(reader):
            # Case-insensitive lookup
            row_lower = {k.strip().lower(): v.strip() for k, v in row.items() if k}

            lat = float(row_lower.get("lat") or row_lower.get("latitude") or row_lower.get("y") or 0.0)
            lon = float(row_lower.get("lon") or row_lower.get("long") or row_lower.get("longitude") or row_lower.get("x") or 0.0)
            heading = float(row_lower.get("heading") or row_lower.get("heading_deg") or row_lower.get("yaw") or 0.0)
            altitude = float(row_lower.get("altitude") or row_lower.get("alt") or row_lower.get("altitude_m") or 10.0)
            depth = float(row_lower.get("depth") or row_lower.get("depth_m") or 25.0)
            speed = float(row_lower.get("speed") or row_lower.get("speed_knots") or 3.0)
            ping_num = int(row_lower.get("ping") or row_lower.get("ping_number") or i)
            ts = float(row_lower.get("timestamp") or row_lower.get("time") or (time.time() + i * 0.1))

            points.append(
                NavigationPoint(
                    timestamp=ts,
                    ping_number=ping_num,
                    latitude=lat,
                    longitude=lon,
                    heading_deg=heading,
                    altitude_m=altitude,
                    speed_knots=speed,
                    depth_m=depth
                )
            )

        return points

    @staticmethod
    def parse_nmea(nmea_text_or_path: Any) -> List[NavigationPoint]:
        """Parses basic NMEA sentences ($GPGGA, $GPRMC, $HEHDT, $GPHDT)."""
        if isinstance(nmea_text_or_path, (str, os.PathLike)) and os.path.exists(str(nmea_text_or_path)):
            with open(nmea_text_or_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        elif isinstance(nmea_text_or_path, str):
            lines = nmea_text_or_path.strip().splitlines()
        elif isinstance(nmea_text_or_path, bytes):
            lines = nmea_text_or_path.decode("utf-8", errors="ignore").strip().splitlines()
        else:
            return []

        points: List[NavigationPoint] = []
        curr_lat, curr_lon, curr_heading, curr_alt = 0.0, 0.0, 0.0, 10.0
        ping_idx = 0

        for line in lines:
            line = line.strip()
            if not line.startswith("$"):
                continue

            parts = line.split("*")[0].split(",")
            sentence = parts[0]

            if sentence in ["$GPGGA", "$GNGGA"] and len(parts) >= 10:
                try:
                    # Latitude DDMM.MMMM
                    raw_lat = parts[2]
                    lat_dir = parts[3]
                    if raw_lat and len(raw_lat) >= 4:
                        deg = float(raw_lat[:2])
                        mins = float(raw_lat[2:])
                        curr_lat = (deg + mins / 60.0) * (-1.0 if lat_dir == "S" else 1.0)

                    # Longitude DDDMM.MMMM
                    raw_lon = parts[4]
                    lon_dir = parts[5]
                    if raw_lon and len(raw_lon) >= 5:
                        deg = float(raw_lon[:3])
                        mins = float(raw_lon[3:])
                        curr_lon = (deg + mins / 60.0) * (-1.0 if lon_dir == "W" else 1.0)

                    if parts[9]:
                        curr_alt = float(parts[9])

                    points.append(
                        NavigationPoint(
                            timestamp=time.time() + ping_idx * 0.1,
                            ping_number=ping_idx,
                            latitude=curr_lat,
                            longitude=curr_lon,
                            heading_deg=curr_heading,
                            altitude_m=curr_alt
                        )
                    )
                    ping_idx += 1
                except Exception:
                    pass

            elif sentence in ["$HEHDT", "$GPHDT"] and len(parts) >= 2:
                try:
                    curr_heading = float(parts[1])
                except Exception:
                    pass

        return points

    @staticmethod
    def interpolate_navigation(nav_points: List[NavigationPoint], total_pings: int) -> List[NavigationPoint]:
        """Interpolates navigation coordinates smoothly across every ping row in the image."""
        if not nav_points:
            # Generate fallback nominal track if empty
            base_lat, base_lon = 18.9220, 72.8346  # Example coastal coordinates (Mumbai harbor / Arabian Sea)
            return [
                NavigationPoint(
                    timestamp=time.time() + i * 0.1,
                    ping_number=i,
                    latitude=base_lat + (i * 0.000005),
                    longitude=base_lon + (i * 0.000005),
                    heading_deg=45.0,
                    altitude_m=10.0,
                    speed_knots=3.0,
                    depth_m=20.0
                )
                for i in range(total_pings)
            ]

        if len(nav_points) == 1:
            pt = nav_points[0]
            return [
                NavigationPoint(
                    timestamp=pt.timestamp + i * 0.1,
                    ping_number=i,
                    latitude=pt.latitude,
                    longitude=pt.longitude,
                    heading_deg=pt.heading_deg,
                    altitude_m=pt.altitude_m,
                    speed_knots=pt.speed_knots,
                    depth_m=pt.depth_m
                )
                for i in range(total_pings)
            ]

        # Multi-point linear interpolation
        orig_indices = np.array([p.ping_number for p in nav_points])
        if orig_indices[0] == orig_indices[-1]:
            orig_indices = np.linspace(0, total_pings - 1, len(nav_points))

        target_indices = np.arange(total_pings)

        lats = np.interp(target_indices, orig_indices, [p.latitude for p in nav_points])
        lons = np.interp(target_indices, orig_indices, [p.longitude for p in nav_points])
        headings = np.interp(target_indices, orig_indices, [p.heading_deg for p in nav_points])
        altitudes = np.interp(target_indices, orig_indices, [p.altitude_m for p in nav_points])
        speeds = np.interp(target_indices, orig_indices, [p.speed_knots for p in nav_points])
        depths = np.interp(target_indices, orig_indices, [p.depth_m for p in nav_points])
        timestamps = np.interp(target_indices, orig_indices, [p.timestamp for p in nav_points])

        interpolated: List[NavigationPoint] = []
        for i in range(total_pings):
            interpolated.append(
                NavigationPoint(
                    timestamp=float(timestamps[i]),
                    ping_number=i,
                    latitude=float(lats[i]),
                    longitude=float(lons[i]),
                    heading_deg=float(headings[i]),
                    altitude_m=float(altitudes[i]),
                    speed_knots=float(speeds[i]),
                    depth_m=float(depths[i])
                )
            )

        return interpolated


class TilingEngine:
    """Slices large continuous sonar swaths into overlapping tiles for neural network inference."""

    def __init__(self, tile_size: int = 512, tile_overlap: int = 64):
        self.tile_size = int(tile_size)
        self.tile_overlap = int(tile_overlap)
        self.step = self.tile_size - self.tile_overlap

    def create_tiles(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extracts overlapping tiles from the image.

        Returns:
            List of (tile_array, (x_min, y_min, x_max, y_max)) in global image coordinates.
        """
        h, w = image.shape[:2]
        tiles: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []

        y_starts = list(range(0, max(1, h - self.tile_size + 1), self.step))
        if len(y_starts) == 0 or y_starts[-1] + self.tile_size < h:
            y_starts.append(max(0, h - self.tile_size))

        x_starts = list(range(0, max(1, w - self.tile_size + 1), self.step))
        if len(x_starts) == 0 or x_starts[-1] + self.tile_size < w:
            x_starts.append(max(0, w - self.tile_size))

        # Eliminate duplicates
        y_starts = sorted(list(set(y_starts)))
        x_starts = sorted(list(set(x_starts)))

        for y in y_starts:
            for x in x_starts:
                y_end = min(h, y + self.tile_size)
                x_end = min(w, x + self.tile_size)

                tile = image[y:y_end, x:x_end]

                # Pad if edge tile is smaller than tile_size
                th, tw = tile.shape[:2]
                if th < self.tile_size or tw < self.tile_size:
                    if tile.ndim == 2:
                        padded = np.zeros((self.tile_size, self.tile_size), dtype=image.dtype)
                        padded[:th, :tw] = tile
                    else:
                        padded = np.zeros((self.tile_size, self.tile_size, tile.shape[2]), dtype=image.dtype)
                        padded[:th, :tw, :] = tile
                    tile = padded

                tiles.append((tile, (x, y, x_end, y_end)))

        return tiles

    @staticmethod
    def map_bbox_to_global(
        local_bbox: BoundingBox,
        tile_origin: Tuple[int, int]
    ) -> BoundingBox:
        """Converts local tile-relative bounding box to global waterfall coordinates."""
        tx, ty = tile_origin
        return BoundingBox(
            xmin=local_bbox.xmin + tx,
            ymin=local_bbox.ymin + ty,
            xmax=local_bbox.xmax + tx,
            ymax=local_bbox.ymax + ty
        )


class SonarIngestEngine:
    """End-to-end ingestion and preprocessing manager."""

    def __init__(
        self,
        enable_tvg: bool = True,
        enable_lee: bool = True,
        enable_slant_range: bool = True,
        max_slant_range_m: float = 50.0
    ):
        self.enable_tvg = enable_tvg
        self.enable_lee = enable_lee
        self.enable_slant_range = enable_slant_range
        self.slant_corrector = SlantRangeCorrector(max_slant_range_m=max_slant_range_m)

    def process(
        self,
        image_input: Any,
        nav_input: Optional[Any] = None,
        altitude_m: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray, List[NavigationPoint]]:
        """
        Executes full preprocessing pipeline:
        1. Ingest raw sonar image
        2. Normalize & TVG Gain Compensation
        3. Adaptive Lee Speckle Filtering
        4. Slant-to-ground range correction
        5. Navigation log parsing and ping interpolation

        Returns:
            (raw_normalized_img, preprocessed_img, nav_points)
        """
        raw_img = SonarImageReader.read_image(image_input)
        h, w = raw_img.shape

        # Normalize
        norm_img = normalize_sonar_image(raw_img)

        # TVG
        proc_img = norm_img
        if self.enable_tvg:
            proc_img = time_varying_gain_correction(proc_img)

        # Adaptive Lee Filter
        if self.enable_lee:
            proc_img = adaptive_lee_filter(proc_img, window_size=7)

        # Slant Range
        if self.enable_slant_range:
            proc_img, _ = self.slant_corrector.correct_image(proc_img, altitude_m=altitude_m)

        # Navigation
        if nav_input is not None:
            if isinstance(nav_input, (str, bytes, os.PathLike)):
                # Try CSV first, then NMEA
                nav_points = NavigationParser.parse_csv(nav_input)
                if not nav_points:
                    nav_points = NavigationParser.parse_nmea(nav_input)
            else:
                nav_points = []
        else:
            nav_points = []

        interp_nav = NavigationParser.interpolate_navigation(nav_points, total_pings=h)

        return norm_img, proc_img, interp_nav
