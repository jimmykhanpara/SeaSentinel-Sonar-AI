"""
Comprehensive Report & Geodata Export Engine
============================================
Generates structured JSON, flat operational CSV, QGIS-compatible GeoJSON,
annotated sonar waterfall mosaics, and thumbnail crop images for cleanup missions.
"""

from __future__ import annotations
import os
import io
import csv
import json
import time
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..types import (
    DetectionResult,
    MissionMetadata,
    PipelineConfig,
    MissionReport,
    AuditStatus,
    DebrisClass
)


class SonarReporter:
    """
    Export and reporting engine for marine debris detection results.
    """

    CLASS_COLORS = {
        DebrisClass.GHOST_NET: (255, 59, 48),       # Bright Red
        DebrisClass.SHIPWRECK: (255, 149, 0),       # Orange
        DebrisClass.PIPE_CYLINDER: (0, 122, 255),   # Blue
        DebrisClass.CONTAINER: (88, 86, 214),       # Purple
        DebrisClass.TIRE: (52, 199, 89),            # Green
        DebrisClass.GENERIC_DEBRIS: (255, 204, 0),  # Yellow
        DebrisClass.UNKNOWN_ANOMALY: (142, 142, 147) # Gray
    }

    @staticmethod
    def generate_json_report(report: MissionReport) -> str:
        """Serializes mission report into a structured JSON string."""
        return report.model_dump_json(indent=2)

    @staticmethod
    def generate_csv_report(detections: List[DetectionResult]) -> str:
        """Generates operational flat CSV for survey vessel and retrieval teams."""
        output = io.StringIO()
        fieldnames = [
            "id",
            "class_name",
            "confidence_percent",
            "latitude",
            "longitude",
            "length_m",
            "width_m",
            "estimated_height_m",
            "area_m2",
            "channel",
            "ground_range_m",
            "slant_range_m",
            "depth_m",
            "status",
            "contrast_ratio",
            "shadow_length_px",
            "bbox_xmin",
            "bbox_ymin",
            "bbox_xmax",
            "bbox_ymax",
            "created_at"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for d in detections:
            writer.writerow({
                "id": d.id,
                "class_name": d.class_name.value if hasattr(d.class_name, "value") else str(d.class_name),
                "confidence_percent": d.confidence_percent,
                "latitude": d.geo_location.latitude,
                "longitude": d.geo_location.longitude,
                "length_m": d.dimensions.length_m,
                "width_m": d.dimensions.width_m,
                "estimated_height_m": d.dimensions.estimated_height_m,
                "area_m2": d.dimensions.area_m2,
                "channel": d.channel.value if hasattr(d.channel, "value") else str(d.channel),
                "ground_range_m": d.geo_location.ground_range_m,
                "slant_range_m": d.geo_location.slant_range_m,
                "depth_m": d.geo_location.depth_m,
                "status": d.status.value if hasattr(d.status, "value") else str(d.status),
                "contrast_ratio": d.acoustic_signature.contrast_ratio,
                "shadow_length_px": d.acoustic_signature.shadow_length_px,
                "bbox_xmin": round(d.bbox.xmin, 1),
                "bbox_ymin": round(d.bbox.ymin, 1),
                "bbox_xmax": round(d.bbox.xmax, 1),
                "bbox_ymax": round(d.bbox.ymax, 1),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(d.created_at))
            })

        return output.getvalue()

    @staticmethod
    def generate_geojson_report(
        detections: List[DetectionResult],
        survey_track: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates QGIS and GIS-compliant GeoJSON FeatureCollection with Point and Polygon features.
        """
        features: List[Dict[str, Any]] = []

        # Add detection point features
        for d in detections:
            props = {
                "id": d.id,
                "class": d.class_name.value if hasattr(d.class_name, "value") else str(d.class_name),
                "confidence": d.confidence_percent,
                "length_m": d.dimensions.length_m,
                "width_m": d.dimensions.width_m,
                "height_m": d.dimensions.estimated_height_m,
                "area_m2": d.dimensions.area_m2,
                "channel": d.channel.value if hasattr(d.channel, "value") else str(d.channel),
                "ground_range_m": d.geo_location.ground_range_m,
                "status": d.status.value if hasattr(d.status, "value") else str(d.status),
                "contrast_ratio": d.acoustic_signature.contrast_ratio,
                "marker_color": "#{:02x}{:02x}{:02x}".format(*SonarReporter.CLASS_COLORS.get(d.class_name, (255, 255, 0)))
            }

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [d.geo_location.longitude, d.geo_location.latitude]
                },
                "properties": props
            })

        # Add vessel survey path as LineString if provided
        if survey_track and len(survey_track) > 1:
            line_coords = [[p.longitude, p.latitude] for p in survey_track]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": line_coords
                },
                "properties": {
                    "name": "Survey Mission Track",
                    "total_pings": len(survey_track),
                    "stroke": "#00ffcc",
                    "stroke-width": 3
                }
            })

        return {
            "type": "FeatureCollection",
            "features": features
        }

    @staticmethod
    def extract_thumbnail_crops(
        sonar_img: np.ndarray,
        detections: List[DetectionResult],
        output_dir: str,
        padding_px: int = 24
    ) -> Dict[str, str]:
        """
        Extracts cropped image thumbnails for each detected target.
        """
        os.makedirs(output_dir, exist_ok=True)
        h, w = sonar_img.shape[:2]
        crop_urls: Dict[str, str] = {}

        img_uint8 = np.clip(sonar_img * 255.0, 0, 255).astype(np.uint8)

        for d in detections:
            x1 = max(0, int(d.bbox.xmin - padding_px))
            y1 = max(0, int(d.bbox.ymin - padding_px))
            x2 = min(w, int(d.bbox.xmax + padding_px))
            y2 = min(h, int(d.bbox.ymax + padding_px))

            crop = img_uint8[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pil_crop = Image.fromarray(crop)
            filename = f"crop_{d.id}_{d.class_name.value if hasattr(d.class_name, 'value') else str(d.class_name)}.png"
            crop_path = os.path.join(output_dir, filename)
            pil_crop.save(crop_path)
            crop_urls[d.id] = filename
            d.thumbnail_url = f"/api/crops/{filename}"

        return crop_urls

    @classmethod
    def generate_annotated_mosaic(
        cls,
        sonar_img: np.ndarray,
        detections: List[DetectionResult]
    ) -> np.ndarray:
        """
        Renders bounding boxes, confidence pills, and labels on the sonar image.
        """
        img_uint8 = np.clip(sonar_img * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8).convert("RGB")
        draw = ImageDraw.Draw(pil_img)

        for d in detections:
            color = cls.CLASS_COLORS.get(d.class_name, (255, 255, 0))
            x1, y1 = int(d.bbox.xmin), int(d.bbox.ymin)
            x2, y2 = int(d.bbox.xmax), int(d.bbox.ymax)

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Label banner
            label = f"{d.class_name.value if hasattr(d.class_name, 'value') else str(d.class_name)}: {d.confidence_percent:.0f}%"
            # Text background badge
            text_w = len(label) * 7 + 8
            draw.rectangle([x1, max(0, y1 - 18), x1 + text_w, y1], fill=color)
            draw.text((x1 + 4, max(0, y1 - 16)), label, fill=(0, 0, 0))

        return np.array(pil_img)
