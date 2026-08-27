"""
Unified End-to-End Side Scan Sonar Marine Debris Pipeline
=========================================================
Orchestrates Ingestion -> Preprocessing -> Tiled Inference -> Physics Verification
-> Geotagging -> Reporting -> Thumbnail Generation in a modular, offline pipeline.
"""

from __future__ import annotations
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from .types import (
    PipelineConfig,
    MissionMetadata,
    MissionReport,
    DetectionResult,
    BoundingBox,
    DebrisClass,
    AuditStatus
)
from .preprocessing import (
    SonarIngestEngine,
    TilingEngine,
    SonarImageReader
)
from .models import (
    PyTorchSonarDetector,
    ONNXSonarDetector,
    BaseSonarDetector
)
from .filtering import (
    SonarPostProcessor
)
from .geotagging import (
    SonarGeoreferencer,
    SonarReporter
)


class SonarDebrisPipeline:
    """
    Complete end-to-end edge-deployable pipeline for automated SSS marine debris detection.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.ingest_engine = SonarIngestEngine(
            enable_tvg=self.config.enable_tvg,
            enable_lee=self.config.enable_lee_filter,
            enable_slant_range=self.config.enable_slant_range_correction
        )
        self.tiling_engine = TilingEngine(
            tile_size=self.config.tile_size,
            tile_overlap=self.config.tile_overlap
        )
        self.post_processor = SonarPostProcessor(
            confidence_threshold=self.config.confidence_threshold_percent,
            low_confidence_threshold=self.config.low_confidence_audit_threshold,
            nms_iou_threshold=self.config.nms_iou_threshold,
            enable_physics_validation=self.config.enable_shadow_physics_validation
        )
        self.georeferencer = SonarGeoreferencer(
            max_slant_range_m=50.0
        )
        self.detector = self._initialize_detector()

    def _initialize_detector(self) -> BaseSonarDetector:
        if self.config.model_type == "onnx" and self.config.onnx_model_path:
            return ONNXSonarDetector(self.config.onnx_model_path)
        return PyTorchSonarDetector()

    def run(
        self,
        sonar_input: Any,
        nav_input: Optional[Any] = None,
        mission_id: Optional[str] = None,
        survey_name: str = "SSS Debris Survey",
        output_dir: Optional[str] = None
    ) -> MissionReport:
        """
        Executes the detection pipeline on a sonar image and navigation log.

        Args:
            sonar_input: Image path, bytes, or numpy array.
            nav_input: Navigation CSV/NMEA path, text, or bytes.
            mission_id: Unique mission identifier string.
            survey_name: Descriptive mission name.
            output_dir: Directory to save thumbnail crops and reports.

        Returns:
            MissionReport containing detections, audit log, metadata, and summary metrics.
        """
        t_start = time.time()
        m_id = mission_id or f"mission_{uuid.uuid4().hex[:8]}"

        # 1. Ingestion & Preprocessing
        norm_img, proc_img, nav_track = self.ingest_engine.process(
            image_input=sonar_input,
            nav_input=nav_input
        )
        h, w = proc_img.shape[:2]

        # 2. Tiling
        tiles = self.tiling_engine.create_tiles(proc_img)

        # 3. Model Inference on Tiles
        raw_candidates: List[Dict[str, Any]] = []
        for tile_arr, (tx1, ty1, tx2, ty2) in tiles:
            tile_detections = self.detector.predict(
                image_tile=tile_arr,
                conf_threshold=0.35  # Low threshold to capture candidates before physics filtering
            )
            for det in tile_detections:
                local_box: BoundingBox = det["bbox"]
                global_box = self.tiling_engine.map_bbox_to_global(local_box, (tx1, ty1))
                # Clip to image boundaries
                global_box.xmin = max(0.0, min(float(w), global_box.xmin))
                global_box.ymin = max(0.0, min(float(h), global_box.ymin))
                global_box.xmax = max(0.0, min(float(w), global_box.xmax))
                global_box.ymax = max(0.0, min(float(h), global_box.ymax))

                cand_dict = dict(det)
                cand_dict["bbox"] = global_box
                raw_candidates.append(cand_dict)

        # 4. Post-processing & Physics Filtering
        avg_alt = np.mean([p.altitude_m for p in nav_track]) if nav_track else 10.0
        primary_raw, audit_raw = self.post_processor.process(
            candidates=raw_candidates,
            sonar_img=proc_img,
            towfish_altitude_m=float(avg_alt)
        )

        # 5. Geotagging & Physical Dimension Calculation
        primary_detections: List[DetectionResult] = []
        for item in primary_raw:
            bbox: BoundingBox = item["bbox"]
            shadow_len = item.get("acoustic_signature", {}).shadow_length_px if hasattr(item.get("acoustic_signature"), "shadow_length_px") else 0.0
            geo_pt, dims = self.georeferencer.geotag_detection(
                bbox=bbox,
                image_shape=(h, w),
                nav_track=nav_track,
                shadow_len_px=shadow_len
            )

            primary_detections.append(
                DetectionResult(
                    id=item["id"],
                    class_name=item["class_name"],
                    confidence_percent=item["confidence_percent"],
                    raw_model_score=item["raw_model_score"],
                    physics_score=item["physics_score"],
                    channel=item["channel"],
                    bbox=bbox,
                    geo_location=geo_pt,
                    dimensions=dims,
                    acoustic_signature=item["acoustic_signature"],
                    source_file=str(mission_id or "sonar_input"),
                    ping_range=(int(bbox.ymin), int(bbox.ymax)),
                    status=AuditStatus.HIGH_CONFIDENCE
                )
            )

        audit_detections: List[DetectionResult] = []
        for item in audit_raw:
            bbox: BoundingBox = item["bbox"]
            geo_pt, dims = self.georeferencer.geotag_detection(
                bbox=bbox,
                image_shape=(h, w),
                nav_track=nav_track
            )
            audit_detections.append(
                DetectionResult(
                    id=item["id"],
                    class_name=item["class_name"],
                    confidence_percent=item["confidence_percent"],
                    raw_model_score=item["raw_model_score"],
                    physics_score=item["physics_score"],
                    channel=item["channel"],
                    bbox=bbox,
                    geo_location=geo_pt,
                    dimensions=dims,
                    acoustic_signature=item["acoustic_signature"],
                    status=AuditStatus.LOW_CONFIDENCE
                )
            )

        # 6. Extract Thumbnails if output_dir specified
        if output_dir and self.config.export_thumbnails:
            crops_dir = os.path.join(output_dir, "crops")
            SonarReporter.extract_thumbnail_crops(
                sonar_img=proc_img,
                detections=primary_detections + audit_detections,
                output_dir=crops_dir
            )

        t_end = time.time()
        elapsed_sec = t_end - t_start

        # Calculate survey coverage area in square kilometers
        swath_width_m = self.georeferencer.max_slant_range_m * 2.0
        survey_length_m = h * 0.05  # nominal ~ 0.05m per ping
        area_sq_km = (swath_width_m * survey_length_m) / 1e6

        # Class counts
        class_counts: Dict[str, int] = {}
        for d in primary_detections:
            c_val = d.class_name.value if hasattr(d.class_name, "value") else str(d.class_name)
            class_counts[c_val] = class_counts.get(c_val, 0) + 1

        ghost_nets_count = class_counts.get(DebrisClass.GHOST_NET.value, 0)
        shipwrecks_count = class_counts.get(DebrisClass.SHIPWRECK.value, 0)
        pipes_count = class_counts.get(DebrisClass.PIPE_CYLINDER.value, 0)

        metadata = MissionMetadata(
            mission_id=m_id,
            survey_name=survey_name,
            filename=str(mission_id or "sonar_scan.png"),
            start_time=nav_track[0].timestamp if nav_track else t_start,
            end_time=nav_track[-1].timestamp if nav_track else t_end,
            total_pings=h,
            image_width=w,
            image_height=h,
            max_slant_range_m=self.georeferencer.max_slant_range_m,
            altitude_avg_m=float(avg_alt)
        )

        summary = {
            "total_detections": len(primary_detections),
            "ghost_nets_found": ghost_nets_count,
            "shipwrecks_found": shipwrecks_count,
            "pipes_found": pipes_count,
            "low_confidence_audit_count": len(audit_detections),
            "class_breakdown": class_counts,
            "survey_area_sq_km": round(area_sq_km, 4),
            "survey_length_m": round(survey_length_m, 1),
            "processing_time_sec": round(elapsed_sec, 3),
            "throughput_pings_per_sec": round(h / max(0.001, elapsed_sec), 1),
            "edge_device_ready": True
        }

        report = MissionReport(
            metadata=metadata,
            config=self.config,
            summary=summary,
            detections=primary_detections,
            audit_log=audit_detections,
            nav_track=nav_track
        )

        return report
