"""
Post-Processing, Confidence Calibration & False-Positive Suppression
====================================================================
Applies Non-Maximum Suppression (NMS), physics-informed confidence calibration,
false-positive suppression, and active learning feedback hooks.
"""

from __future__ import annotations
import os
import json
import time
import uuid
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from ..types import (
    DetectionResult,
    BoundingBox,
    DebrisClass,
    AuditStatus,
    ChannelType,
    GeoPoint,
    PhysicalDimensions,
    AcousticSignature
)
from .shadow_analyzer import AcousticShadowAnalyzer


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    x_left = max(box1.xmin, box2.xmin)
    y_top = max(box1.ymin, box2.ymin)
    x_right = min(box1.xmax, box2.xmax)
    y_bottom = min(box1.ymax, box2.ymax)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = box1.area
    area2 = box2.area
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0

    return float(intersection_area / union_area)


def non_maximum_suppression(
    candidates: List[Dict[str, Any]],
    iou_threshold: float = 0.45
) -> List[Dict[str, Any]]:
    """Applies Non-Maximum Suppression to deduplicate multi-scale / overlapping tile detections."""
    if not candidates:
        return []

    # Sort descending by confidence
    sorted_candidates = sorted(candidates, key=lambda c: c.get("confidence", 0.0), reverse=True)
    selected: List[Dict[str, Any]] = []

    while sorted_candidates:
        best = sorted_candidates.pop(0)
        selected.append(best)
        best_box: BoundingBox = best["bbox"]

        remaining = []
        for cand in sorted_candidates:
            cand_box: BoundingBox = cand["bbox"]
            # Check class match and IoU
            if cand.get("class_name") == best.get("class_name"):
                iou = calculate_iou(best_box, cand_box)
                if iou < iou_threshold:
                    remaining.append(cand)
            else:
                # Different classes but nearly identical box (> 0.7 IoU) -> keep the higher confidence one
                iou = calculate_iou(best_box, cand_box)
                if iou < 0.70:
                    remaining.append(cand)

        sorted_candidates = remaining

    return selected


class SonarPostProcessor:
    """
    Orchestrates physics validation, confidence calibration, thresholding,
    and audit log partitioning for raw sonar detection candidates.
    """

    def __init__(
        self,
        confidence_threshold: float = 60.0,
        low_confidence_threshold: float = 30.0,
        nms_iou_threshold: float = 0.45,
        enable_physics_validation: bool = True,
        feedback_log_path: str = "active_learning_feedback.jsonl"
    ):
        self.conf_threshold = float(confidence_threshold)
        self.low_conf_threshold = float(low_confidence_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.enable_physics = enable_physics_validation
        self.shadow_analyzer = AcousticShadowAnalyzer()
        self.feedback_log_path = feedback_log_path

    def process(
        self,
        candidates: List[Dict[str, Any]],
        sonar_img: np.ndarray,
        towfish_altitude_m: float = 10.0,
        max_slant_range_m: float = 50.0
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Runs physics verification, confidence calibration, and partitions into
        (high_confidence_detections, low_confidence_audit_log).

        Returns:
            Tuple of (primary_detections, audit_log)
        """
        h, w = sonar_img.shape[:2]
        center_x = w / 2.0

        calibrated_candidates: List[Dict[str, Any]] = []

        for cand in candidates:
            bbox: BoundingBox = cand["bbox"]
            raw_model_score = float(cand.get("raw_score") or cand.get("confidence") or 0.5)
            cls_name: DebrisClass = cand.get("class_name", DebrisClass.GENERIC_DEBRIS)

            # Suppress explicit rock clutter negative class
            if cls_name == DebrisClass.ROCK_CLUTTER:
                cand["confidence"] = 0.15
                cand["status"] = AuditStatus.LOW_CONFIDENCE
                continue

            # Physics validation
            if self.enable_physics:
                physics_score, acoustic_sig, is_phys_valid = self.shadow_analyzer.analyze_candidate(
                    sonar_img=sonar_img,
                    bbox=bbox,
                    towfish_altitude_m=towfish_altitude_m,
                    max_slant_range_m=max_slant_range_m
                )
            else:
                physics_score = raw_model_score
                acoustic_sig = AcousticSignature()
                is_phys_valid = True

            # Calibrated confidence formula
            # If physics is strongly invalid (e.g. shadow points into nadir), down-weight heavily
            if not is_phys_valid:
                physics_score = physics_score * 0.4

            calibrated_prob = 0.55 * raw_model_score + 0.45 * physics_score
            calibrated_percent = float(np.clip(calibrated_prob * 100.0, 0.0, 100.0))

            cand_center_x = bbox.center[0]
            channel = ChannelType.PORT if cand_center_x < center_x else ChannelType.STARBOARD

            cand_updated = dict(cand)
            cand_updated["id"] = cand.get("id") or str(uuid.uuid4())[:8]
            cand_updated["raw_model_score"] = round(raw_model_score, 3)
            cand_updated["physics_score"] = round(physics_score, 3)
            cand_updated["confidence"] = round(calibrated_prob, 3)
            cand_updated["confidence_percent"] = round(calibrated_percent, 1)
            cand_updated["acoustic_signature"] = acoustic_sig
            cand_updated["channel"] = channel

            calibrated_candidates.append(cand_updated)

        # Apply NMS
        deduped = non_maximum_suppression(calibrated_candidates, iou_threshold=self.nms_iou_threshold)

        # Split into primary and audit log
        primary_detections = []
        audit_log = []

        for item in deduped:
            conf = item["confidence_percent"]
            if conf >= self.conf_threshold:
                item["status"] = AuditStatus.HIGH_CONFIDENCE
                primary_detections.append(item)
            elif conf >= self.low_conf_threshold:
                item["status"] = AuditStatus.LOW_CONFIDENCE
                audit_log.append(item)

        return primary_detections, audit_log

    def log_analyst_feedback(
        self,
        detection_id: str,
        class_name: str,
        is_confirmed: bool,
        notes: Optional[str] = None,
        bbox: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Active Learning Hook: Saves human analyst feedback on false positives or confirmed targets.
        """
        record = {
            "timestamp": time.time(),
            "detection_id": detection_id,
            "class_name": class_name,
            "action": "confirmed" if is_confirmed else "rejected_false_positive",
            "notes": notes or "",
            "bbox": bbox or {}
        }
        try:
            with open(self.feedback_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            return True
        except Exception:
            return False
