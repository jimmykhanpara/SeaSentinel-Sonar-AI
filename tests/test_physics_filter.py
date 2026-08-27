import pytest
import numpy as np

from sonar_debris.filtering.shadow_analyzer import AcousticShadowAnalyzer
from sonar_debris.filtering.postprocessor import SonarPostProcessor, calculate_iou, non_maximum_suppression
from sonar_debris.types import BoundingBox, DebrisClass


def test_iou_calculation():
    box1 = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)
    box2 = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)
    assert calculate_iou(box1, box2) == 1.0

    box3 = BoundingBox(xmin=60, ymin=60, xmax=100, ymax=100)
    assert calculate_iou(box1, box3) == 0.0

    box4 = BoundingBox(xmin=30, ymin=10, xmax=50, ymax=50)
    assert 0.0 < calculate_iou(box1, box4) < 1.0


def test_nms():
    cands = [
        {"id": "1", "class_name": DebrisClass.GHOST_NET, "confidence": 0.9, "bbox": BoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)},
        {"id": "2", "class_name": DebrisClass.GHOST_NET, "confidence": 0.8, "bbox": BoundingBox(xmin=12, ymin=12, xmax=48, ymax=48)},
        {"id": "3", "class_name": DebrisClass.GHOST_NET, "confidence": 0.7, "bbox": BoundingBox(xmin=100, ymin=100, xmax=140, ymax=140)}
    ]
    deduped = non_maximum_suppression(cands, iou_threshold=0.4)
    assert len(deduped) == 2
    ids = [c["id"] for c in deduped]
    assert "1" in ids and "3" in ids


def test_acoustic_shadow_analyzer():
    analyzer = AcousticShadowAnalyzer()
    img = np.full((100, 200), 0.3, dtype=np.float32)

    # Place a valid Starboard object (center_x = 100, object at x=150)
    # Highlight at 145..155, Shadow at 155..175 (pointing RIGHT, away from nadir)
    img[40:60, 145:155] = 0.95
    img[40:60, 155:175] = 0.02

    bbox_valid = BoundingBox(xmin=140, ymin=35, xmax=180, ymax=65)
    score_valid, sig_valid, is_valid = analyzer.analyze_candidate(img, bbox_valid)

    assert score_valid > 0.5
    assert is_valid is True
    assert sig_valid.contrast_ratio > 2.0
