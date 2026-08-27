import pytest
import json
import numpy as np

from sonar_debris.geotagging.reporter import SonarReporter
from sonar_debris.types import (
    DetectionResult,
    BoundingBox,
    GeoPoint,
    PhysicalDimensions,
    AcousticSignature,
    DebrisClass,
    AuditStatus,
    MissionReport,
    MissionMetadata,
    PipelineConfig
)


@pytest.fixture
def sample_detection():
    return DetectionResult(
        id="det_001",
        class_name=DebrisClass.GHOST_NET,
        confidence_percent=88.5,
        raw_model_score=0.85,
        physics_score=0.92,
        bbox=BoundingBox(xmin=100.0, ymin=120.0, xmax=160.0, ymax=180.0),
        geo_location=GeoPoint(latitude=18.9225, longitude=72.8350, depth_m=22.0, ground_range_m=15.4),
        dimensions=PhysicalDimensions(length_m=3.0, width_m=2.5, estimated_height_m=1.2, area_m2=7.5),
        acoustic_signature=AcousticSignature(highlight_mean=0.88, shadow_mean=0.03, contrast_ratio=3.4),
        status=AuditStatus.HIGH_CONFIDENCE
    )


def test_json_reporting(sample_detection):
    report = MissionReport(
        metadata=MissionMetadata(mission_id="test_mission"),
        config=PipelineConfig(),
        summary={"total_detections": 1},
        detections=[sample_detection],
        audit_log=[],
        nav_track=[]
    )
    json_str = SonarReporter.generate_json_report(report)
    parsed = json.loads(json_str)
    assert parsed["metadata"]["mission_id"] == "test_mission"
    assert len(parsed["detections"]) == 1
    assert parsed["detections"][0]["class_name"] == "ghost_net"


def test_csv_reporting(sample_detection):
    csv_str = SonarReporter.generate_csv_report([sample_detection])
    lines = csv_str.strip().splitlines()
    assert len(lines) == 2  # header + 1 row
    assert "det_001" in lines[1]
    assert "ghost_net" in lines[1]


def test_geojson_reporting(sample_detection):
    geojson = SonarReporter.generate_geojson_report([sample_detection])
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1
    feat = geojson["features"][0]
    assert feat["geometry"]["type"] == "Point"
    assert feat["geometry"]["coordinates"] == [72.8350, 18.9225]
    assert feat["properties"]["class"] == "ghost_net"
