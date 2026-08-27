import pytest
import numpy as np

from sonar_debris.pipeline import SonarDebrisPipeline
from sonar_debris.types import PipelineConfig, DebrisClass
from sonar_debris.models.synthetic_generator import SyntheticSonarGenerator


def test_full_pipeline_run(tmp_path):
    # 1. Generate synthetic mission
    gen = SyntheticSonarGenerator(image_width=256, image_height=256, seed=42)
    img, targets, nav = gen.generate_mission(num_targets=4)

    # 2. Configure pipeline
    config = PipelineConfig(
        confidence_threshold_percent=50.0,
        tile_size=256,
        tile_overlap=32,
        export_thumbnails=True
    )
    pipeline = SonarDebrisPipeline(config=config)

    # 3. Run pipeline
    out_dir = str(tmp_path / "mission_out")
    report = pipeline.run(
        sonar_input=img,
        nav_input=None,
        mission_id="pytest_mission",
        survey_name="Pytest SSS Mission",
        output_dir=out_dir
    )

    assert report.metadata.mission_id == "pytest_mission"
    assert "total_detections" in report.summary
    assert report.summary["processing_time_sec"] > 0
    assert len(report.detections) + len(report.audit_log) > 0

    # Verify georeferenced coordinates
    for d in report.detections:
        assert -90.0 <= d.geo_location.latitude <= 90.0
        assert -180.0 <= d.geo_location.longitude <= 180.0
        assert d.dimensions.length_m > 0
        assert d.dimensions.width_m > 0
        assert 0.0 <= d.confidence_percent <= 100.0
