import pytest
import math
from sonar_debris.geotagging.georeferencer import SonarGeoreferencer
from sonar_debris.types import BoundingBox, NavigationPoint


def test_georeferencing_calculation():
    georef = SonarGeoreferencer(max_slant_range_m=50.0)

    # Vessel at 18.9220 N, 72.8346 E heading 0 deg (North), altitude 10m
    nav = [
        NavigationPoint(
            ping_number=i,
            latitude=18.9220,
            longitude=72.8346,
            heading_deg=0.0,
            altitude_m=10.0,
            speed_knots=3.0,
            depth_m=20.0
        )
        for i in range(100)
    ]

    # Target on Starboard channel (right half of image: x=150 in 200px width)
    bbox_stbd = BoundingBox(xmin=140, ymin=40, xmax=160, ymax=60)
    geo_pt_stbd, dims_stbd = georef.geotag_detection(
        bbox=bbox_stbd,
        image_shape=(100, 200),
        nav_track=nav,
        shadow_len_px=20.0
    )

    # When heading North (0 deg), Starboard is East (90 deg)
    # Longitude should increase, latitude should stay roughly unchanged
    assert geo_pt_stbd.longitude > 72.8346
    assert geo_pt_stbd.latitude == pytest.approx(18.9220, abs=1e-4)
    assert dims_stbd.length_m > 0.0
    assert dims_stbd.width_m > 0.0
    assert dims_stbd.estimated_height_m > 0.0

    # Target on Port channel (left half: x=50)
    bbox_port = BoundingBox(xmin=40, ymin=40, xmax=60, ymax=60)
    geo_pt_port, dims_port = georef.geotag_detection(
        bbox=bbox_port,
        image_shape=(100, 200),
        nav_track=nav,
        shadow_len_px=20.0
    )

    # Port is West (270 deg) -> Longitude should decrease
    assert geo_pt_port.longitude < 72.8346
