import pytest
import numpy as np

from sonar_debris.preprocessing.slant_range import SlantRangeCorrector
from sonar_debris.preprocessing.filters import (
    adaptive_lee_filter,
    enhanced_lee_filter,
    time_varying_gain_correction,
    detect_nadir_and_dropouts,
    normalize_sonar_image
)
from sonar_debris.preprocessing.ingest import NavigationParser, TilingEngine
from sonar_debris.types import BoundingBox


def test_slant_range_correction():
    corrector = SlantRangeCorrector(max_slant_range_m=50.0, default_altitude_m=10.0)
    raw = np.random.uniform(0.1, 0.9, (100, 200)).astype(np.float32)

    corrected, ground_axis = corrector.correct_image(raw, altitude_m=10.0)

    assert corrected.shape == (100, 200)
    assert len(ground_axis) == 200
    assert ground_axis[0] < 0  # Port is negative
    assert ground_axis[-1] > 0  # Starboard is positive

    # Test pixel to ground range mapping
    ground_r, channel = corrector.pixel_to_ground_range(20, 200, altitude_m=10.0)
    assert channel == "port"
    assert ground_r > 0.0

    ground_r_stbd, channel_stbd = corrector.pixel_to_ground_range(180, 200, altitude_m=10.0)
    assert channel_stbd == "starboard"
    assert ground_r_stbd > 0.0


def test_adaptive_lee_filter():
    img = np.full((64, 64), 0.5, dtype=np.float32)
    # Add speckle noise
    noisy = img + np.random.normal(0, 0.1, (64, 64)).astype(np.float32)
    filtered = adaptive_lee_filter(noisy, window_size=5)

    assert filtered.shape == (64, 64)
    assert np.var(filtered) < np.var(noisy)


def test_time_varying_gain():
    img = np.full((50, 100), 0.3, dtype=np.float32)
    tvg_img = time_varying_gain_correction(img)
    assert tvg_img.shape == (50, 100)
    # Outer columns should receive higher gain than nadir
    assert tvg_img[25, 95] > tvg_img[25, 50]


def test_nadir_detection():
    img = np.full((100, 200), 0.4, dtype=np.float32)
    # Simulate nadir water column in center
    img[:, 95:105] = 0.02
    nadir_mask, dropout_mask = detect_nadir_and_dropouts(img)

    assert nadir_mask.shape == (100, 200)
    assert np.any(nadir_mask[:, 98:102])


def test_navigation_parser_and_interpolation():
    csv_text = """ping,lat,lon,heading,altitude,speed,depth
0,18.92200,72.83460,45.0,10.0,3.0,20.0
10,18.92250,72.83510,45.0,10.0,3.0,20.0
"""
    nav_pts = NavigationParser.parse_csv(csv_text)
    assert len(nav_pts) == 2

    # Interpolate to 100 pings
    interp = NavigationParser.interpolate_navigation(nav_pts, total_pings=100)
    assert len(interp) == 100
    assert interp[0].latitude == pytest.approx(18.92200, rel=1e-4)
    assert interp[-1].latitude == pytest.approx(18.92250, rel=1e-4)


def test_tiling_engine():
    tiler = TilingEngine(tile_size=256, tile_overlap=32)
    img = np.zeros((600, 600), dtype=np.float32)
    tiles = tiler.create_tiles(img)

    assert len(tiles) >= 4
    for tile_arr, coords in tiles:
        assert tile_arr.shape == (256, 256)
        x1, y1, x2, y2 = coords
        assert x2 <= 600 and y2 <= 600

    # Global bbox mapping
    local_box = BoundingBox(xmin=10, ymin=20, xmax=50, ymax=60)
    global_box = TilingEngine.map_bbox_to_global(local_box, (100, 200))
    assert global_box.xmin == 110
    assert global_box.ymin == 220
