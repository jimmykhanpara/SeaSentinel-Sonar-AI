"""
Physics-Based Acoustic Shadow & Highlight Analyzer
===================================================
Distinguishes artificial marine debris from natural seabed clutter (rocks,
sand ripples, seafloor depressions) through acoustic shadow geometry,
directional nadir constraints, and highlight-to-shadow contrast ratios.
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.ndimage import sobel

from ..types import BoundingBox, AcousticSignature, ChannelType


class AcousticShadowAnalyzer:
    """
    Evaluates acoustic physical plausibility of detected anomaly candidates.
    """

    def __init__(
        self,
        min_contrast_ratio: float = 1.5,
        min_shadow_sharpness: float = 0.12,
        weight_contrast: float = 0.35,
        weight_direction: float = 0.40,
        weight_sharpness: float = 0.25
    ):
        self.min_contrast_ratio = float(min_contrast_ratio)
        self.min_shadow_sharpness = float(min_shadow_sharpness)
        self.w_contrast = float(weight_contrast)
        self.w_direction = float(weight_direction)
        self.w_sharpness = float(weight_sharpness)

    def analyze_candidate(
        self,
        sonar_img: np.ndarray,
        bbox: BoundingBox,
        towfish_altitude_m: float = 10.0,
        max_slant_range_m: float = 50.0
    ) -> Tuple[float, AcousticSignature, bool]:
        """
        Analyzes a candidate detection bounding box on the full sonar waterfall.

        Args:
            sonar_img: 2D normalized sonar image [0, 1].
            bbox: Bounding box in global image coordinates.
            towfish_altitude_m: Tow-fish altitude above seafloor.
            max_slant_range_m: Maximum sonar slant range in meters.

        Returns:
            Tuple of (physics_score [0..1], acoustic_signature, is_valid_physics)
        """
        h, w = sonar_img.shape[:2]
        center_x = w / 2.0

        # Pixel boundaries
        x1 = int(np.clip(bbox.xmin, 0, w - 1))
        y1 = int(np.clip(bbox.ymin, 0, h - 1))
        x2 = int(np.clip(bbox.xmax, 0, w))
        y2 = int(np.clip(bbox.ymax, 0, h))

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            # Too small
            sig = AcousticSignature()
            return 0.2, sig, False

        patch = sonar_img[y1:y2, x1:x2]
        obj_center_x = (x1 + x2) / 2.0
        channel = "port" if obj_center_x < center_x else "starboard"

        # Background seabed estimation from local surroundings
        pad_y1 = max(0, y1 - 20)
        pad_y2 = min(h, y2 + 20)
        pad_x1 = max(0, x1 - 20)
        pad_x2 = min(w, x2 + 20)
        surrounding = sonar_img[pad_y1:pad_y2, pad_x1:pad_x2]
        seabed_mean = float(np.median(surrounding))

        # 1. Segment Highlight vs Shadow inside patch
        p_med = np.median(patch)
        p_max = np.max(patch)
        p_min = np.min(patch)

        # Highlight pixels: top 25% brightest
        highlight_thresh = max(0.60, float(np.percentile(patch, 75)))
        hl_pixels = patch[patch >= highlight_thresh]
        hl_mean = float(np.mean(hl_pixels)) if hl_pixels.size > 0 else float(p_max)

        # Shadow pixels: bottom 25% darkest
        shadow_thresh = min(0.15, float(np.percentile(patch, 25)))
        s_pixels = patch[patch <= shadow_thresh]
        s_mean = float(np.mean(s_pixels)) if s_pixels.size > 0 else float(p_min)

        # 2. Contrast Ratio
        contrast_ratio = (hl_mean - s_mean) / max(0.05, seabed_mean)
        contrast_score = np.clip((contrast_ratio - 1.0) / 3.0, 0.0, 1.0)

        # 3. Directional Alignment (Acoustic shadow MUST cast away from nadir)
        # Split patch into left half and right half
        pw = patch.shape[1]
        left_half = patch[:, :pw // 2]
        right_half = patch[:, pw // 2:]

        left_darkness = float(np.mean(left_half < shadow_thresh)) if left_half.size > 0 else 0.0
        right_darkness = float(np.mean(right_half < shadow_thresh)) if right_half.size > 0 else 0.0

        if channel == "port":
            # Shadow should be on the LEFT side of the highlight (away from nadir at center)
            if left_darkness > right_darkness * 1.1:
                dir_score = 1.0
            elif abs(left_darkness - right_darkness) < 0.15:
                dir_score = 0.6  # Ambiguous / centered
            else:
                dir_score = 0.2  # Shadow points toward nadir (physically impossible for acoustic reflection)
        else:  # starboard
            # Shadow should be on the RIGHT side of the highlight
            if right_darkness > left_darkness * 1.1:
                dir_score = 1.0
            elif abs(left_darkness - right_darkness) < 0.15:
                dir_score = 0.6
            else:
                dir_score = 0.2

        # 4. Shadow Edge Sharpness (gradient magnitude at highlight-shadow transition)
        grad_x = sobel(patch, axis=1)
        grad_y = sobel(patch, axis=0)
        grad_mag = np.hypot(grad_x, grad_y)
        edge_sharpness = float(np.percentile(grad_mag, 85))
        sharpness_score = np.clip(edge_sharpness / 0.5, 0.0, 1.0)

        # Shadow length estimation in pixels
        shadow_mask = patch <= shadow_thresh
        shadow_cols = np.where(np.any(shadow_mask, axis=0))[0]
        shadow_len_px = float(len(shadow_cols))

        # Composite Physics Score
        physics_score = (
            self.w_contrast * contrast_score +
            self.w_direction * dir_score +
            self.w_sharpness * sharpness_score
        )
        physics_score = float(np.clip(physics_score, 0.0, 1.0))

        is_valid = (contrast_ratio >= self.min_contrast_ratio) and (dir_score >= 0.4)

        sig = AcousticSignature(
            highlight_mean=round(hl_mean, 3),
            shadow_mean=round(s_mean, 3),
            seabed_mean=round(seabed_mean, 3),
            contrast_ratio=round(contrast_ratio, 2),
            shadow_length_px=round(shadow_len_px, 1),
            shadow_edge_sharpness=round(edge_sharpness, 3),
            direction_alignment_score=round(dir_score, 2)
        )

        return physics_score, sig, is_valid
