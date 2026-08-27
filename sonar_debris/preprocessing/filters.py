"""
Acoustic Image Filtering & Preprocessing
========================================
Implements Adaptive Lee filtering, Enhanced Lee filtering, Time-Varying Gain (TVG)
compensation, nadir blind-zone masking, and motion artifact dropout detection.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter
from typing import Tuple, Optional


def adaptive_lee_filter(
    img: np.ndarray,
    window_size: int = 7,
    noise_variance: Optional[float] = None
) -> np.ndarray:
    """
    Adaptive Lee Filter for acoustic speckle noise reduction.
    Preserves sharp highlight and shadow boundaries of man-made objects.

    Formula:
        W = max(0, (sigma^2 - sigma_n^2) / sigma^2)
        filtered = mean + W * (pixel - mean)

    Args:
        img: 2D float array with pixel values in [0, 1] or [0, 255].
        window_size: Size of local estimation window (must be odd).
        noise_variance: Estimated speckle noise variance. If None, estimated automatically.

    Returns:
        Filtered 2D array of same shape and dtype.
    """
    orig_dtype = img.dtype
    img_f = img.astype(np.float32)

    # Local mean
    mean = uniform_filter(img_f, size=window_size)

    # Local variance
    mean_sq = uniform_filter(img_f**2, size=window_size)
    var = np.maximum(0.0, mean_sq - mean**2)

    # Estimate noise variance from homogeneous dark/low-var regions if not provided
    if noise_variance is None:
        # 10th percentile of local variance as baseline speckle variance
        noise_variance = float(np.percentile(var, 15)) + 1e-6

    # Weighting coefficient
    weight = np.maximum(0.0, (var - noise_variance) / (var + 1e-6))
    weight = np.clip(weight, 0.0, 1.0)

    # Filtered output
    filtered = mean + weight * (img_f - mean)

    if np.issubdtype(orig_dtype, np.integer):
        return np.clip(filtered, 0, 255).astype(orig_dtype)
    return filtered.astype(orig_dtype)


def enhanced_lee_filter(
    img: np.ndarray,
    window_size: int = 7,
    damping_factor: float = 1.0
) -> np.ndarray:
    """
    Enhanced Lee Filter with multi-region classification (homogeneous, heterogeneous, point target).
    """
    orig_dtype = img.dtype
    img_f = img.astype(np.float32)

    mean = uniform_filter(img_f, size=window_size)
    mean_sq = uniform_filter(img_f**2, size=window_size)
    var = np.maximum(0.0, mean_sq - mean**2)
    std = np.sqrt(var)

    # Coefficient of variation
    ci = std / (mean + 1e-6)
    cu = float(np.percentile(ci, 10)) + 1e-4  # Noise variation in homogeneous area
    c_max = np.sqrt(1.0 + 2.0 / (window_size**2)) * cu  # Point target threshold

    weight = np.exp(-damping_factor * (ci - cu) / (c_max - ci + 1e-6))
    weight = np.clip(weight, 0.0, 1.0)

    # Homogeneous areas (ci <= cu) -> mean
    # Point targets (ci >= c_max) -> original pixel
    # Intermediate -> weighted
    result = np.where(ci <= cu, mean, np.where(ci >= c_max, img_f, mean * weight + img_f * (1.0 - weight)))

    if np.issubdtype(orig_dtype, np.integer):
        return np.clip(result, 0, 255).astype(orig_dtype)
    return result.astype(orig_dtype)


def time_varying_gain_correction(
    img: np.ndarray,
    absorption_coeff: float = 0.005,
    spreading_coeff: float = 1.0
) -> np.ndarray:
    """
    Applies Time-Varying Gain (TVG) compensation across the sonar swath to equalize
    acoustic intensity attenuation with slant range.

    Args:
        img: 2D image (H pings, W samples).
        absorption_coeff: Seawater acoustic absorption attenuation factor.
        spreading_coeff: Spherical spreading geometric loss exponent.

    Returns:
        Gain-equalized image.
    """
    orig_dtype = img.dtype
    img_f = img.astype(np.float32)
    h, w = img_f.shape
    center_x = w / 2.0

    # Distance from nadir for each column [0..1]
    col_indices = np.arange(w)
    dist_from_nadir = np.abs(col_indices - center_x) / (center_x + 1e-6)  # [0, 1]

    # TVG gain curve: compensates for spreading (R^gamma) and absorption exp(alpha * R)
    # Range is 0 at nadir, 1 at max range
    gain_curve = (1.0 + dist_from_nadir * spreading_coeff) * np.exp(absorption_coeff * dist_from_nadir * 50.0)
    gain_curve = gain_curve / (np.median(gain_curve) + 1e-6)

    # Broadcast across all pings
    corrected = img_f * gain_curve[np.newaxis, :]

    if np.issubdtype(orig_dtype, np.integer):
        return np.clip(corrected, 0, 255).astype(orig_dtype)
    return corrected.astype(orig_dtype)


def detect_nadir_and_dropouts(
    img: np.ndarray,
    threshold_energy: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects nadir water column gap and horizontal ping dropout lines.

    Args:
        img: 2D sonar image.
        threshold_energy: Threshold relative to median intensity.

    Returns:
        Tuple of (nadir_mask, dropout_mask) where True indicates invalid/water-column pixels.
    """
    img_f = img.astype(np.float32)
    h, w = img_f.shape
    center_x = int(w // 2)

    # 1. Nadir column estimation
    # Average intensity per column
    col_means = np.mean(img_f, axis=0)
    bg_level = np.median(col_means)

    # Look around center for low-energy water column
    nadir_mask = np.zeros_like(img, dtype=bool)
    search_half_width = int(w * 0.15)
    center_slice = col_means[center_x - search_half_width: center_x + search_half_width]

    low_val_cols = np.where(center_slice < bg_level * 0.4)[0]
    if len(low_val_cols) > 0:
        nadir_start = center_x - search_half_width + low_val_cols[0]
        nadir_end = center_x - search_half_width + low_val_cols[-1]
        nadir_mask[:, nadir_start:nadir_end + 1] = True

    # 2. Ping dropout detection (rows with zero or near-zero total acoustic energy)
    row_means = np.mean(img_f, axis=1)
    dropout_rows = np.where(row_means < bg_level * 0.1)[0]
    dropout_mask = np.zeros_like(img, dtype=bool)
    if len(dropout_rows) > 0:
        dropout_mask[dropout_rows, :] = True

    return nadir_mask, dropout_mask


def normalize_sonar_image(
    img: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0
) -> np.ndarray:
    """
    Robust percentile contrast stretching to canonical [0, 1] float range.
    """
    img_f = img.astype(np.float32)
    v_min = np.percentile(img_f, p_low)
    v_max = np.percentile(img_f, p_high)

    if v_max <= v_min:
        v_max = v_min + 1e-6

    stretched = np.clip((img_f - v_min) / (v_max - v_min), 0.0, 1.0)
    return stretched
