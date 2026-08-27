"""
Slant-Range to Ground-Range Geometric Correction
=================================================
Corrects the geometric distortion in Side Scan Sonar (SSS) data caused by
slant-range projection. Transforms raw acoustic returns into equidistant
ground-range pixels on the seafloor.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


class SlantRangeCorrector:
    """
    Performs slant-range to ground-range conversion:
    Rg = sqrt(max(0, Rs^2 - h^2))
    where Rs is slant range and h is the tow-fish altitude above the seabed.
    """

    def __init__(self, max_slant_range_m: float = 50.0, default_altitude_m: float = 10.0):
        self.max_slant_range_m = float(max_slant_range_m)
        self.default_altitude_m = float(default_altitude_m)

    def correct_image(
        self,
        sonar_img: np.ndarray,
        altitude_m: Optional[float] = None,
        fill_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies slant-range correction to a dual-channel (Port + Starboard) SSS image.

        Args:
            sonar_img: 2D numpy array of shape (height/pings, width/samples) or 3D (H, W, C).
            altitude_m: Tow-fish altitude in meters. If None, uses default_altitude_m.
            fill_value: Pixel value for nadir blind-zone or invalid samples.

        Returns:
            Tuple of (corrected_image, ground_range_axis_meters)
        """
        if altitude_m is None:
            altitude_m = self.default_altitude_m

        is_3d = sonar_img.ndim == 3
        if is_3d:
            h_pings, w_samples, channels = sonar_img.shape
            corrected_channels = []
            for c in range(channels):
                corr, ground_axis = self._correct_2d_channel(sonar_img[:, :, c], altitude_m, fill_value)
                corrected_channels.append(corr)
            return np.stack(corrected_channels, axis=-1), ground_axis
        else:
            return self._correct_2d_channel(sonar_img, altitude_m, fill_value)

    def _correct_2d_channel(
        self,
        img: np.ndarray,
        altitude_m: float,
        fill_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        h_pings, w_samples = img.shape
        center_x = w_samples / 2.0
        samples_per_channel = int(w_samples // 2)

        if samples_per_channel <= 0:
            return img.copy(), np.zeros(w_samples)

        # Max ground range possible: sqrt(Rs_max^2 - h^2)
        if self.max_slant_range_m > altitude_m:
            max_ground_range_m = np.sqrt(self.max_slant_range_m**2 - altitude_m**2)
        else:
            max_ground_range_m = self.max_slant_range_m

        # Ground range grid for one channel (0 to max_ground_range_m)
        ground_grid_m = np.linspace(0.0, max_ground_range_m, samples_per_channel)

        # Corresponding slant range Rs = sqrt(Rg^2 + h^2)
        slant_grid_m = np.sqrt(ground_grid_m**2 + altitude_m**2)

        # Map slant range in meters to slant sample index [0, samples_per_channel - 1]
        slant_indices = (slant_grid_m / self.max_slant_range_m) * (samples_per_channel - 1)
        slant_indices = np.clip(slant_indices, 0, samples_per_channel - 1)

        # Correct Port Channel (left half: reversed from center to left)
        port_raw = img[:, :samples_per_channel]  # column 0 is far port, column end is nadir
        # In standard layout, column (samples_per_channel-1) is nadir, 0 is far port
        port_from_nadir = port_raw[:, ::-1]  # index 0 is nadir, index end is far port
        port_corrected_from_nadir = np.zeros_like(port_raw)

        # Correct Starboard Channel (right half: index 0 is nadir, index end is far stbd)
        stbd_raw = img[:, samples_per_channel:2 * samples_per_channel]
        stbd_corrected = np.zeros_like(stbd_raw)

        idx_floor = np.floor(slant_indices).astype(int)
        idx_ceil = np.clip(idx_floor + 1, 0, samples_per_channel - 1)
        weight_ceil = slant_indices - idx_floor
        weight_floor = 1.0 - weight_ceil

        for ping in range(h_pings):
            # Port interpolation
            p_row = port_from_nadir[ping, :]
            port_interp = p_row[idx_floor] * weight_floor + p_row[idx_ceil] * weight_ceil
            port_corrected_from_nadir[ping, :] = port_interp

            # Starboard interpolation
            s_row = stbd_raw[ping, :]
            stbd_interp = s_row[idx_floor] * weight_floor + s_row[idx_ceil] * weight_ceil
            stbd_corrected[ping, :] = stbd_interp

        # Recombine: port (far left to nadir) + starboard (nadir to far right)
        port_corrected = port_corrected_from_nadir[:, ::-1]
        full_corrected = np.hstack([port_corrected, stbd_corrected])

        # Full ground axis in meters (-max to +max)
        full_ground_axis = np.concatenate([-ground_grid_m[::-1], ground_grid_m])

        return full_corrected, full_ground_axis

    def pixel_to_ground_range(
        self,
        pixel_x: float,
        image_width: int,
        altitude_m: Optional[float] = None
    ) -> Tuple[float, str]:
        """
        Converts a horizontal pixel x-coordinate to lateral ground range in meters.

        Returns:
            (ground_range_m, channel_name) where ground_range_m >= 0 and channel_name is 'port' or 'starboard'
        """
        if altitude_m is None:
            altitude_m = self.default_altitude_m

        center_x = image_width / 2.0
        samples_per_channel = image_width / 2.0

        offset = pixel_x - center_x
        if offset < 0:
            channel = "port"
            norm_dist = abs(offset) / samples_per_channel
        else:
            channel = "starboard"
            norm_dist = offset / samples_per_channel

        slant_r = norm_dist * self.max_slant_range_m
        if slant_r <= altitude_m:
            ground_r = 0.0
        else:
            ground_r = np.sqrt(slant_r**2 - altitude_m**2)

        return float(ground_r), channel
