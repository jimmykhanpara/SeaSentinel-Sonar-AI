"""
Physics-Based Synthetic Side Scan Sonar (SSS) Generator
======================================================
Generates photorealistic acoustic seabed imagery, seafloor backscatter textures,
nadir water column gaps, speckle noise, and target signatures (ghost nets,
shipwrecks, pipes, containers, tires, and natural rock clutter) with physically
accurate acoustic shadow casting based on tow-fish altitude and grazing angle.
"""

from __future__ import annotations
import math
import random
import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.ndimage import gaussian_filter

from ..types import DebrisClass, BoundingBox, NavigationPoint, GeoPoint, PhysicalDimensions


class SyntheticSonarGenerator:
    """
    Simulates acoustic backscatter and target occlusion for Side Scan Sonar missions.
    """

    def __init__(
        self,
        image_width: int = 1024,
        image_height: int = 1024,
        max_slant_range_m: float = 50.0,
        altitude_m: float = 10.0,
        seed: Optional[int] = None
    ):
        self.width = int(image_width)
        self.height = int(image_height)
        self.max_slant_range_m = float(max_slant_range_m)
        self.altitude_m = float(altitude_m)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate_mission(
        self,
        num_targets: int = 6,
        include_ghost_nets: bool = True,
        include_wrecks: bool = True,
        include_pipes: bool = True,
        include_rock_clutter: bool = True,
        start_lat: float = 18.9220,
        start_lon: float = 72.8346,
        heading_deg: float = 45.0
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[NavigationPoint]]:
        """
        Generates a complete synthetic survey strip with ground-truth targets and navigation track.

        Returns:
            Tuple of (sonar_image, ground_truth_targets, nav_points)
        """
        # 1. Base seafloor backscatter
        img = self._generate_seabed_background()

        # 2. Add nadir water column
        img = self._apply_nadir_column(img)

        # 3. Add targets and cast acoustic shadows
        targets: List[Dict[str, Any]] = []
        center_x = self.width / 2.0

        available_classes = []
        if include_ghost_nets:
            available_classes.extend([DebrisClass.GHOST_NET] * 3)
        if include_wrecks:
            available_classes.extend([DebrisClass.SHIPWRECK] * 2)
        if include_pipes:
            available_classes.extend([DebrisClass.PIPE_CYLINDER] * 2)
            available_classes.extend([DebrisClass.CONTAINER, DebrisClass.TIRE, DebrisClass.GENERIC_DEBRIS])
        if include_rock_clutter:
            available_classes.extend([DebrisClass.ROCK_CLUTTER] * 3)

        if not available_classes:
            available_classes = [DebrisClass.GHOST_NET]

        # Margin away from nadir and edges
        min_x_port = int(self.width * 0.08)
        max_x_port = int(center_x - self.width * 0.08)
        min_x_stbd = int(center_x + self.width * 0.08)
        max_x_stbd = int(self.width * 0.92)

        placed_boxes = []

        scale = max(0.25, self.width / 1024.0)

        for _ in range(num_targets):
            cls = random.choice(available_classes)
            # Choose port or starboard
            if random.random() < 0.5:
                tx = random.randint(min_x_port, max_x_port)
                channel = "port"
            else:
                tx = random.randint(min_x_stbd, max_x_stbd)
                channel = "starboard"

            ty = random.randint(int(self.height * 0.08), int(self.height * 0.92))

            # Target size scaled to image resolution
            if cls == DebrisClass.SHIPWRECK:
                size_x = int(random.randint(60, 110) * scale)
                size_y = int(random.randint(70, 140) * scale)
            elif cls == DebrisClass.GHOST_NET:
                size_x = int(random.randint(40, 85) * scale)
                size_y = int(random.randint(45, 95) * scale)
            elif cls == DebrisClass.PIPE_CYLINDER:
                size_x = int(random.randint(20, 45) * scale)
                size_y = int(random.randint(70, 130) * scale)
            elif cls == DebrisClass.CONTAINER:
                size_x = int(random.randint(35, 60) * scale)
                size_y = int(random.randint(50, 80) * scale)
            elif cls == DebrisClass.TIRE:
                size_x = int(random.randint(25, 45) * scale)
                size_y = int(random.randint(25, 45) * scale)
            elif cls == DebrisClass.ROCK_CLUTTER:
                size_x = int(random.randint(30, 60) * scale)
                size_y = int(random.randint(30, 60) * scale)
            else:
                size_x = int(random.randint(30, 60) * scale)
                size_y = int(random.randint(30, 60) * scale)

            size_x = max(12, size_x)
            size_y = max(12, size_y)

            # Avoid tight overlaps
            x1 = max(0, tx - size_x // 2)
            y1 = max(0, ty - size_y // 2)
            x2 = min(self.width, tx + size_x // 2)
            y2 = min(self.height, ty + size_y // 2)

            overlap = False
            for bx1, by1, bx2, by2 in placed_boxes:
                if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):
                    overlap = True
                    break
            if overlap:
                continue

            placed_boxes.append((x1, y1, x2, y2))

            # Render target and shadow on image
            target_meta = self._render_target(img, cls, tx, ty, size_x, size_y, channel)
            if target_meta is not None:
                targets.append(target_meta)

        # 4. Add multiplicative speckle noise
        speckle = np.random.gamma(shape=10.0, scale=0.1, size=img.shape).astype(np.float32)
        img = np.clip(img * speckle, 0.0, 1.0)

        # 5. Generate matching navigation track
        nav_points = self._generate_nav_track(
            total_pings=self.height,
            start_lat=start_lat,
            start_lon=start_lon,
            heading_deg=heading_deg
        )

        return img, targets, nav_points

    def _generate_seabed_background(self) -> np.ndarray:
        """Generates realistic acoustic seabed backscatter with sand ripples and micro-roughness."""
        # 1. Base Lambertian background intensity (~0.25-0.35)
        base = np.full((self.height, self.width), 0.28, dtype=np.float32)

        # 2. Perlin-like low frequency seafloor depth/sediment variations
        low_freq = np.random.randn(self.height // 32, self.width // 32).astype(np.float32)
        low_freq = gaussian_filter(low_freq, sigma=2.0)
        # Resize to full image
        from scipy.ndimage import zoom
        low_freq_full = zoom(low_freq, (self.height / low_freq.shape[0], self.width / low_freq.shape[1]), order=1)[:self.height, :self.width]
        base += low_freq_full * 0.08

        # 3. Directional sand ripples
        ripple_angle = random.uniform(0.2, 0.8)
        y_grid, x_grid = np.mgrid[0:self.height, 0:self.width]
        ripple_freq = random.uniform(0.08, 0.15)
        ripples = np.sin(x_grid * np.cos(ripple_angle) * ripple_freq + y_grid * np.sin(ripple_angle) * ripple_freq)
        base += ripples * 0.04

        # 4. Lateral acoustic intensity falloff with range (before TVG)
        center_x = self.width / 2.0
        dist_norm = np.abs(x_grid - center_x) / center_x
        # Falloff
        falloff = 1.0 - (dist_norm * 0.3)
        base = base * falloff

        return np.clip(base, 0.05, 0.95)

    def _apply_nadir_column(self, img: np.ndarray) -> np.ndarray:
        """Simulates the dark nadir water column at the center of the image."""
        center_x = self.width / 2.0
        # Nadir width corresponds to tow-fish altitude
        nadir_half_width_px = int((self.altitude_m / self.max_slant_range_m) * (self.width / 2.0) * 0.8)
        nadir_half_width_px = max(6, min(nadir_half_width_px, int(self.width * 0.15)))

        x1 = int(center_x - nadir_half_width_px)
        x2 = int(center_x + nadir_half_width_px)

        # Water column is dark with very low acoustic backscatter (~0.02)
        img[:, x1:x2] = np.random.uniform(0.01, 0.04, (self.height, x2 - x1)).astype(np.float32)

        # Bottom return bright specular highlight line at nadir borders
        if x1 > 0:
            img[:, max(0, x1 - 2):x1 + 1] = np.random.uniform(0.7, 0.95, (self.height, min(3, x1 + 1)))
        if x2 < self.width:
            img[:, x2:min(self.width, x2 + 3)] = np.random.uniform(0.7, 0.95, (self.height, min(3, self.width - x2)))

        return img

    def _render_target(
        self,
        img: np.ndarray,
        cls: DebrisClass,
        tx: int,
        ty: int,
        size_x: int,
        size_y: int,
        channel: str
    ) -> Optional[Dict[str, Any]]:
        """
        Draws acoustic highlight and cast shadow for a given target class.
        """
        center_x = self.width / 2.0
        dist_from_nadir = abs(tx - center_x)
        # Ground range in meters
        slant_r = (dist_from_nadir / (self.width / 2.0)) * self.max_slant_range_m
        ground_r = math.sqrt(max(1.0, slant_r**2 - self.altitude_m**2))

        # Shadow direction:
        # Port (tx < center_x) -> shadow points LEFT (-1)
        # Starboard (tx > center_x) -> shadow points RIGHT (+1)
        shadow_dir = -1 if channel == "port" else +1

        # Calculate physical object height and shadow length
        if cls == DebrisClass.SHIPWRECK:
            obj_height_m = random.uniform(2.5, 5.0)
            highlight_val = random.uniform(0.85, 1.0)
            shadow_val = random.uniform(0.01, 0.03)
        elif cls == DebrisClass.GHOST_NET:
            obj_height_m = random.uniform(1.2, 3.0)
            highlight_val = random.uniform(0.75, 0.95)
            shadow_val = random.uniform(0.02, 0.06)
        elif cls == DebrisClass.PIPE_CYLINDER:
            obj_height_m = random.uniform(0.8, 1.8)
            highlight_val = random.uniform(0.88, 1.0)
            shadow_val = random.uniform(0.01, 0.04)
        elif cls == DebrisClass.CONTAINER:
            obj_height_m = random.uniform(2.0, 3.2)
            highlight_val = random.uniform(0.9, 1.0)
            shadow_val = random.uniform(0.01, 0.03)
        elif cls == DebrisClass.TIRE:
            obj_height_m = random.uniform(0.5, 1.0)
            highlight_val = random.uniform(0.7, 0.85)
            shadow_val = random.uniform(0.02, 0.05)
        elif cls == DebrisClass.ROCK_CLUTTER:
            obj_height_m = random.uniform(0.5, 1.5)
            highlight_val = random.uniform(0.65, 0.8)
            shadow_val = random.uniform(0.08, 0.18)  # Diffuse/incomplete shadow
        else:
            obj_height_m = random.uniform(1.0, 2.0)
            highlight_val = random.uniform(0.75, 0.9)
            shadow_val = random.uniform(0.02, 0.05)

        # Acoustic shadow length formula: Ls = (Ho * Rg) / (h - Ho)
        denom = max(0.5, self.altitude_m - obj_height_m)
        shadow_len_m = (obj_height_m * ground_r) / denom
        # Convert to pixels
        m_per_px = self.max_slant_range_m / (self.width / 2.0)
        shadow_len_px = int(shadow_len_m / m_per_px)
        shadow_len_px = max(10, min(shadow_len_px, int(self.width * 0.25)))

        # Target bounding geometry
        hx1 = max(0, tx - size_x // 2)
        hy1 = max(0, ty - size_y // 2)
        hx2 = min(self.width, tx + size_x // 2)
        hy2 = min(self.height, ty + size_y // 2)

        # Draw highlight and shadow based on class
        if cls == DebrisClass.GHOST_NET:
            # Ghost nets have irregular, tangled webbing with trailing fibers
            self._draw_ghost_net(img, hx1, hy1, hx2, hy2, shadow_dir, shadow_len_px, highlight_val, shadow_val)
        elif cls == DebrisClass.SHIPWRECK:
            self._draw_shipwreck(img, hx1, hy1, hx2, hy2, shadow_dir, shadow_len_px, highlight_val, shadow_val)
        elif cls == DebrisClass.PIPE_CYLINDER:
            self._draw_pipe(img, hx1, hy1, hx2, hy2, shadow_dir, shadow_len_px, highlight_val, shadow_val)
        elif cls == DebrisClass.CONTAINER:
            self._draw_container(img, hx1, hy1, hx2, hy2, shadow_dir, shadow_len_px, highlight_val, shadow_val)
        elif cls == DebrisClass.TIRE:
            self._draw_tire(img, hx1, hy1, hx2, hy2, shadow_dir, shadow_len_px, highlight_val, shadow_val)
        elif cls == DebrisClass.ROCK_CLUTTER:
            self._draw_rock_clutter(img, hx1, hy1, hx2, hy2, shadow_dir, shadow_len_px, highlight_val, shadow_val)
        else:
            self._draw_generic_anomaly(img, hx1, hy1, hx2, hy2, shadow_dir, shadow_len_px, highlight_val, shadow_val)

        # Full bounding box enclosing highlight AND cast shadow
        if shadow_dir < 0:
            full_x1 = max(0, hx1 - shadow_len_px)
            full_x2 = hx2
        else:
            full_x1 = hx1
            full_x2 = min(self.width, hx2 + shadow_len_px)

        full_y1 = hy1
        full_y2 = hy2

        # Physical dimensions
        phys_length_m = size_y * m_per_px
        phys_width_m = size_x * m_per_px
        phys_area_m2 = phys_length_m * phys_width_m

        return {
            "class_name": cls,
            "channel": channel,
            "center_px": (tx, ty),
            "highlight_bbox": BoundingBox(xmin=float(hx1), ymin=float(hy1), xmax=float(hx2), ymax=float(hy2)),
            "full_bbox": BoundingBox(xmin=float(full_x1), ymin=float(full_y1), xmax=float(full_x2), ymax=float(full_y2)),
            "dimensions": PhysicalDimensions(
                length_m=round(phys_length_m, 2),
                width_m=round(phys_width_m, 2),
                estimated_height_m=round(obj_height_m, 2),
                area_m2=round(phys_area_m2, 2)
            ),
            "ground_range_m": round(ground_r, 2),
            "shadow_length_px": shadow_len_px,
            "shadow_direction": shadow_dir
        }

    def _draw_ghost_net(
        self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
        s_dir: int, s_len: int, h_val: float, s_val: float
    ):
        # 1. Cast shadow region
        if s_dir < 0:
            sx1, sx2 = max(0, x1 - s_len), x1
        else:
            sx1, sx2 = x2, min(self.width, x2 + s_len)
        img[y1:y2, sx1:sx2] = np.random.uniform(s_val * 0.8, s_val * 1.5, (y2 - y1, sx2 - sx1))

        # 2. Net highlight: mesh/lattice web structure + frayed filaments
        h_patch = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        # Random ropes/lines
        for _ in range(8):
            px1, py1 = random.randint(0, x2 - x1 - 1), random.randint(0, y2 - y1 - 1)
            px2, py2 = random.randint(0, x2 - x1 - 1), random.randint(0, y2 - y1 - 1)
            # Simple line raster
            num_pts = max(abs(px2 - px1), abs(py2 - py1), 2)
            xs = np.linspace(px1, px2, num_pts).astype(int)
            ys = np.linspace(py1, py2, num_pts).astype(int)
            h_patch[ys, xs] = h_val

        # Thicken net knots
        h_patch = gaussian_filter(h_patch, sigma=1.2) * 2.5
        h_patch = np.clip(h_patch, 0.0, h_val)
        mask = h_patch > 0.2
        img[y1:y2, x1:x2] = np.where(mask, h_patch, img[y1:y2, x1:x2])

    def _draw_shipwreck(
        self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
        s_dir: int, s_len: int, h_val: float, s_val: float
    ):
        # Sharp elongated acoustic shadow
        if s_dir < 0:
            sx1, sx2 = max(0, x1 - s_len), x1
        else:
            sx1, sx2 = x2, min(self.width, x2 + s_len)
        img[y1:y2, sx1:sx2] = np.random.uniform(s_val * 0.7, s_val * 1.2, (y2 - y1, sx2 - sx1))

        # Hull highlight: tapered bow/stern outline and structural frames
        hull = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        # Outline
        hull[:, 0:max(1, (x2 - x1) // 3)] = h_val
        hull[0:3, :] = h_val
        hull[-3:, :] = h_val
        # Internal bulkhead ribs
        for ry in range(4, y2 - y1 - 4, 12):
            hull[ry:ry + 2, :] = h_val * 0.9

        hull = np.clip(hull + np.random.uniform(-0.1, 0.1, hull.shape), 0.0, 1.0)
        img[y1:y2, x1:x2] = hull

    def _draw_pipe(
        self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
        s_dir: int, s_len: int, h_val: float, s_val: float
    ):
        # Cylindrical pipe has a continuous linear highlight and a uniform parallel shadow
        if s_dir < 0:
            sx1, sx2 = max(0, x1 - s_len), x1
        else:
            sx1, sx2 = x2, min(self.width, x2 + s_len)
        img[y1:y2, sx1:sx2] = s_val

        # Linear highlight
        pipe_w = max(2, (x2 - x1) // 2)
        img[y1:y2, x1:x1 + pipe_w] = h_val

    def _draw_container(
        self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
        s_dir: int, s_len: int, h_val: float, s_val: float
    ):
        if s_dir < 0:
            sx1, sx2 = max(0, x1 - s_len), x1
        else:
            sx1, sx2 = x2, min(self.width, x2 + s_len)
        img[y1:y2, sx1:sx2] = s_val

        img[y1:y2, x1:x2] = np.random.uniform(h_val * 0.9, h_val, (y2 - y1, x2 - x1))

    def _draw_tire(
        self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
        s_dir: int, s_len: int, h_val: float, s_val: float
    ):
        if s_dir < 0:
            sx1, sx2 = max(0, x1 - s_len), x1
        else:
            sx1, sx2 = x2, min(self.width, x2 + s_len)
        img[y1:y2, sx1:sx2] = s_val

        # Toroidal ring
        cy, cx = (y2 - y1) / 2.0, (x2 - x1) / 2.0
        y_g, x_g = np.ogrid[:y2 - y1, :x2 - x1]
        dist_from_c = np.sqrt((x_g - cx)**2 + (y_g - cy)**2)
        r_outer = min(cx, cy)
        r_inner = r_outer * 0.45
        ring_mask = (dist_from_c <= r_outer) & (dist_from_c >= r_inner)
        img[y1:y2, x1:x2] = np.where(ring_mask, h_val, img[y1:y2, x1:x2])

    def _draw_rock_clutter(
        self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
        s_dir: int, s_len: int, h_val: float, s_val: float
    ):
        # Natural rock clusters have amorphous, non-directional, jagged shadow and highlight
        patch_h = y2 - y1
        patch_w = x2 - x1
        num_rocks = random.randint(3, 8)
        for _ in range(num_rocks):
            rx = random.randint(0, patch_w - 4)
            ry = random.randint(0, patch_h - 4)
            rw = random.randint(3, 8)
            rh = random.randint(3, 8)
            img[y1 + ry:y1 + ry + rh, x1 + rx:x1 + rx + rw] = h_val
            # Short fuzzy shadow
            img[y1 + ry:y1 + ry + rh, max(0, x1 + rx - 5):x1 + rx] = s_val

    def _draw_generic_anomaly(
        self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
        s_dir: int, s_len: int, h_val: float, s_val: float
    ):
        if s_dir < 0:
            sx1, sx2 = max(0, x1 - s_len), x1
        else:
            sx1, sx2 = x2, min(self.width, x2 + s_len)
        img[y1:y2, sx1:sx2] = s_val
        img[y1:y2, x1:x2] = h_val

    def _generate_nav_track(
        self,
        total_pings: int,
        start_lat: float,
        start_lon: float,
        heading_deg: float
    ) -> List[NavigationPoint]:
        """Generates realistic vehicle navigation trajectory with slight realistic yaw/drift."""
        track: List[NavigationPoint] = []
        heading_rad = math.radians(heading_deg)
        # Vessel speed ~ 3.0 knots ~ 1.54 m/s
        speed_mps = 1.54
        dt = 0.1  # 10 Hz ping rate -> 0.154 m per ping
        curr_lat = start_lat
        curr_lon = start_lon
        curr_heading = heading_deg
        t0 = time.time() - (total_pings * dt)

        # 1 deg lat ~ 111,000 meters
        # 1 deg lon ~ 111,000 * cos(lat)
        m_to_lat = 1.0 / 111000.0
        m_to_lon = 1.0 / (111000.0 * math.cos(math.radians(start_lat)))

        for i in range(total_pings):
            # Minor yaw drift
            curr_heading += random.uniform(-0.15, 0.15)
            h_rad = math.radians(curr_heading)

            d_north = speed_mps * dt * math.cos(h_rad)
            d_east = speed_mps * dt * math.sin(h_rad)

            curr_lat += d_north * m_to_lat
            curr_lon += d_east * m_to_lon

            track.append(
                NavigationPoint(
                    timestamp=t0 + i * dt,
                    ping_number=i,
                    latitude=curr_lat,
                    longitude=curr_lon,
                    heading_deg=curr_heading % 360.0,
                    altitude_m=self.altitude_m + random.uniform(-0.1, 0.1),
                    speed_knots=3.0,
                    depth_m=20.0
                )
            )

        return track
