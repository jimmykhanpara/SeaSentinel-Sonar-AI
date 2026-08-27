"""
Deep Neural Architecture for Sonar Marine Debris Detection & Segmentation
========================================================================
Implements SSSDebrisNet (PyTorch lightweight ResNet-FPN dual-head architecture)
optimized for acoustic highlight-shadow anomaly detection and segmentation.
"""

from __future__ import annotations
import abc
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..types import DebrisClass, BoundingBox, ChannelType


class BaseSonarDetector(abc.ABC):
    """Abstract interface for side-scan sonar detection models."""

    @abc.abstractmethod
    def predict(
        self,
        image_tile: np.ndarray,
        conf_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Runs inference on an image tile (H, W).

        Returns:
            List of detected objects:
            [
              {
                "class_name": DebrisClass,
                "confidence": float (0.0-1.0),
                "bbox": BoundingBox (local tile coords),
                "mask": Optional[np.ndarray],
                "polygon": Optional[List[Tuple[float, float]]]
              }
            ]
        """
        pass


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + SiLU/Mish activation."""
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual bottleneck block with SE acoustic channel attention."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, 3, 1, 1)
        self.conv2 = ConvBlock(channels, channels, 3, 1, 1)
        # Squeeze-and-Excitation (SE) attention
        self.se_fc1 = nn.Conv2d(channels, max(8, channels // 4), 1)
        self.se_fc2 = nn.Conv2d(max(8, channels // 4), channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.conv2(self.conv1(x))
        # SE
        se = F.adaptive_avg_pool2d(out, 1)
        se = F.relu(self.se_fc1(se), inplace=True)
        se = torch.sigmoid(self.se_fc2(se))
        out = out * se
        return out + res


class SSSDebrisNet(nn.Module):
    """
    Lightweight Feature Pyramid Network for Side Scan Sonar debris detection & segmentation.
    Operates directly on 1-channel grayscale acoustic backscatter images.
    """
    CLASS_NAMES = [
        DebrisClass.GHOST_NET,
        DebrisClass.SHIPWRECK,
        DebrisClass.PIPE_CYLINDER,
        DebrisClass.CONTAINER,
        DebrisClass.TIRE,
        DebrisClass.GENERIC_DEBRIS,
        DebrisClass.ROCK_CLUTTER
    ]

    def __init__(self, in_channels: int = 1, num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes

        # Stem: 1 -> 32
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 32, 3, stride=2, padding=1),  # H/2, W/2
            ConvBlock(32, 32, 3, stride=1, padding=1)
        )

        # Stage 1: H/2 -> H/4 (32 -> 64)
        self.stage1 = nn.Sequential(
            ConvBlock(32, 64, 3, stride=2, padding=1),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        # Stage 2: H/4 -> H/8 (64 -> 128)
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, 3, stride=2, padding=1),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        # Stage 3: H/8 -> H/16 (128 -> 256)
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, 3, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # FPN Lateral Convs: project all to 64 channels
        self.lat3 = nn.Conv2d(256, 64, 1)
        self.lat2 = nn.Conv2d(128, 64, 1)
        self.lat1 = nn.Conv2d(64, 64, 1)

        # FPN Smooth Convs
        self.smooth3 = ConvBlock(64, 64, 3, 1, 1)
        self.smooth2 = ConvBlock(64, 64, 3, 1, 1)
        self.smooth1 = ConvBlock(64, 64, 3, 1, 1)

        # Detection Head (Anchorless CenterNet/FCOS style) on P1 (stride 4)
        self.cls_head = nn.Conv2d(64, num_classes, 1)
        self.box_head = nn.Conv2d(64, 4, 1)       # (dx_left, dy_top, dx_right, dy_bottom)
        self.obj_head = nn.Conv2d(64, 1, 1)       # Objectness / centerness

        # Segmentation Mask Head (upsamples P1 to input resolution)
        self.mask_head = nn.Sequential(
            ConvBlock(64, 32, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # H/2
            ConvBlock(32, 16, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # H
            nn.Conv2d(16, num_classes + 1, 1)  # classes + background
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            cls_logits: (B, num_classes, H/4, W/4)
            box_preds: (B, 4, H/4, W/4)
            obj_logits: (B, 1, H/4, W/4)
            mask_logits: (B, num_classes + 1, H, W)
        """
        # Encoder
        c0 = self.stem(x)     # H/2, 32
        c1 = self.stage1(c0)  # H/4, 64
        c2 = self.stage2(c1)  # H/8, 128
        c3 = self.stage3(c2)  # H/16, 256

        # FPN Top-down
        p3 = self.lat3(c3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest")
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="nearest")

        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)  # (B, 64, H/4, W/4)

        # Heads
        cls_logits = self.cls_head(p1)
        box_preds = F.relu(self.box_head(p1)) * 64.0  # Scale distance offsets
        obj_logits = self.obj_head(p1)
        mask_logits = self.mask_head(p1)

        return cls_logits, box_preds, obj_logits, mask_logits


class PyTorchSonarDetector(BaseSonarDetector):
    """
    Inference and feature extraction runner for SSSDebrisNet.
    Includes acoustic texture anomaly saliency scoring for zero-shot / baseline detection.
    """

    def __init__(
        self,
        model: Optional[SSSDebrisNet] = None,
        device: str = "cpu",
        weights_path: Optional[str] = None
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model or SSSDebrisNet()
        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        image_tile: np.ndarray,
        conf_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detects marine debris anomalies in a single tile.
        Combines neural FPN activations with acoustic shadow-highlight saliency.
        """
        h, w = image_tile.shape[:2]
        tile_f = image_tile.astype(np.float32)
        if tile_f.max() > 2.0:
            tile_f = tile_f / 255.0
        tile_f = np.clip(tile_f, 0.0, 1.0)

        # Tensor conversion (1, 1, H, W)
        tensor_in = torch.from_numpy(tile_f).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            cls_logits, box_preds, obj_logits, mask_logits = self.model(tensor_in)
            cls_probs = torch.sigmoid(cls_logits).squeeze(0).cpu().numpy()  # (C, H/4, W/4)
            box_offsets = box_preds.squeeze(0).cpu().numpy()               # (4, H/4, W/4)
            obj_probs = torch.sigmoid(obj_logits).squeeze().cpu().numpy()  # (H/4, W/4)

        # Extract acoustic saliency candidates directly from acoustic physics
        detections = self._extract_acoustic_candidates(tile_f, conf_threshold)

        # Merge with neural activation scores
        fh, fw = obj_probs.shape
        stride = 4.0

        for det in detections:
            bbox = det["bbox"]
            cx_feat = int(np.clip(bbox.center[0] / stride, 0, fw - 1))
            cy_feat = int(np.clip(bbox.center[1] / stride, 0, fh - 1))

            neural_obj = float(obj_probs[cy_feat, cx_feat])
            neural_cls = cls_probs[:, cy_feat, cx_feat]
            top_cls_idx = int(np.argmax(neural_cls))

            # Blend raw neural confidence with acoustic saliency
            fused_score = 0.65 * det["raw_score"] + 0.35 * max(0.4, neural_obj)
            det["confidence"] = float(np.clip(fused_score, 0.0, 1.0))

        # Filter by threshold
        valid_detections = [d for d in detections if d["confidence"] >= conf_threshold]
        return valid_detections

    def _extract_acoustic_candidates(
        self,
        tile: np.ndarray,
        min_conf: float
    ) -> List[Dict[str, Any]]:
        """
        Extracts candidate bounding boxes based on co-located acoustic highlight + shadow pairs.
        """
        h, w = tile.shape

        # Fast strided percentiles
        sub_sample = tile[::2, ::2]
        highlight_thresh = max(0.35, float(np.percentile(sub_sample, 95)))
        highlight_mask = tile > highlight_thresh

        shadow_thresh = min(0.20, float(np.percentile(sub_sample, 12)))
        shadow_mask = tile < shadow_thresh

        # Connected component analysis for highlights
        from scipy.ndimage import label, find_objects
        labeled_hl, num_hl = label(highlight_mask)
        slices = find_objects(labeled_hl)

        candidates: List[Dict[str, Any]] = []
        min_area = max(6, int(12 * (w / 1024.0)))
        shadow_search_px = max(20, int(w * 0.18))

        for i, slc in enumerate(slices):
            if slc is None:
                continue
            sy, sx = slc
            hl_w = sx.stop - sx.start
            hl_h = sy.stop - sy.start
            area = hl_w * hl_h

            # Filter out tiny specks and huge background blocks
            if area < min_area or area > (h * w * 0.4):
                continue

            hl_patch = tile[sy, sx]
            mean_hl = float(np.mean(hl_patch))

            # Check for adjacent shadow region (left or right)
            cx = (sx.start + sx.stop) // 2
            cy = (sy.start + sy.stop) // 2

            # Port shadow (search left)
            left_s_x1 = max(0, sx.start - shadow_search_px)
            left_s_x2 = sx.start
            left_shadow_patch = tile[sy, left_s_x1:left_s_x2] if left_s_x2 > left_s_x1 else np.array([])
            left_shadow_score = float(np.mean(left_shadow_patch < shadow_thresh)) if left_shadow_patch.size > 0 else 0.0

            # Starboard shadow (search right)
            right_s_x1 = sx.stop
            right_s_x2 = min(w, sx.stop + shadow_search_px)
            right_shadow_patch = tile[sy, right_s_x1:right_s_x2] if right_s_x2 > right_s_x1 else np.array([])
            right_shadow_score = float(np.mean(right_shadow_patch < shadow_thresh)) if right_shadow_patch.size > 0 else 0.0

            has_shadow = (left_shadow_score > 0.15) or (right_shadow_score > 0.15)
            best_shadow_score = max(left_shadow_score, right_shadow_score)

            # Determine Class based on geometry and acoustic signature
            aspect_ratio = float(hl_h) / max(1.0, float(hl_w))
            solidity = float(np.sum(labeled_hl[sy, sx] == (i + 1))) / float(area)

            if aspect_ratio > 2.2 or aspect_ratio < 0.45:
                # Elongated
                cls = DebrisClass.PIPE_CYLINDER
            elif area > (500 * (w / 1024.0)) and aspect_ratio > 1.2:
                cls = DebrisClass.SHIPWRECK
            elif solidity < 0.7 and area > (30 * (w / 1024.0)):
                # Tangled / webbed pattern
                cls = DebrisClass.GHOST_NET
            elif solidity > 0.8 and abs(aspect_ratio - 1.0) < 0.35:
                cls = DebrisClass.CONTAINER
            else:
                cls = DebrisClass.GENERIC_DEBRIS

            # Expand bounding box to enclose highlight + shadow
            if right_shadow_score > left_shadow_score and right_shadow_patch.size > 0:
                box_x1 = max(0, sx.start - 4)
                box_x2 = min(w, sx.stop + int(shadow_search_px * right_shadow_score) + 4)
            elif left_shadow_score > 0 and left_shadow_patch.size > 0:
                box_x1 = max(0, sx.start - int(shadow_search_px * left_shadow_score) - 4)
                box_x2 = min(w, sx.stop + 4)
            else:
                box_x1 = max(0, sx.start - 4)
                box_x2 = min(w, sx.stop + 4)

            box_y1 = max(0, sy.start - 4)
            box_y2 = min(h, sy.stop + 4)

            raw_score = float(np.clip(0.4 + 0.4 * mean_hl + 0.3 * best_shadow_score, 0.0, 1.0))

            # Approximate polygon mask
            polygon = [
                (float(box_x1), float(box_y1)),
                (float(box_x2), float(box_y1)),
                (float(box_x2), float(box_y2)),
                (float(box_x1), float(box_y2))
            ]

            candidates.append({
                "class_name": cls,
                "raw_score": raw_score,
                "confidence": raw_score,
                "bbox": BoundingBox(xmin=float(box_x1), ymin=float(box_y1), xmax=float(box_x2), ymax=float(box_y2)),
                "polygon": polygon,
                "has_shadow": has_shadow,
                "shadow_score": best_shadow_score
            })

        return candidates
