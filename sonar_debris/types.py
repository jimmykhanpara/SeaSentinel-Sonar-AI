from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
import time


class DebrisClass(str, Enum):
    GHOST_NET = "ghost_net"
    SHIPWRECK = "shipwreck"
    PIPE_CYLINDER = "pipe_cylinder"
    GENERIC_DEBRIS = "generic_debris"
    TIRE = "tire"
    CONTAINER = "container"
    UNKNOWN_ANOMALY = "unknown_anomaly"
    ROCK_CLUTTER = "rock_clutter"  # Used for negative/suppression classification


class ChannelType(str, Enum):
    PORT = "port"
    STARBOARD = "starboard"
    BOTH = "both"


class AuditStatus(str, Enum):
    HIGH_CONFIDENCE = "high_confidence"
    LOW_CONFIDENCE = "low_confidence"
    ANALYST_CONFIRMED = "analyst_confirmed"
    ANALYST_REJECTED = "analyst_rejected"


class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height


class GeoPoint(BaseModel):
    latitude: float
    longitude: float
    depth_m: Optional[float] = 0.0
    towfish_altitude_m: Optional[float] = 10.0
    slant_range_m: Optional[float] = 0.0
    ground_range_m: Optional[float] = 0.0


class PhysicalDimensions(BaseModel):
    length_m: float
    width_m: float
    estimated_height_m: float = 0.0
    area_m2: float = 0.0


class AcousticSignature(BaseModel):
    highlight_mean: float = 0.0
    shadow_mean: float = 0.0
    seabed_mean: float = 0.0
    contrast_ratio: float = 0.0
    shadow_length_px: float = 0.0
    shadow_edge_sharpness: float = 0.0
    direction_alignment_score: float = 1.0  # 1.0 = points correctly away from nadir


class NavigationPoint(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    ping_number: int = 0
    latitude: float
    longitude: float
    heading_deg: float = 0.0  # 0-360 degrees
    altitude_m: float = 10.0  # Towfish altitude above seabed
    speed_knots: float = 3.0
    depth_m: float = 25.0


class DetectionResult(BaseModel):
    id: str
    class_name: DebrisClass
    confidence_percent: float  # 0.0 - 100.0%
    raw_model_score: float  # 0.0 - 1.0
    physics_score: float  # 0.0 - 1.0
    channel: ChannelType = ChannelType.BOTH
    bbox: BoundingBox
    polygon: Optional[List[Tuple[float, float]]] = None
    geo_location: GeoPoint
    dimensions: PhysicalDimensions
    acoustic_signature: AcousticSignature
    thumbnail_url: Optional[str] = None
    source_file: str = ""
    ping_range: Tuple[int, int] = (0, 0)
    status: AuditStatus = AuditStatus.HIGH_CONFIDENCE
    analyst_notes: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


class MissionMetadata(BaseModel):
    mission_id: str
    survey_name: str = "SSS Survey Mission"
    filename: str = ""
    start_time: float = Field(default_factory=time.time)
    end_time: float = Field(default_factory=time.time)
    total_pings: int = 0
    image_width: int = 0
    image_height: int = 0
    across_track_res_m: float = 0.05
    along_track_res_m: float = 0.05
    max_slant_range_m: float = 50.0
    altitude_avg_m: float = 10.0
    vessel_name: str = "Autonomous Marine Drone"


class PipelineConfig(BaseModel):
    model_type: str = "cnn_fpn"  # "cnn_fpn", "onnx", "unet"
    onnx_model_path: Optional[str] = None
    confidence_threshold_percent: float = 60.0
    low_confidence_audit_threshold: float = 30.0
    nms_iou_threshold: float = 0.45
    enable_tvg: bool = True
    enable_lee_filter: bool = True
    enable_slant_range_correction: bool = True
    enable_shadow_physics_validation: bool = True
    lee_window_size: int = 7
    canonical_resolution_m: float = 0.05
    tile_size: int = 512
    tile_overlap: int = 64
    export_thumbnails: bool = True


class MissionReport(BaseModel):
    metadata: MissionMetadata
    config: PipelineConfig
    summary: Dict[str, Any]
    detections: List[DetectionResult]
    audit_log: List[DetectionResult]
    nav_track: List[NavigationPoint]
