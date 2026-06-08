"""
Core domain modules.

Modules:
- coordinate_transform: pixel ↔ world coordinate conversion
- target_converter: TrackedTarget pixels → WorldTarget mm
"""

from .coordinate_transform import CoordinateTransformer, ExtrinsicCalibration, WorldPoint
from .pick_orientation import (
    DetectedBBox,
    OrientationEstimate,
    canonicalize_axis_yaw_deg,
    estimate_image_axis_yaw_from_bbox,
    estimate_orientation_from_binary_mask,
    normalize_angle_deg,
)
from .target_converter import TargetConverter, WorldTarget

__all__ = [
    "CoordinateTransformer",
    "DetectedBBox",
    "ExtrinsicCalibration",
    "OrientationEstimate",
    "TargetConverter",
    "WorldPoint",
    "WorldTarget",
    "canonicalize_axis_yaw_deg",
    "estimate_image_axis_yaw_from_bbox",
    "estimate_orientation_from_binary_mask",
    "normalize_angle_deg",
]
