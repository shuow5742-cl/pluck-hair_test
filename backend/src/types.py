"""SegDetection — autoweaver Detection sub-class that carries segmentation results.

YOLOSegStep emits SegDetection instances into ``ctx.detections`` so downstream
tasks can use the standard ``for d in detections:`` iteration AND, when the
detection comes from a segmentation step, access mask + pick-point fields via
``isinstance(d, SegDetection)``.

Schema choice (option B from the integration design): the picked point /
angle / safety flag / shape class travel on the detection itself, not in a
side-channel ``ctx.metadata`` key. This keeps the pipeline → task contract
simple — every consumer reads ``ctx.detections``, and pickers add fields by
subclassing rather than by introducing parallel metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from autoweaver.pipeline import BoundingBox, Detection


@dataclass
class SegDetection(Detection):
    """A Detection carrying per-target segmentation + pick-point output."""

    # Geometry — full-image pixel coordinates.
    center_xy: list[float] = field(default_factory=list)
    polygon_xy: list[list[float]] = field(default_factory=list)
    mask_bbox_xyxy: Optional[list[int]] = None
    mask_area: int = 0

    # On-disk artifacts (when YoloSegmentationRuntime.save_artifacts=True).
    mask_path: Optional[str] = None
    mask_crop_path: Optional[str] = None
    bbox_crop_path: Optional[str] = None
    masked_crop_path: Optional[str] = None

    # Pick point output.
    pick_point_xy: Optional[list[float]] = None
    pick_angle_deg: Optional[float] = None
    pick_method: Optional[str] = None
    pick_score: Optional[float] = None
    distance_to_metal_px: Optional[float] = None
    distance_to_edge_px: Optional[float] = None
    preferred_epson_tool: Optional[int] = None

    # World-coordinate pick (mm). Filled by PixelToWorldTask after pixel
    # picking — None means either the pixel pick was None or no flange
    # pose was available at the moment of capture.
    world_xy: Optional[list[float]] = None

    # Shape descriptors (debug / diagnostics).
    shape_class: Optional[str] = None
    object_length_px: Optional[float] = None
    object_width_px: Optional[float] = None
    object_aspect_ratio: Optional[float] = None
    extent: Optional[float] = None
    solidity: Optional[float] = None
    hair_candidate_area: Optional[int] = None


__all__ = ["BoundingBox", "Detection", "SegDetection"]
