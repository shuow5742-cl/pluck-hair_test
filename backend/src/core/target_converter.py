"""
Convert pixel-space tracked targets to world-coordinate pick targets (mm).

Standalone module — does not modify existing business logic. Takes
TrackedTarget-like data (pixel x/y + bbox) and produces WorldTarget
with all dimensions in mm, ready for PLC consumption.

Usage::

    converter = TargetConverter.from_yaml(
        extrinsic_path="config/calibration/extrinsic.yaml",
        intrinsic_path="config/calibration/camera_intrinsic.yaml",
    )

    world_targets = converter.convert(tracked_targets, arm_x=0.0, arm_y=0.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.coordinate_transform import (
    CoordinateTransformer,
    ExtrinsicCalibration,
)


@dataclass(frozen=True, slots=True)
class WorldTarget:
    """Pick target with world coordinates (mm)."""

    track_id: int
    x: float  # world X (mm)
    y: float  # world Y (mm)
    width: float  # bbox width (mm)
    height: float  # bbox height (mm)
    confidence: float
    object_type: str


class TargetConverter:
    """Converts pixel-space targets to world-coordinate targets."""

    def __init__(self, calibration: ExtrinsicCalibration) -> None:
        self._cal = calibration
        self._transformer = CoordinateTransformer(calibration)

    @classmethod
    def from_yaml(
        cls,
        extrinsic_path: str | Path,
        intrinsic_path: str | Path,
    ) -> TargetConverter:
        """Create converter from extrinsic + intrinsic calibration YAML."""
        cal = ExtrinsicCalibration.load(extrinsic_path, intrinsic_path)
        return cls(cal)

    def convert_one(
        self,
        *,
        track_id: int,
        x: float,
        y: float,
        width: float,
        height: float,
        confidence: float,
        object_type: str,
        arm_x: float = 0.0,
        arm_y: float = 0.0,
    ) -> WorldTarget:
        """Convert a single target from pixels to mm."""
        wp = self._transformer.pixel_to_world(x, y, arm_x, arm_y)
        scale = self._cal.mm_per_pixel
        return WorldTarget(
            track_id=track_id,
            x=wp.x,
            y=wp.y,
            width=width * scale,
            height=height * scale,
            confidence=confidence,
            object_type=object_type,
        )

    def convert(
        self,
        targets: list,
        arm_x: float = 0.0,
        arm_y: float = 0.0,
    ) -> list[WorldTarget]:
        """Convert a list of TrackedTarget (or any object with matching attrs) to mm.

        Accepts any object with track_id, x, y, width, height, confidence,
        object_type attributes — no hard dependency on TrackedTarget.
        """
        return [
            self.convert_one(
                track_id=t.track_id,
                x=t.x,
                y=t.y,
                width=t.width,
                height=t.height,
                confidence=t.confidence,
                object_type=t.object_type,
                arm_x=arm_x,
                arm_y=arm_y,
            )
            for t in targets
        ]
