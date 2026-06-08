"""Unit tests for target_converter module."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from src.core.coordinate_transform import ExtrinsicCalibration
from src.core.target_converter import TargetConverter, WorldTarget


# Minimal stand-in for TrackedTarget — proves no hard dependency
@dataclass
class FakeTarget:
    track_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float
    object_type: str


def _make_converter(
    mm_per_pixel: float = 0.009857,
    cx: float = 1023.2,
    cy: float = 766.3,
    dx: float = 0.0,
    dy: float = 0.0,
    flip_y: bool = False,
) -> TargetConverter:
    cal = ExtrinsicCalibration(
        mm_per_pixel=mm_per_pixel, dx=dx, dy=dy, cx=cx, cy=cy, flip_y=flip_y
    )
    return TargetConverter(cal)


class TestConvertOne:
    def test_center_pixel_at_origin(self):
        """Target at principal point → world (0, 0) when arm at origin."""
        conv = _make_converter(dx=0.0, dy=0.0)
        wt = conv.convert_one(
            track_id=1,
            x=1023.2, y=766.3,
            width=10.0, height=8.0,
            confidence=0.95,
            object_type="hair",
        )
        assert math.isclose(wt.x, 0.0, abs_tol=1e-9)
        assert math.isclose(wt.y, 0.0, abs_tol=1e-9)
        assert wt.track_id == 1
        assert wt.object_type == "hair"
        assert math.isclose(wt.confidence, 0.95)

    def test_known_offset(self):
        """Target 100px right of center → positive world X offset."""
        conv = _make_converter(mm_per_pixel=0.01, cx=500.0, cy=500.0)
        wt = conv.convert_one(
            track_id=2,
            x=600.0, y=500.0,
            width=20.0, height=10.0,
            confidence=0.8,
            object_type="hair",
        )
        # (600-500)*0.01 = 1.0 mm
        assert math.isclose(wt.x, 1.0, abs_tol=1e-9)
        assert math.isclose(wt.y, 0.0, abs_tol=1e-9)

    def test_bbox_scaled_to_mm(self):
        """Width and height are converted from pixels to mm."""
        conv = _make_converter(mm_per_pixel=0.01, cx=500.0, cy=500.0)
        wt = conv.convert_one(
            track_id=3,
            x=500.0, y=500.0,
            width=20.0, height=10.0,
            confidence=0.9,
            object_type="black_spot",
        )
        assert math.isclose(wt.width, 0.2, abs_tol=1e-9)   # 20 * 0.01
        assert math.isclose(wt.height, 0.1, abs_tol=1e-9)   # 10 * 0.01

    def test_with_arm_offset(self):
        """When arm is not at origin, world coords include arm position."""
        conv = _make_converter(mm_per_pixel=0.01, cx=500.0, cy=500.0, dx=0.0, dy=0.0)
        wt = conv.convert_one(
            track_id=4,
            x=500.0, y=500.0,
            width=10.0, height=10.0,
            confidence=0.9,
            object_type="hair",
            arm_x=50.0, arm_y=30.0,
        )
        assert math.isclose(wt.x, 50.0, abs_tol=1e-9)
        assert math.isclose(wt.y, 30.0, abs_tol=1e-9)


class TestConvertBatch:
    def test_batch_duck_typing(self):
        """convert() works with any object that has the right attrs."""
        conv = _make_converter(mm_per_pixel=0.01, cx=500.0, cy=500.0)
        targets = [
            FakeTarget(track_id=1, x=510.0, y=500.0, width=10.0, height=5.0, confidence=0.9, object_type="hair"),
            FakeTarget(track_id=2, x=490.0, y=500.0, width=8.0, height=4.0, confidence=0.8, object_type="hair"),
        ]
        results = conv.convert(targets)
        assert len(results) == 2
        assert math.isclose(results[0].x, 0.1, abs_tol=1e-9)   # (510-500)*0.01
        assert math.isclose(results[1].x, -0.1, abs_tol=1e-9)  # (490-500)*0.01

    def test_empty_list(self):
        conv = _make_converter()
        assert conv.convert([]) == []

    def test_world_target_is_frozen(self):
        """WorldTarget should be immutable."""
        conv = _make_converter(mm_per_pixel=0.01, cx=500.0, cy=500.0)
        wt = conv.convert_one(
            track_id=1, x=500.0, y=500.0,
            width=10.0, height=10.0,
            confidence=0.9, object_type="hair",
        )
        with pytest.raises(AttributeError):
            wt.x = 999.0
