"""Tests for yolo_seg coordinate re-projection after crop."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import sys
import types

from src.core.coordinate_transform import CoordinateTransformer, ExtrinsicCalibration

_pipeline_mod = types.ModuleType("autoweaver.pipeline")


@dataclass
class _StubBoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class _StubDetection:
    bbox: _StubBoundingBox
    object_type: str
    confidence: float
    detection_id: str | None = None


class _StubProcessStep:
    def __init__(self, params=None):
        self._params = params or {}
        self._custom_name = self._params.pop("_custom_name", None)

    @property
    def params(self):
        return self._params


def _stub_register_step(_name, _step_class) -> None:
    return None


_pipeline_mod.BoundingBox = _StubBoundingBox
_pipeline_mod.Detection = _StubDetection
_pipeline_mod.PipelineContext = object
_pipeline_mod.ProcessStep = _StubProcessStep
_pipeline_mod.register_step = _stub_register_step
sys.modules.setdefault("autoweaver", types.ModuleType("autoweaver"))
sys.modules["autoweaver.pipeline"] = _pipeline_mod

from src.steps.yolo_seg import _to_seg_detections


@dataclass
class _FakeSegmentDetection:
    index: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    center_xy: list[float]
    mask_area: int
    mask_bbox_xyxy: list[int] | None = None
    polygon_xy: list[list[float]] = field(default_factory=list)
    bbox_crop_path: str | None = None
    mask_path: str | None = None
    mask_crop_path: str | None = None
    masked_crop_path: str | None = None
    pick_point_xy: list[float] | None = None
    pick_angle_deg: float | None = None
    pick_method: str | None = None
    pick_score: float | None = None
    object_length_px: float | None = None
    object_width_px: float | None = None
    object_aspect_ratio: float | None = None
    distance_to_edge_px: float | None = None
    distance_to_metal_px: float | None = None
    hair_candidate_area: int | None = None
    extent: float | None = None
    solidity: float | None = None
    shape_class: str | None = None


def test_to_seg_detections_restores_crop_offset_to_full_frame_pixels():
    raw = _FakeSegmentDetection(
        index=0,
        class_id=0,
        class_name="hair",
        confidence=0.95,
        bbox_xyxy=[10.0, 20.0, 50.0, 60.0],
        center_xy=[30.0, 40.0],
        mask_area=1200,
        mask_bbox_xyxy=[12, 22, 48, 58],
        polygon_xy=[[10.0, 20.0], [50.0, 20.0], [50.0, 60.0]],
        pick_point_xy=[25.0, 35.0],
        pick_angle_deg=12.0,
        pick_method="mock",
        pick_score=0.8,
    )

    det = _to_seg_detections([raw], origin_xy=(100, 200))[0]

    assert det.bbox.to_xyxy() == (110.0, 220.0, 150.0, 260.0)
    assert det.center_xy == [130.0, 240.0]
    assert det.mask_bbox_xyxy == [112, 222, 148, 258]
    assert det.pick_point_xy == [125.0, 235.0]
    assert det.polygon_xy == [[110.0, 220.0], [150.0, 220.0], [150.0, 260.0]]


def test_restored_pick_point_produces_full_frame_world_coordinates():
    raw = _FakeSegmentDetection(
        index=0,
        class_id=0,
        class_name="hair",
        confidence=0.95,
        bbox_xyxy=[0.0, 0.0, 10.0, 10.0],
        center_xy=[5.0, 5.0],
        mask_area=100,
        pick_point_xy=[25.0, 35.0],
    )
    det = _to_seg_detections([raw], origin_xy=(100, 200))[0]

    cal = ExtrinsicCalibration(
        mm_per_pixel=0.1,
        dx=0.0,
        dy=0.0,
        cx=0.0,
        cy=0.0,
        flange_x_from="+px",
        flange_y_from="+py",
    )
    transformer = CoordinateTransformer(cal)
    wp = transformer.pixel_to_world(
        px=det.pick_point_xy[0],
        py=det.pick_point_xy[1],
        arm_x=0.0,
        arm_y=0.0,
    )

    assert math.isclose(wp.x, 12.5, abs_tol=1e-9)
    assert math.isclose(wp.y, 23.5, abs_tol=1e-9)
