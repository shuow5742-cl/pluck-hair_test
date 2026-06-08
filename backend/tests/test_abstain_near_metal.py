"""Tests for hard-safe-box and crop-trust behavior in abstain_near_metal."""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
import types

import numpy as np

_pipeline_mod = types.ModuleType("autoweaver.pipeline")


@dataclass
class _StubBoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


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
_pipeline_mod.ProcessStep = _StubProcessStep
_pipeline_mod.PipelineContext = object
_pipeline_mod.register_step = _stub_register_step
sys.modules.setdefault("autoweaver", types.ModuleType("autoweaver"))
sys.modules["autoweaver.pipeline"] = _pipeline_mod

from src.steps.abstain_near_metal import AbstainNearMetalStep
from src.types import BoundingBox, SegDetection


@dataclass
class _FakePipelineContext:
    original_image: np.ndarray
    processed_image: np.ndarray | None = None
    detections: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.processed_image is None:
            self.processed_image = self.original_image.copy()


def _make_detection(
    *,
    pick_xy: tuple[float, float],
    detection_id: str,
    distance_to_metal_px: float = 999.0,
) -> SegDetection:
    px, py = pick_xy
    return SegDetection(
        bbox=BoundingBox(x1=px - 5, y1=py - 5, x2=px + 5, y2=py + 5),
        object_type="foreign_matter",
        confidence=0.95,
        detection_id=detection_id,
        center_xy=[px, py],
        pick_point_xy=[px, py],
        pick_angle_deg=50.0,
        pick_method="mock",
        pick_score=0.9,
        distance_to_metal_px=distance_to_metal_px,
    )


def test_fallback_crop_skips_whole_region() -> None:
    step = AbstainNearMetalStep({"safety_margin_px": 20})
    ctx = _FakePipelineContext(
        original_image=np.full((200, 200, 3), 180, dtype=np.uint8),
        detections=[_make_detection(pick_xy=(100, 100), detection_id="d1")],
        metadata={
            "crop_single_square": {
                "source": "geometric_fallback",
                "match_score": None,
                "box_xyxy_in_original": [20, 20, 180, 180],
            }
        },
    )

    out = step.process(ctx)

    assert out.detections == []
    meta = out.metadata["abstain_near_metal"]
    assert meta["crop_valid"] is False
    assert meta["crop_guard_reason"] == "geometric_fallback"


def test_pick_outside_yellow_safe_box_is_dropped() -> None:
    step = AbstainNearMetalStep({"safety_margin_px": 20})
    ctx = _FakePipelineContext(
        original_image=np.full((200, 200, 3), 180, dtype=np.uint8),
        detections=[
            _make_detection(pick_xy=(100, 100), detection_id="inside"),
            _make_detection(pick_xy=(30, 100), detection_id="outside"),
        ],
        metadata={
            "crop_single_square": {
                "source": "template_match",
                "match_score": 0.9,
                "box_xyxy_in_original": [20, 20, 180, 180],
                "square_xyxy_in_original": [40, 40, 160, 160],
                "frame_px": 16,
            }
        },
    )

    out = step.process(ctx)

    assert [d.detection_id for d in out.detections] == ["inside"]
    meta = out.metadata["abstain_near_metal"]
    assert meta["safe_box_xyxy"] == [40, 40, 160, 160]
    assert meta["out_of_safe_box_count"] == 1
    preview_only = out.metadata["preview_only_detections"]
    assert len(preview_only) == 1
    assert preview_only[0].detection_id == "outside"
    assert preview_only[0].pick_point_xy is None
    assert preview_only[0].pick_angle_deg is None


def test_distance_to_metal_filter_still_applies_inside_safe_box() -> None:
    step = AbstainNearMetalStep({"safety_margin_px": 20})
    ctx = _FakePipelineContext(
        original_image=np.full((200, 200, 3), 180, dtype=np.uint8),
        detections=[
            _make_detection(pick_xy=(100, 100), detection_id="near", distance_to_metal_px=10.0),
            _make_detection(pick_xy=(110, 100), detection_id="far", distance_to_metal_px=30.0),
        ],
        metadata={
            "crop_single_square": {
                "source": "template_match",
                "match_score": 0.9,
                "box_xyxy_in_original": [20, 20, 180, 180],
                "square_xyxy_in_original": [40, 40, 160, 160],
                "frame_px": 16,
            }
        },
    )

    out = step.process(ctx)

    assert [d.detection_id for d in out.detections] == ["far"]
    meta = out.metadata["abstain_near_metal"]
    assert meta["too_close_to_metal_count"] == 1
