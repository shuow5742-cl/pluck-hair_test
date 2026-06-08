"""Tests for crop_single_square pipeline step."""

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
_pipeline_mod.ProcessStep = _StubProcessStep
_pipeline_mod.PipelineContext = object
_pipeline_mod.register_step = _stub_register_step
sys.modules.setdefault("autoweaver", types.ModuleType("autoweaver"))
sys.modules["autoweaver.pipeline"] = _pipeline_mod

from src.steps.crop_single_square import (
    PROCESSED_ORIGIN_METADATA_KEY,
    SQUARE_CROP_METADATA_KEY,
    CropSingleSquareStep,
)


@dataclass
class _FakePipelineContext:
    original_image: np.ndarray
    processed_image: np.ndarray | None = None
    detections: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.processed_image is None:
            self.processed_image = self.original_image.copy()


# Synthetic image params chosen so the step's defaults (mm_per_pixel = 0.009857,
# 10 mm cell + 0.5 mm chamfer + 0.5 mm collar) work without overrides.
_FRAME_W, _FRAME_H = 2048, 1536
_TARGET_TL = (520, 270)  # cell top-left in the frame; deliberately off-center
_BORDER = 60


def _step_with_calibration(**overrides) -> CropSingleSquareStep:
    """Build a step using the production calibration plus tweaks."""
    params = {"border_px": _BORDER, "center_bias": 0.0}
    params.update(overrides)
    return CropSingleSquareStep(params)


def _make_grid_image(target_tl: tuple[int, int]) -> np.ndarray:
    """Render a synthetic frame containing one ideal target cell.

    The cell has a bright interior, chamfered corners, and a dark collar
    matching the step's template. Surroundings are mid-gray (above the
    metal threshold) so they don't accidentally look like cells.
    """
    step = _step_with_calibration()
    side = step.cell_side_px
    chamfer = step.chamfer_px
    frame = step.frame_px
    image = np.full((_FRAME_H, _FRAME_W, 3), 160, dtype=np.uint8)  # neutral
    tx, ty = target_tl
    # Dark collar around the cell.
    image[ty - frame:ty + side + frame, tx - frame:tx + side + frame] = 30
    # Bright cell interior.
    image[ty:ty + side, tx:tx + side] = 230
    # Chamfered corners (paint back to dark).
    for cx, cy, sx, sy in [
        (tx, ty, +1, +1),
        (tx + side - 1, ty, -1, +1),
        (tx, ty + side - 1, +1, -1),
        (tx + side - 1, ty + side - 1, -1, -1),
    ]:
        for dx in range(chamfer):
            for dy in range(chamfer - dx):
                image[cy + sy * dy, cx + sx * dx] = 30
    return image


def _make_no_match_image() -> np.ndarray:
    """Solid dark frame — every match-template position scores low."""
    return np.full((_FRAME_H, _FRAME_W, 3), 20, dtype=np.uint8)


def test_template_match_finds_synthetic_cell():
    step = _step_with_calibration()
    image = _make_grid_image(_TARGET_TL)
    ctx = _FakePipelineContext(original_image=image)

    out = step.process(ctx)

    crop_meta = out.metadata[SQUARE_CROP_METADATA_KEY]
    assert crop_meta["applied"] is True
    assert crop_meta["source"] == "template_match"
    # Match should land on or very near the synthetic cell's top-left.
    sq = crop_meta["square_xyxy_in_input"]
    tx, ty = _TARGET_TL
    assert abs(sq[0] - tx) <= 2
    assert abs(sq[1] - ty) <= 2
    assert sq[2] - sq[0] == step.cell_side_px
    assert sq[3] - sq[1] == step.cell_side_px
    # Match score should be near 1 because the synthetic cell is built from
    # the template itself.
    assert crop_meta["match_score"] is not None
    assert crop_meta["match_score"] > 0.9


def test_geometric_fallback_when_match_score_below_threshold():
    step = _step_with_calibration(min_match_score=0.99)  # impossible to clear
    image = _make_no_match_image()
    ctx = _FakePipelineContext(original_image=image)

    out = step.process(ctx)

    crop_meta = out.metadata[SQUARE_CROP_METADATA_KEY]
    assert crop_meta["source"] == "geometric_fallback"
    # Centered fallback box of side `fallback_side_px` (default 1100).
    bx = crop_meta["box_xyxy_in_input"]
    assert bx[2] - bx[0] == step.fallback_side_px
    assert bx[3] - bx[1] == step.fallback_side_px


def test_origin_accumulates_with_prior_origin():
    step = _step_with_calibration()
    image = _make_grid_image(_TARGET_TL)
    ctx = _FakePipelineContext(original_image=image)
    ctx.metadata[PROCESSED_ORIGIN_METADATA_KEY] = [50, 70]

    out = step.process(ctx)

    # Final origin = prior + crop top-left (which is target - border/2).
    half = _BORDER // 2
    expected = [50 + _TARGET_TL[0] - half, 70 + _TARGET_TL[1] - half]
    assert out.metadata[PROCESSED_ORIGIN_METADATA_KEY] == expected
