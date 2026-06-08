"""
Runtime pixel → world coordinate transformation.

Uses extrinsic calibration (camera-to-flange offset + axis mapping) together
with camera intrinsics (principal point) and the telecentric lens
mm_per_pixel constant to convert pixel detections into world-frame positions.

Core formula (eye-in-hand, 3-DOF flange, telecentric lens)::

    world_x = arm_x + dx + map_x(px - cx, py - cy) * mm_per_pixel
    world_y = arm_y + dy + map_y(px - cx, py - cy) * mm_per_pixel

where ``map_x`` and ``map_y`` are picked from the axis_mapping config:
the flange X/Y axes may correspond to ±image-X or ±image-Y depending on
how the camera is mounted relative to the user-frame on the controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from tools.calibrate.intrinsic_models import IntrinsicCalibrationResult


AxisToken = Literal["+px", "-px", "+py", "-py"]
_ALLOWED_TOKENS = {"+px", "-px", "+py", "-py"}


def _parse_axis_token(token: str, field_name: str) -> AxisToken:
    cleaned = (token or "").strip().lower()
    if cleaned not in _ALLOWED_TOKENS:
        raise ValueError(
            f"{field_name}={token!r} is not a valid axis token. "
            f"Allowed: {sorted(_ALLOWED_TOKENS)}"
        )
    return cleaned  # type: ignore[return-value]


def _eval_token(token: AxisToken, dpx_mm: float, dpy_mm: float) -> float:
    """Pick (and possibly negate) one of the two pixel-displacement deltas."""
    if token == "+px":
        return dpx_mm
    if token == "-px":
        return -dpx_mm
    if token == "+py":
        return dpy_mm
    if token == "-py":
        return -dpy_mm
    raise ValueError(f"unreachable: {token}")


@dataclass(frozen=True)
class ExtrinsicCalibration:
    """Loaded extrinsic calibration parameters."""

    mm_per_pixel: float
    dx: float                # camera optical center X offset from flange (mm)
    dy: float                # camera optical center Y offset from flange (mm)
    cx: float                # principal point X (pixels), from intrinsics
    cy: float                # principal point Y (pixels), from intrinsics
    flange_x_from: AxisToken = "+px"   # which image-axis (signed) feeds flange X
    flange_y_from: AxisToken = "+py"   # which image-axis (signed) feeds flange Y

    def __post_init__(self) -> None:
        # Validate tokens — direct construction (used by tests + future
        # programmatic callers) must reject garbage just like the yaml path.
        if self.flange_x_from not in _ALLOWED_TOKENS:
            raise ValueError(
                f"flange_x_from={self.flange_x_from!r} is not a valid axis token. "
                f"Allowed: {sorted(_ALLOWED_TOKENS)}"
            )
        if self.flange_y_from not in _ALLOWED_TOKENS:
            raise ValueError(
                f"flange_y_from={self.flange_y_from!r} is not a valid axis token. "
                f"Allowed: {sorted(_ALLOWED_TOKENS)}"
            )
        # axis_mapping is meaningful only if the two flange axes pull from
        # different image axes — otherwise the transform collapses (e.g.
        # both flange X and Y would advance with px alone, no py info).
        x_base = self.flange_x_from.lstrip("+-")
        y_base = self.flange_y_from.lstrip("+-")
        if x_base == y_base:
            raise ValueError(
                f"axis_mapping invalid: flange_x_from={self.flange_x_from!r} and "
                f"flange_y_from={self.flange_y_from!r} both reference '{x_base}'. "
                "Two flange axes must come from different image axes."
            )

    @classmethod
    def load(
        cls,
        extrinsic_path: str | Path,
        intrinsic_path: str | Path,
    ) -> ExtrinsicCalibration:
        """Load calibration from YAML files.

        Backward compat: if ``axis_mapping`` is absent, falls back to the
        legacy direct map (flange_x_from=+px, flange_y_from=+py), with the
        legacy ``flip_y`` switching flange_y_from to ``-py``.
        """
        intrinsic = IntrinsicCalibrationResult.load(intrinsic_path)

        with open(extrinsic_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        t_cam = data.get("T_cam_to_flange")
        if not isinstance(t_cam, dict):
            raise ValueError(
                f"Missing or invalid 'T_cam_to_flange' section in {extrinsic_path}"
            )

        axis_mapping = data.get("axis_mapping")
        if isinstance(axis_mapping, dict):
            flange_x_from = _parse_axis_token(
                axis_mapping.get("flange_x_from", ""), "axis_mapping.flange_x_from"
            )
            flange_y_from = _parse_axis_token(
                axis_mapping.get("flange_y_from", ""), "axis_mapping.flange_y_from"
            )
        else:
            # Legacy mode: identity mapping, with flip_y for the one common
            # symmetry we already supported.
            flange_x_from = "+px"
            flange_y_from = "-py" if bool(data.get("flip_y", False)) else "+py"

        return cls(
            mm_per_pixel=float(data.get("mm_per_pixel", 0)),
            dx=float(t_cam.get("dx", 0)),
            dy=float(t_cam.get("dy", 0)),
            cx=intrinsic.cx,
            cy=intrinsic.cy,
            flange_x_from=flange_x_from,
            flange_y_from=flange_y_from,
        )


@dataclass(frozen=True)
class WorldPoint:
    """A point in world coordinates (mm)."""

    x: float
    y: float


class CoordinateTransformer:
    """Converts pixel coordinates to world coordinates given arm pose."""

    def __init__(self, calibration: ExtrinsicCalibration) -> None:
        self._cal = calibration

    def pixel_to_world(
        self,
        px: float,
        py: float,
        arm_x: float,
        arm_y: float,
    ) -> WorldPoint:
        """Convert a single pixel coordinate to world coordinates.

        Parameters
        ----------
        px, py:
            Pixel coordinates in the image.
        arm_x, arm_y:
            Current robot arm flange position in world frame (mm).
        """
        cal = self._cal
        dpx_mm = (px - cal.cx) * cal.mm_per_pixel
        dpy_mm = (py - cal.cy) * cal.mm_per_pixel
        flange_dx_mm = _eval_token(cal.flange_x_from, dpx_mm, dpy_mm)
        flange_dy_mm = _eval_token(cal.flange_y_from, dpx_mm, dpy_mm)
        return WorldPoint(
            x=arm_x + cal.dx + flange_dx_mm,
            y=arm_y + cal.dy + flange_dy_mm,
        )

    def batch_pixel_to_world(
        self,
        pixels: list[tuple[float, float]],
        arm_x: float,
        arm_y: float,
    ) -> list[WorldPoint]:
        """Convert multiple pixel coordinates from the same frame.

        All pixels share the same arm pose (single snapshot).
        """
        return [self.pixel_to_world(px, py, arm_x, arm_y) for px, py in pixels]
