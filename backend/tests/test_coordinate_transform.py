"""Unit tests for coordinate_transform module."""

from __future__ import annotations

import math

import pytest

from src.core.coordinate_transform import (
    CoordinateTransformer,
    ExtrinsicCalibration,
    WorldPoint,
)


def _make_cal(
    *,
    mm_per_pixel: float = 0.0069,
    dx: float = -2.5,
    dy: float = 1.3,
    cx: float = 1024.0,
    cy: float = 768.0,
    flange_x_from: str = "+px",
    flange_y_from: str = "+py",
) -> ExtrinsicCalibration:
    return ExtrinsicCalibration(
        mm_per_pixel=mm_per_pixel,
        dx=dx,
        dy=dy,
        cx=cx,
        cy=cy,
        flange_x_from=flange_x_from,
        flange_y_from=flange_y_from,
    )


class TestPixelToWorld:
    """Core formula: world = arm + d + map(pixel - principal) * scale."""

    def test_pixel_at_principal_arm_at_origin(self):
        """When pixel == principal point and arm at origin, world == (dx, dy)."""
        cal = _make_cal(dx=-2.5, dy=1.3)
        t = CoordinateTransformer(cal)
        wp = t.pixel_to_world(px=1024.0, py=768.0, arm_x=0.0, arm_y=0.0)
        assert math.isclose(wp.x, -2.5, abs_tol=1e-9)
        assert math.isclose(wp.y, 1.3, abs_tol=1e-9)

    def test_identity_mapping(self):
        """Default mapping (+px, +py) — verify with hand-calculated values."""
        cal = _make_cal(mm_per_pixel=0.01, dx=0.0, dy=0.0, cx=500.0, cy=500.0)
        t = CoordinateTransformer(cal)
        # px=600 → flange_dx = (600-500)*0.01 = +1.0
        # py=400 → flange_dy = (400-500)*0.01 = -1.0
        # arm at (10, 20) → world = (10+1, 20-1) = (11, 19)
        wp = t.pixel_to_world(px=600.0, py=400.0, arm_x=10.0, arm_y=20.0)
        assert math.isclose(wp.x, 11.0, abs_tol=1e-9)
        assert math.isclose(wp.y, 19.0, abs_tol=1e-9)

    def test_symmetry(self):
        """Pixels equidistant from principal point produce symmetric world offsets."""
        cal = _make_cal(mm_per_pixel=0.01, dx=0.0, dy=0.0, cx=500.0, cy=500.0)
        t = CoordinateTransformer(cal)
        wp1 = t.pixel_to_world(px=510.0, py=500.0, arm_x=0.0, arm_y=0.0)
        wp2 = t.pixel_to_world(px=490.0, py=500.0, arm_x=0.0, arm_y=0.0)
        assert math.isclose(wp1.x, -wp2.x, abs_tol=1e-9)
        assert math.isclose(wp1.y, wp2.y, abs_tol=1e-9)

    def test_negated_y_mapping(self):
        """flange_y_from=-py reverses the image-Y component, leaving X alone."""
        cal_normal = _make_cal(
            mm_per_pixel=0.01, dx=0.0, dy=0.0, cx=500.0, cy=500.0,
            flange_x_from="+px", flange_y_from="+py",
        )
        cal_flipped = _make_cal(
            mm_per_pixel=0.01, dx=0.0, dy=0.0, cx=500.0, cy=500.0,
            flange_x_from="+px", flange_y_from="-py",
        )
        wp_n = CoordinateTransformer(cal_normal).pixel_to_world(
            px=500.0, py=600.0, arm_x=0.0, arm_y=0.0
        )
        wp_f = CoordinateTransformer(cal_flipped).pixel_to_world(
            px=500.0, py=600.0, arm_x=0.0, arm_y=0.0
        )
        assert math.isclose(wp_n.x, wp_f.x, abs_tol=1e-9)
        assert math.isclose(wp_n.y, 1.0, abs_tol=1e-9)
        assert math.isclose(wp_f.y, -1.0, abs_tol=1e-9)

    def test_rig_axis_mapping(self):
        """Real rig (2026-05-19): flange X+ = camera moves toward image -Y;
        flange Y+ = camera moves toward image -X. Mapping is (-py, -px).
        Sanity: if camera is centered on world origin at flange (-60.9999, 0.4798)
        then dx=60.9999, dy=-0.4798 puts world origin back at pixel (cx, cy)."""
        cal = _make_cal(
            mm_per_pixel=0.009857,
            dx=60.9999, dy=-0.4798,
            cx=1023.1584, cy=766.2959,
            flange_x_from="-py", flange_y_from="-px",
        )
        t = CoordinateTransformer(cal)
        # Alignment: crosshair on world origin, flange at the recorded pose.
        wp = t.pixel_to_world(
            px=1023.1584, py=766.2959,
            arm_x=-60.9999, arm_y=0.4798,
        )
        assert math.isclose(wp.x, 0.0, abs_tol=1e-6)
        assert math.isclose(wp.y, 0.0, abs_tol=1e-6)

    def test_rig_axis_mapping_after_jog(self):
        """After flange jog the scene must stay at the same world coords.

        Start: crosshair on world origin, flange = (-60.9999, 0.4798).
        Jog flange X+ 11 mm (flange now (-50, 0.4798)).
        Scene point still at world (0,0), but camera moved physical -Y by
        11 mm → world origin appears 11/mmpp ≈ 1116 pixels lower (py larger)
        in image. py_new = 766.2959 + 11/0.009857.
        Plug that back into the transform; world must still be (0, 0).
        """
        mmpp = 0.009857
        cal = _make_cal(
            mm_per_pixel=mmpp,
            dx=60.9999, dy=-0.4798,
            cx=1023.1584, cy=766.2959,
            flange_x_from="-py", flange_y_from="-px",
        )
        t = CoordinateTransformer(cal)
        py_new = 766.2959 + 11.0 / mmpp
        wp = t.pixel_to_world(
            px=1023.1584, py=py_new,
            arm_x=-50.0, arm_y=0.4798,
        )
        assert math.isclose(wp.x, 0.0, abs_tol=1e-3)
        assert math.isclose(wp.y, 0.0, abs_tol=1e-3)


class TestAxisMappingValidation:
    def test_rejects_same_axis(self):
        with pytest.raises(ValueError, match="reference"):
            _make_cal(flange_x_from="+px", flange_y_from="-px")

    def test_rejects_unknown_token(self):
        with pytest.raises(ValueError, match="not a valid axis token"):
            ExtrinsicCalibration(
                mm_per_pixel=0.01, dx=0, dy=0, cx=0, cy=0,
                flange_x_from="north",  # type: ignore[arg-type]
                flange_y_from="+py",
            )


class TestBatchConversion:
    def test_batch_matches_individual(self):
        cal = _make_cal()
        t = CoordinateTransformer(cal)
        pixels = [(100.0, 200.0), (300.0, 400.0), (1024.0, 768.0)]
        arm_x, arm_y = 50.0, 60.0
        batch = t.batch_pixel_to_world(pixels, arm_x, arm_y)
        for (px, py), wp in zip(pixels, batch):
            expected = t.pixel_to_world(px, py, arm_x, arm_y)
            assert math.isclose(wp.x, expected.x, abs_tol=1e-12)
            assert math.isclose(wp.y, expected.y, abs_tol=1e-12)


class TestExtrinsicCalibrationLoad:
    def _write_intrinsic(self, path) -> None:
        path.write_text(
            """\
calibration_date: "2026-01-01"
num_images: 10
image_size:
  width: 2048
  height: 1536
camera_matrix:
  fx: 2500.0
  fy: 2500.0
  cx: 1024.0
  cy: 768.0
distortion:
  k1: 0.0
  k2: 0.0
  p1: 0.0
  p2: 0.0
  k3: 0.0
reprojection_error_px: 0.1
"""
        )

    def test_load_with_axis_mapping(self, tmp_path):
        """When axis_mapping is present it overrides flip_y."""
        intrinsic_yaml = tmp_path / "intrinsic.yaml"
        self._write_intrinsic(intrinsic_yaml)
        extrinsic_yaml = tmp_path / "extrinsic.yaml"
        extrinsic_yaml.write_text(
            """\
mm_per_pixel: 0.0069
T_cam_to_flange:
  dx: -2.5
  dy: 1.3
axis_mapping:
  flange_x_from: "-py"
  flange_y_from: "-px"
flip_y: true
"""
        )
        cal = ExtrinsicCalibration.load(extrinsic_yaml, intrinsic_yaml)
        assert cal.flange_x_from == "-py"
        assert cal.flange_y_from == "-px"
        assert math.isclose(cal.dx, -2.5)
        assert math.isclose(cal.cx, 1024.0)

    def test_load_legacy_flip_y(self, tmp_path):
        """No axis_mapping → flip_y=true means flange_y_from=-py, identity for X."""
        intrinsic_yaml = tmp_path / "intrinsic.yaml"
        self._write_intrinsic(intrinsic_yaml)
        extrinsic_yaml = tmp_path / "extrinsic.yaml"
        extrinsic_yaml.write_text(
            """\
mm_per_pixel: 0.0069
T_cam_to_flange:
  dx: -2.5
  dy: 1.3
flip_y: true
"""
        )
        cal = ExtrinsicCalibration.load(extrinsic_yaml, intrinsic_yaml)
        assert cal.flange_x_from == "+px"
        assert cal.flange_y_from == "-py"

    def test_load_legacy_no_flip(self, tmp_path):
        """No axis_mapping, flip_y absent → identity."""
        intrinsic_yaml = tmp_path / "intrinsic.yaml"
        self._write_intrinsic(intrinsic_yaml)
        extrinsic_yaml = tmp_path / "extrinsic.yaml"
        extrinsic_yaml.write_text(
            """\
mm_per_pixel: 0.0069
T_cam_to_flange:
  dx: 0.0
  dy: 0.0
"""
        )
        cal = ExtrinsicCalibration.load(extrinsic_yaml, intrinsic_yaml)
        assert cal.flange_x_from == "+px"
        assert cal.flange_y_from == "+py"
