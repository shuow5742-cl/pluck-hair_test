"""Tests for calibrated image-axis -> pick-U mapping."""

from __future__ import annotations

import math

from src.core.pick_point_estimator import _image_axis_angle_to_pick_u_deg


def test_image_axis_angle_to_pick_u_deg_matches_validated_chart() -> None:
    # Operator-validated chart:
    # displayed axis -50/-20/0/20/40/60 deg -> true U 0/30/50/70/90/110 deg
    # The estimator first measures angles in image coordinates (+Y downward),
    # so the corresponding raw image-axis angles are mirrored across X:
    # 50/20/0/-20/-40/-60 deg -> true U 0/30/50/70/90/110 deg
    cases = [
        (50.0, 0.0),
        (20.0, 30.0),
        (0.0, 50.0),
        (-20.0, 70.0),
        (-40.0, 90.0),
        (-60.0, 110.0),
    ]

    for image_angle_deg, expected_u_deg in cases:
        actual = _image_axis_angle_to_pick_u_deg(image_angle_deg)
        assert actual is not None
        assert math.isclose(actual, expected_u_deg, abs_tol=1e-9)


def test_image_axis_angle_to_pick_u_deg_wraps_opposite_half_into_same_axis() -> None:
    # Left/down opposite half must collapse to the same 0-180 axis angle.
    assert math.isclose(_image_axis_angle_to_pick_u_deg(50.0), 0.0, abs_tol=1e-9)
    assert math.isclose(_image_axis_angle_to_pick_u_deg(-130.0), 0.0, abs_tol=1e-9)
