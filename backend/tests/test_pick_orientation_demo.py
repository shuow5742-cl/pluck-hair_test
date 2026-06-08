"""Tests for pick_orientation_demo helpers."""

from __future__ import annotations

import math

import cv2
import numpy as np

from tools.pick_orientation_demo import (
    canonicalize_axis_yaw_deg,
    estimate_orientation_from_binary_mask,
    normalize_angle_deg,
)


def _make_rotated_mask(angle_deg: float) -> np.ndarray:
    mask = np.zeros((200, 200), dtype=np.uint8)
    rect = ((100.0, 100.0), (120.0, 18.0), angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(mask, box, 255)
    return mask


def test_normalize_angle_deg_wraps_to_signed_range():
    assert math.isclose(normalize_angle_deg(190.0), -170.0, abs_tol=1e-9)
    assert math.isclose(normalize_angle_deg(-200.0), 160.0, abs_tol=1e-9)


def test_canonicalize_axis_yaw_deg_collapses_180_symmetry():
    assert math.isclose(canonicalize_axis_yaw_deg(135.0), -45.0, abs_tol=1e-9)
    assert math.isclose(canonicalize_axis_yaw_deg(-135.0), 45.0, abs_tol=1e-9)
    assert math.isclose(canonicalize_axis_yaw_deg(30.0), 30.0, abs_tol=1e-9)


def test_estimate_orientation_from_binary_mask_matches_rotated_shape():
    mask = _make_rotated_mask(30.0)
    estimate = estimate_orientation_from_binary_mask(mask, flip_y=False)
    assert estimate is not None
    assert math.isclose(estimate.target_yaw_deg, 30.0, abs_tol=3.0)
    assert estimate.elongation_ratio > 5.0


def test_estimate_orientation_respects_flip_y():
    mask = _make_rotated_mask(30.0)
    estimate = estimate_orientation_from_binary_mask(mask, flip_y=True)
    assert estimate is not None
    assert math.isclose(estimate.world_axis_angle_deg, -30.0, abs_tol=3.0)
    assert math.isclose(estimate.target_yaw_deg, -30.0, abs_tol=3.0)
