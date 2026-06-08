"""Runtime-oriented tests for image-plane pick yaw estimation."""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.core.pick_orientation import DetectedBBox, estimate_image_axis_yaw_from_bbox


def test_estimate_image_axis_yaw_from_bbox_uses_image_clockwise_convention():
    image = np.full((220, 220, 3), 255, dtype=np.uint8)
    rect = ((110.0, 110.0), (120.0, 18.0), 30.0)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(image, box, (0, 0, 0))

    yaw = estimate_image_axis_yaw_from_bbox(
        image,
        bbox=DetectedBBox(x1=40.0, y1=60.0, x2=180.0, y2=160.0),
    )

    assert yaw is not None
    assert math.isclose(yaw, 30.0, abs_tol=3.0)
