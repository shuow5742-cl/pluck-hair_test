"""Helpers for estimating a grasp axis from a target ROI."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class DetectedBBox:
    """Normalized bbox representation used by orientation helpers."""

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

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


@dataclass(frozen=True, slots=True)
class OrientationEstimate:
    """Principal-axis orientation estimated from a single ROI."""

    centroid_x: float
    centroid_y: float
    axis_vx: float
    axis_vy: float
    image_angle_deg: float
    world_axis_angle_deg: float
    target_yaw_deg: float
    contour_area: float
    elongation_ratio: float
    major_variance: float
    minor_variance: float


def normalize_angle_deg(angle_deg: float) -> float:
    """Normalize angle to [-180, 180)."""
    normalized = (angle_deg + 180.0) % 360.0 - 180.0
    if normalized == -180.0:
        return 180.0
    return normalized


def canonicalize_axis_yaw_deg(angle_deg: float) -> float:
    """Collapse a 180-degree symmetric axis angle into [-90, 90)."""
    return (angle_deg + 90.0) % 180.0 - 90.0


def crop_roi(
    image: np.ndarray,
    bbox: DetectedBBox,
    *,
    padding: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a padded ROI, clamped to image bounds."""
    h, w = image.shape[:2]
    x1 = max(0, int(math.floor(bbox.x1)) - padding)
    y1 = max(0, int(math.floor(bbox.y1)) - padding)
    x2 = min(w, int(math.ceil(bbox.x2)) + padding)
    y2 = min(h, int(math.ceil(bbox.y2)) + padding)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def build_dark_mask(
    roi_bgr: np.ndarray,
    *,
    blur_kernel: int = 5,
    morph_kernel: int = 3,
    open_iterations: int = 1,
    close_iterations: int = 1,
) -> np.ndarray:
    """Build a binary mask where dark target pixels are white."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
    )
    if open_iterations > 0:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations
        )
    if close_iterations > 0:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations
        )
    return mask


def compute_contour_centroid(contour: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments(contour)
    if moments["m00"] <= 1e-6:
        return None
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def select_primary_contour(
    contours: list[np.ndarray],
    *,
    bbox_center_in_roi: tuple[float, float],
    roi_shape: tuple[int, int],
    min_area: float,
) -> np.ndarray | None:
    """Select the contour most likely to correspond to the detection target."""
    if not contours:
        return None

    roi_h, roi_w = roi_shape
    roi_diag = max(math.hypot(roi_w, roi_h), 1.0)
    cx, cy = bbox_center_in_roi

    best_contour: np.ndarray | None = None
    best_score = -float("inf")

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        centroid = compute_contour_centroid(contour)
        if centroid is None:
            continue

        distance = math.hypot(centroid[0] - cx, centroid[1] - cy)
        center_weight = math.exp(-((distance / (0.35 * roi_diag)) ** 2))
        score = area * center_weight

        if score > best_score:
            best_score = score
            best_contour = contour

    return best_contour


def estimate_orientation_from_binary_mask(
    mask: np.ndarray,
    *,
    flip_y: bool,
) -> OrientationEstimate | None:
    """Estimate dominant-axis orientation from a filled binary mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=None)
    centroid_x, centroid_y = float(mean[0, 0]), float(mean[0, 1])
    axis_vx, axis_vy = float(eigenvectors[0, 0]), float(eigenvectors[0, 1])
    major_variance = float(eigenvalues[0, 0])
    minor_variance = float(eigenvalues[1, 0]) if len(eigenvalues) > 1 else 0.0
    elongation_ratio = (
        major_variance / max(minor_variance, 1e-6) if major_variance > 0.0 else 0.0
    )

    image_angle_deg = normalize_angle_deg(math.degrees(math.atan2(axis_vy, axis_vx)))
    world_vy = -axis_vy if flip_y else axis_vy
    world_axis_angle_deg = normalize_angle_deg(
        math.degrees(math.atan2(world_vy, axis_vx))
    )
    target_yaw_deg = canonicalize_axis_yaw_deg(world_axis_angle_deg)

    return OrientationEstimate(
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        axis_vx=axis_vx,
        axis_vy=axis_vy,
        image_angle_deg=image_angle_deg,
        world_axis_angle_deg=world_axis_angle_deg,
        target_yaw_deg=target_yaw_deg,
        contour_area=float(len(points)),
        elongation_ratio=elongation_ratio,
        major_variance=major_variance,
        minor_variance=minor_variance,
    )


def estimate_image_axis_yaw_from_bbox(
    image: np.ndarray,
    *,
    bbox: DetectedBBox,
    padding: int = 8,
    min_contour_area: float = 20.0,
) -> float | None:
    """Estimate grasp axis in image coordinates for one bbox.

    The returned yaw follows the image-plane convention:
    3 o'clock is 0 deg, clockwise toward image-down is positive,
    and the 180-degree symmetric axis is folded to [-90, 90).
    """
    roi, (ox1, oy1, _ox2, _oy2) = crop_roi(image, bbox, padding=padding)
    if roi.size == 0:
        return None

    mask = build_dark_mask(roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox_cx, bbox_cy = bbox.center
    selected_contour = select_primary_contour(
        contours,
        bbox_center_in_roi=(bbox_cx - ox1, bbox_cy - oy1),
        roi_shape=mask.shape[:2],
        min_area=min_contour_area,
    )
    if selected_contour is None:
        return None

    contour_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [selected_contour], -1, 255, thickness=-1)

    orientation = estimate_orientation_from_binary_mask(contour_mask, flip_y=False)
    if orientation is None:
        return None

    return canonicalize_axis_yaw_deg(orientation.image_angle_deg)


__all__ = [
    "DetectedBBox",
    "OrientationEstimate",
    "build_dark_mask",
    "canonicalize_axis_yaw_deg",
    "compute_contour_centroid",
    "crop_roi",
    "estimate_image_axis_yaw_from_bbox",
    "estimate_orientation_from_binary_mask",
    "normalize_angle_deg",
    "select_primary_contour",
]
