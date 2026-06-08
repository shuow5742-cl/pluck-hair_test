"""Estimate a practical plucking point from a YOLO-seg mask and original image.

The segmentation mask is useful, but it is not always the true hair position.
For the bird's-nest plucking task the final output must be one safe pick point.
This module therefore treats the YOLO mask as a coarse search region, then uses
image evidence to refine the point:

1. keep the existing mask-skeleton strategy for clearly elongated masks
2. otherwise search for dark, thin hair-like pixels inside/near the mask and bbox
3. keep the point away from the dark metal pressing plate
4. fall back to safe distance-transform center, safe centroid, then bbox center

All coordinates returned by this module are full-image pixel coordinates.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np


# Shape-class thresholds (angle-invariant indicators).
# Calibrated 2026-05-19 against 107 labeled polygons in 05.15_4_2 (labelme).
STRAIGHT_THIN_ASPECT_MIN = 3.0
STRAIGHT_THIN_EXTENT_MIN = 0.55       # mask_area / minAreaRect_area
STRAIGHT_THIN_SOLIDITY_MIN = 0.72     # mask_area / convex_hull_area

CURVED_SOLIDITY_MAX = 0.72            # the linchpin: bent hair fails the hull test
CURVED_ASPECT_MIN = 1.2               # don't classify near-blobs as curved

DOWN_CLUMP_ASPECT_MAX = 2.5
DOWN_CLUMP_EXTENT_MIN = 0.55
DOWN_CLUMP_SOLIDITY_MIN = 0.75

# dense_clump: a big mask with internal density structure (irregular shape OK).
# Distinct from down_clump, which is the small uniform feather-puff case.
DENSE_CLUMP_MIN_AREA_PX = 5000        # absolute pixel count — recalibrate per camera
DENSE_CLUMP_MIN_CONTRAST = 25.0       # p90 - p10 of gray within mask
DENSE_CLUMP_MIN_DARK_FRACTION = 0.08  # at least 8% pixels notably below mask median

# Empirical image-axis -> robot-U calibration from the operator-validated
# angle chart. The chart is interpreted in a Cartesian-style display frame:
# +x points right, +y points up, U=50 deg lies on the rightward horizontal
# axis, and the grasp axis is 180-degree symmetric. Raw image geometry is
# still measured in OpenCV image coordinates (+y downward), so the image-axis
# angle must be mirrored across X before applying the U offset.
PICK_U_ZERO_OFFSET_DEG = 50.0


@dataclass
class PickPointResult:
    """Structured pick point information for one segmented target."""

    pick_point_xy: list[float]
    pick_angle_deg: float | None
    pick_method: str
    pick_score: float
    length_px: float | None = None
    width_px: float | None = None
    aspect_ratio: float | None = None
    distance_to_edge_px: float | None = None
    distance_to_metal_px: float | None = None
    hair_candidate_area: int | None = None
    extent: float | None = None
    solidity: float | None = None
    shape_class: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetalSafetyContext:
    """Frame-level shared metal safety data reusable across detections."""

    gray: np.ndarray | None
    metal_mask: np.ndarray
    forbidden_mask: np.ndarray
    metal_distance: np.ndarray | None
    timing_ms: dict[str, float] = field(default_factory=dict)


def build_metal_safety_context(
    *,
    frame_bgr: np.ndarray | None,
    yolo_mask_union: np.ndarray | None = None,
    metal_safety_margin_px: int = 78,
) -> MetalSafetyContext:
    """Precompute frame-level metal safety artifacts once per image."""
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        shape = (
            frame_bgr.shape[:2]
            if frame_bgr is not None and getattr(frame_bgr, "ndim", 0) >= 2
            else (0, 0)
        )
        height, width = int(shape[0]), int(shape[1])
        return MetalSafetyContext(
            gray=None,
            metal_mask=np.zeros((height, width), dtype=np.uint8),
            forbidden_mask=np.zeros((height, width), dtype=np.uint8),
            metal_distance=None,
            timing_ms={},
        )

    total_start = time.perf_counter()
    timing_ms: dict[str, float] = {}

    def _record(name: str, start: float) -> None:
        timing_ms[name] = round((time.perf_counter() - start) * 1000.0, 2)

    start = time.perf_counter()
    gray = (
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if frame_bgr.ndim == 3
        else frame_bgr.copy()
    )
    _record("gray_convert_ms", start)

    start = time.perf_counter()
    metal_mask = _detect_metal_plate(gray, exclude_mask=yolo_mask_union)
    _record("metal_detect_ms", start)

    start = time.perf_counter()
    metal_distance = cv2.distanceTransform(
        (metal_mask == 0).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    _record("metal_distance_ms", start)

    start = time.perf_counter()
    forbidden_mask = (metal_distance <= float(metal_safety_margin_px)).astype(np.uint8)
    _record("metal_forbidden_dilate_ms", start)
    timing_ms["shared_metal_total_ms"] = round(
        (time.perf_counter() - total_start) * 1000.0, 2
    )
    return MetalSafetyContext(
        gray=gray,
        metal_mask=metal_mask,
        forbidden_mask=forbidden_mask,
        metal_distance=metal_distance,
        timing_ms=timing_ms,
    )


def estimate_pick_point(
    mask_bin: np.ndarray,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    *,
    frame_bgr: np.ndarray | None = None,
    elongated_aspect_ratio: float = 2.5,
    min_mask_area: int = 5,
    metal_safety_margin_px: int = 78,
    yolo_mask_union: np.ndarray | None = None,
    metal_safety_context: MetalSafetyContext | None = None,
) -> PickPointResult:
    """Estimate the best plucking point from one full-image mask.

    Args:
        mask_bin: Full-image binary mask. Non-zero pixels belong to YOLO target.
        bbox_xyxy: Detection bbox in full-image coordinates.
        frame_bgr: Original full-image BGR frame. If supplied, dark hair-line
            refinement and metal-plate safety checks are enabled.
        elongated_aspect_ratio: Ratio threshold to treat YOLO mask as hair-like.
        min_mask_area: Area below which the mask is considered unreliable.
        metal_safety_margin_px: Minimum desired distance from the metal plate.
        yolo_mask_union: Optional full-image binary mask of ALL YOLO foreign-
            matter detections in this frame. Pixels here are excluded from
            metal-plate candidates so dark fiber clumps aren't mistaken for
            plate bars.

    Returns:
        PickPointResult with full-image pixel coordinates.
    """

    total_start = time.perf_counter()
    timing_ms: dict[str, float] = {}

    def _record(name: str, start: float) -> None:
        timing_ms[name] = round((time.perf_counter() - start) * 1000.0, 2)

    def _finish(result: PickPointResult) -> PickPointResult:
        result.timing_ms = dict(timing_ms)
        result.timing_ms["pick_total_ms"] = round((time.perf_counter() - total_start) * 1000.0, 2)
        return result

    height, width = mask_bin.shape[:2]
    x1, y1, x2, y2 = _clip_bbox_to_mask(bbox_xyxy, width, height)
    bbox_center = [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]

    gray: np.ndarray | None = None
    metal_mask = np.zeros((height, width), dtype=np.uint8)
    forbidden_mask = np.zeros((height, width), dtype=np.uint8)
    metal_distance: np.ndarray | None = None
    if metal_safety_context is not None:
        gray = metal_safety_context.gray
        metal_mask = metal_safety_context.metal_mask
        forbidden_mask = metal_safety_context.forbidden_mask
        metal_distance = metal_safety_context.metal_distance
    elif frame_bgr is not None and frame_bgr.size > 0:
        shared_ctx = build_metal_safety_context(
            frame_bgr=frame_bgr,
            yolo_mask_union=yolo_mask_union,
            metal_safety_margin_px=metal_safety_margin_px,
        )
        gray = shared_ctx.gray
        metal_mask = shared_ctx.metal_mask
        forbidden_mask = shared_ctx.forbidden_mask
        metal_distance = shared_ctx.metal_distance
        for key, value in shared_ctx.timing_ms.items():
            if key != "shared_metal_total_ms":
                timing_ms[key] = float(value)

    start = time.perf_counter()
    binary = (mask_bin > 0).astype(np.uint8)
    _record("mask_binarize_ms", start)
    if int(binary.sum()) < min_mask_area:
        finalize_start = time.perf_counter()
        result = _finalize_result(
            PickPointResult(
                pick_point_xy=bbox_center,
                pick_angle_deg=None,
                pick_method="bbox_center_fallback",
                pick_score=0.1,
            ),
            metal_distance,
            forbidden_mask,
            metal_safety_margin_px,
        )
        _record("finalize_ms", finalize_start)
        return _finish(result)

    # Work on the largest connected component to reduce YOLO mask noise.
    start = time.perf_counter()
    component = _largest_connected_component(binary)
    _record("largest_component_ms", start)
    if int(component.sum()) < min_mask_area:
        finalize_start = time.perf_counter()
        result = _finalize_result(
            PickPointResult(
                pick_point_xy=bbox_center,
                pick_angle_deg=None,
                pick_method="bbox_center_fallback",
                pick_score=0.1,
            ),
            metal_distance,
            forbidden_mask,
            metal_safety_margin_px,
        )
        _record("finalize_ms", finalize_start)
        return _finish(result)

    start = time.perf_counter()
    mask_bbox = _mask_bbox(component)
    _record("mask_bbox_ms", start)
    if mask_bbox is None:
        finalize_start = time.perf_counter()
        result = _finalize_result(
            PickPointResult(
                pick_point_xy=bbox_center,
                pick_angle_deg=None,
                pick_method="bbox_center_fallback",
                pick_score=0.1,
            ),
            metal_distance,
            forbidden_mask,
            metal_safety_margin_px,
        )
        _record("finalize_ms", finalize_start)
        return _finish(result)

    mx1, my1, mx2, my2 = mask_bbox
    start = time.perf_counter()
    roi = _clean_mask(component[my1:my2, mx1:mx2])
    _record("clean_mask_ms", start)

    start = time.perf_counter()
    descriptors = _shape_descriptors(roi)
    _record("shape_descriptors_ms", start)
    aspect_ratio = descriptors["aspect_ratio"]
    angle_deg = descriptors["angle_deg"]
    length_px = descriptors["length_px"]
    width_px = descriptors["width_px"]
    extent = descriptors["extent"]
    solidity = descriptors["solidity"]
    # Use the full-image gray + full-image mask component so dense-core
    # detection sees actual pixel intensities, not just the geometry.
    start = time.perf_counter()
    shape_class = _classify_shape(descriptors, gray=gray, mask=component)
    _record("classify_shape_ms", start)

    def _annotate(result: PickPointResult) -> PickPointResult:
        # Overwrite all shape descriptors from the outer YOLO mask so the
        # stored aspect/extent/solidity match the mask shape_class was computed
        # against. Internal helpers (e.g. dark_line_refinement) may set their
        # own aspect_ratio from a refined sub-mask, but mixing them with the
        # outer extent/solidity would be incoherent.
        result.length_px = length_px
        result.width_px = width_px
        result.aspect_ratio = aspect_ratio
        result.extent = extent
        result.solidity = solidity
        result.shape_class = shape_class
        return result

    # ----- 3-class shape dispatch -----
    # Order matters: curved is detected by low solidity regardless of aspect,
    # because bent hair fails the convex-hull test even when aspect drops
    # below the "straight thin" gate.
    if shape_class == "straight_thin":
        start = time.perf_counter()
        result = _pick_from_skeleton(roi, mx1, my1, angle_deg, aspect_ratio, metal_distance, forbidden_mask)
        _record("pick_straight_thin_ms", start)
        if result is not None:
            finalize_start = time.perf_counter()
            result = _finalize_result(_annotate(result), metal_distance, forbidden_mask, metal_safety_margin_px)
            _record("finalize_ms", finalize_start)
            return _finish(result)

    if shape_class == "curved":
        start = time.perf_counter()
        result = _pick_from_curved_skeleton(roi, mx1, my1, metal_distance, forbidden_mask)
        _record("pick_curved_ms", start)
        if result is not None:
            finalize_start = time.perf_counter()
            result = _finalize_result(_annotate(result), metal_distance, forbidden_mask, metal_safety_margin_px)
            _record("finalize_ms", finalize_start)
            return _finish(result)

    if shape_class in ("dense_clump", "down_clump") and gray is not None:
        start = time.perf_counter()
        result = _pick_from_density_thickness(
            gray=gray,
            yolo_mask=component,
            bbox_xyxy=(x1, y1, x2, y2),
            metal_mask=metal_mask,
            forbidden_mask=forbidden_mask,
            metal_distance=metal_distance,
            safety_margin_px=metal_safety_margin_px,
        )
        _record("pick_density_thickness_ms", start)
        if result is not None:
            finalize_start = time.perf_counter()
            result = _finalize_result(_annotate(result), metal_distance, forbidden_mask, metal_safety_margin_px)
            _record("finalize_ms", finalize_start)
            return _finish(result)

    # Ambiguous, or the class-specific strategy returned None: use original
    # image evidence as a coarse ROI and pick the darkest hair pixels.
    if gray is not None:
        start = time.perf_counter()
        dark_line_result = _pick_from_dark_line_refinement(
            gray=gray,
            yolo_mask=component,
            bbox_xyxy=(x1, y1, x2, y2),
            metal_mask=metal_mask,
            forbidden_mask=forbidden_mask,
            metal_distance=metal_distance,
            safety_margin_px=metal_safety_margin_px,
        )
        _record("pick_dark_line_ms", start)
        if dark_line_result is not None:
            finalize_start = time.perf_counter()
            result = _finalize_result(_annotate(dark_line_result), metal_distance, forbidden_mask, metal_safety_margin_px)
            _record("finalize_ms", finalize_start)
            return _finish(result)

    # Blob-like or refinement failed: pick the safest inner point after removing
    # metal plate forbidden area from the candidate mask.
    safe_component = component.copy()
    if forbidden_mask.any():
        safe_component[forbidden_mask > 0] = 0
    start = time.perf_counter()
    safe_bbox = _mask_bbox(safe_component)
    _record("safe_mask_bbox_ms", start)
    if safe_bbox is not None:
        sx1, sy1, sx2, sy2 = safe_bbox
        start = time.perf_counter()
        safe_roi = _clean_mask(safe_component[sy1:sy2, sx1:sx2])
        _record("safe_clean_mask_ms", start)
        start = time.perf_counter()
        distance_result = _pick_from_distance_transform(safe_roi, sx1, sy1, angle_deg, aspect_ratio)
        _record("pick_distance_transform_ms", start)
        if distance_result is not None:
            finalize_start = time.perf_counter()
            result = _finalize_result(_annotate(distance_result), metal_distance, forbidden_mask, metal_safety_margin_px)
            _record("finalize_ms", finalize_start)
            return _finish(result)

        start = time.perf_counter()
        centroid = _mask_centroid(safe_roi)
        _record("safe_centroid_ms", start)
        if centroid is not None:
            px, py = centroid
            finalize_start = time.perf_counter()
            result = _finalize_result(
                _annotate(PickPointResult(
                    pick_point_xy=[float(px + sx1), float(py + sy1)],
                    pick_angle_deg=_image_axis_angle_to_pick_u_deg(angle_deg),
                    pick_method="safe_mask_centroid_fallback",
                    pick_score=0.45,
                )),
                metal_distance,
                forbidden_mask,
                metal_safety_margin_px,
            )
            _record("finalize_ms", finalize_start)
            return _finish(result)

    # Last fallback: original mask centroid, then bbox center.
    start = time.perf_counter()
    centroid = _mask_centroid(roi)
    _record("mask_centroid_ms", start)
    if centroid is not None:
        px, py = centroid
        finalize_start = time.perf_counter()
        result = _finalize_result(
            _annotate(PickPointResult(
                pick_point_xy=[float(px + mx1), float(py + my1)],
                pick_angle_deg=_image_axis_angle_to_pick_u_deg(angle_deg),
                pick_method="mask_centroid_fallback",
                pick_score=0.35,
            )),
            metal_distance,
            forbidden_mask,
            metal_safety_margin_px,
        )
        _record("finalize_ms", finalize_start)
        return _finish(result)

    finalize_start = time.perf_counter()
    result = _finalize_result(
        _annotate(PickPointResult(
            pick_point_xy=bbox_center,
            pick_angle_deg=_image_axis_angle_to_pick_u_deg(angle_deg),
            pick_method="bbox_center_fallback",
            pick_score=0.1,
        )),
        metal_distance,
        forbidden_mask,
        metal_safety_margin_px,
    )
    _record("finalize_ms", finalize_start)
    return _finish(result)


def _pick_from_density_thickness(
    *,
    gray: np.ndarray,
    yolo_mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    metal_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    metal_distance: np.ndarray | None,
    safety_margin_px: int,
) -> PickPointResult | None:
    """Pick where the mask is BOTH thick AND dark — argmax(raw_weight).

    For clump-class targets, the right pick maximizes the same darkness ×
    distance-from-edge composite that ``_pick_from_broad_dark_core`` already
    builds as ``raw_weight``, but with all of broad_dark_core's downstream
    centroid-blending stripped away. Empirically (cell_03 walkthrough) the
    raw_weight map itself targets the right region; the previous bias came
    from the candidate selection + mask-centroid blend pulling the pick
    toward the geometric center of a sprawling mask.

    Used by both dense_clump (big tangled chunks) and down_clump (small
    uniform feather puffs). Other shape classes are unaffected.
    """

    height, width = gray.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    pad = max(8, int(max(x2 - x1, y2 - y1) * 0.04))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(width, x2 + pad)
    ry2 = min(height, y2 + pad)
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi_gray = gray[ry1:ry2, rx1:rx2]
    roi_mask = (yolo_mask[ry1:ry2, rx1:rx2] > 0).astype(np.uint8)
    roi_forbidden = (
        forbidden_mask[ry1:ry2, rx1:rx2]
        if forbidden_mask.size
        else np.zeros_like(roi_mask)
    )
    roi_metal = metal_mask[ry1:ry2, rx1:rx2]

    # Forbidden / metal pixels are off-limits — gripper would clamp the plate.
    safe_mask = roi_mask.copy()
    safe_mask[roi_forbidden > 0] = 0
    safe_mask[roi_metal > 0] = 0
    safe_mask = _clean_mask(safe_mask)
    safe_area = int(safe_mask.sum())
    if safe_area < 80:
        return None

    vals = roi_gray[safe_mask > 0].astype(np.float32)
    if vals.size < 80:
        return None
    p10 = float(np.percentile(vals, 10))
    p90 = float(np.percentile(vals, 90))
    contrast = max(p90 - p10, 5.0)

    # Mask thickness map (distance to nearest mask boundary).
    dist_obj = cv2.distanceTransform((safe_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    max_obj_dist = max(float(dist_obj.max()), 1.0)
    dist_norm = np.clip(dist_obj / max_obj_dist, 0.0, 1.0).astype(np.float32)

    # Edge guard: only consider pixels at least inner_thr px inside the mask,
    # so a single-pixel dark vein near the boundary can't win.
    inner_thr = max(1.5, min(10.0, max_obj_dist * 0.10))
    inner_mask = ((safe_mask > 0) & (dist_obj >= inner_thr)).astype(np.uint8)
    if int(inner_mask.sum()) < max(30, int(safe_area * 0.05)):
        inner_mask = safe_mask.copy()

    # Darkness signals: absolute (vs mask p90) and local (vs median-blur background).
    dark = np.clip(
        (p90 - roi_gray.astype(np.float32)) / contrast, 0.0, 1.0
    )
    k = _odd_kernel(max(31, min(151, int(max(roi_gray.shape[:2]) * 0.24))))
    local_bg = cv2.medianBlur(roi_gray, k).astype(np.float32)
    local_dark = np.clip(
        (local_bg - roi_gray.astype(np.float32)) / contrast, 0.0, 1.0
    )

    # raw_weight: same formula as _pick_from_broad_dark_core's intermediate.
    # First term = darkness in two flavors; second factor = thickness weighting.
    raw_weight = (
        0.70 * np.power(dark, 2.2) + 0.30 * np.power(local_dark, 1.6)
    ).astype(np.float32)
    raw_weight[inner_mask == 0] = 0.0
    raw_weight *= (0.35 + 0.65 * np.power(dist_norm, 0.45)).astype(np.float32)

    if float(raw_weight.max()) <= 1e-6:
        return None

    # Argmax of raw_weight. Full stop. No candidate set, no morphology, no
    # component selection, no mask_centroid blend. The whole point of this
    # picker is that raw_weight already encodes "thick AND dark"; trusting
    # its argmax avoids the bias toward geometric center.
    flat = int(np.argmax(raw_weight))
    py_local, px_local = np.unravel_index(flat, raw_weight.shape)
    px = int(px_local + rx1)
    py = int(py_local + ry1)

    peak = float(raw_weight[py_local, px_local])
    edge_px = float(dist_obj[py_local, px_local])

    # Pick angle from PCA on the BRIGHT region of raw_weight (the dense
    # backbone of the clump). Whole-mask PCA would be pulled by tails /
    # branches; restricting to raw_weight >= 0.5 * peak keeps only the
    # high-density-and-thick core pixels, which is the meaningful axis the
    # gripper should clamp perpendicular to.
    bright = raw_weight >= max(0.5 * raw_weight.max(), 1e-6)
    angle = None
    if int(bright.sum()) >= 10:
        points_yx = np.column_stack(np.where(bright)).astype(np.float32)
        points_xy = points_yx[:, ::-1]
        angle = _pca_angle_deg(points_xy)

    return PickPointResult(
        pick_point_xy=[float(px), float(py)],
        pick_angle_deg=_image_axis_angle_to_pick_u_deg(angle),
        pick_method="density_thickness_max",
        pick_score=float(min(0.55 + 0.35 * peak + 0.10 * (edge_px / max_obj_dist), 0.95)),
        distance_to_edge_px=edge_px,
    )


def _pick_from_broad_dark_core(
    *,
    gray: np.ndarray,
    yolo_mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    metal_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    metal_distance: np.ndarray | None,
    safety_margin_px: int,
) -> PickPointResult | None:
    """Pick the visual center of the dark/thick body for broad impurities.

    This is intentionally different from the hair-line strategy.  In many bird's
    nest images the YOLO mask is a broad orange/brown feather-like area, while
    the actual robust plucking point is the darker dense knot/body inside that
    mask.  Skeleton or distance-transform methods can drift to an edge or tail.

    Strategy:
    1. Treat YOLO mask as the legal coarse object area.
    2. Remove metal plate / metal safety forbidden pixels.
    3. Build a darkness map from the original gray image.
    4. Use a smoothed dark-density map plus a weak mask-centroid prior to find
       the dominant dark mass, not a thin dark boundary.
    5. Return a point near that dark mass center, snapped to a safe mask pixel.
    """

    height, width = gray.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    pad = max(8, int(max(x2 - x1, y2 - y1) * 0.04))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(width, x2 + pad)
    ry2 = min(height, y2 + pad)
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi_gray = gray[ry1:ry2, rx1:rx2]
    roi_mask = (yolo_mask[ry1:ry2, rx1:rx2] > 0).astype(np.uint8)
    roi_forbidden = forbidden_mask[ry1:ry2, rx1:rx2]
    roi_metal = metal_mask[ry1:ry2, rx1:rx2]

    safe_mask = roi_mask.copy()
    safe_mask[roi_forbidden > 0] = 0
    safe_mask[roi_metal > 0] = 0
    safe_mask = _clean_mask(safe_mask)
    safe_area = int(safe_mask.sum())
    if safe_area < 80:
        return None

    values = roi_gray[safe_mask > 0].astype(np.float32)
    if values.size < 80:
        return None

    p10 = float(np.percentile(values, 10))
    p35 = float(np.percentile(values, 35))
    p70 = float(np.percentile(values, 70))
    p90 = float(np.percentile(values, 90))
    contrast = p90 - p10
    if contrast < 5.0:
        return None

    dist_obj = cv2.distanceTransform((safe_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    max_obj_dist = max(float(dist_obj.max()), 1.0)
    dist_norm = np.clip(dist_obj / max_obj_dist, 0.0, 1.0).astype(np.float32)

    # Keep broad-object points away from mask borders/tails, but do not erode so
    # much that a near-edge dark body disappears.
    inner_thr = max(1.5, min(10.0, max_obj_dist * 0.10))
    inner_mask = ((safe_mask > 0) & (dist_obj >= inner_thr)).astype(np.uint8)
    if int(inner_mask.sum()) < max(30, int(safe_area * 0.05)):
        inner_mask = safe_mask.copy()

    # Absolute darkness: darker-than-object pixels.  Use the object distribution,
    # not the whole image, so lighting changes are handled locally.
    dark = np.clip((p90 - roi_gray.astype(np.float32)) / max(contrast, 1.0), 0.0, 1.0)

    # Local darkness: hair/dense fibers darker than nearby nest material.
    k = _odd_kernel(max(31, min(151, int(max(roi_gray.shape[:2]) * 0.24))))
    local_bg = cv2.medianBlur(roi_gray, k).astype(np.float32)
    local_dark = np.clip((local_bg - roi_gray.astype(np.float32)) / max(contrast, 1.0), 0.0, 1.0)

    # The broad dark core should be supported by many nearby dark pixels.  Large
    # Gaussian smoothing makes the dense knot win over isolated dark strands.
    raw_weight = (0.70 * np.power(dark, 2.2) + 0.30 * np.power(local_dark, 1.6)).astype(np.float32)
    raw_weight[inner_mask == 0] = 0.0
    # Reduce edge/tail attraction without forcing the exact geometric center.
    raw_weight *= (0.35 + 0.65 * np.power(dist_norm, 0.45)).astype(np.float32)

    if float(raw_weight.max()) <= 1e-6:
        return None

    sigma = max(7.0, min(28.0, math.sqrt(float(safe_area)) * 0.16))
    density = cv2.GaussianBlur(raw_weight, (0, 0), sigmaX=sigma, sigmaY=sigma)
    density[inner_mask == 0] = 0.0

    valid_density = density[inner_mask > 0]
    if valid_density.size == 0 or float(valid_density.max()) <= 1e-6:
        return None

    mask_centroid = _mask_centroid(safe_mask)
    if mask_centroid is None:
        return None
    mask_cx, mask_cy = float(mask_centroid[0]), float(mask_centroid[1])

    # Candidate set: dark enough and density-supported.  Avoid relying on a
    # single maximum pixel; we want the center of a dark mass.
    density_thr = float(np.percentile(valid_density, 68))
    dark_gate = roi_gray.astype(np.float32) <= min(p70, p35 + 28.0)
    candidate = ((density >= density_thr) & (raw_weight >= 0.08) & dark_gate & (inner_mask > 0)).astype(np.uint8)
    if int(candidate.sum()) < max(20, int(safe_area * 0.006)):
        density_thr = float(np.percentile(valid_density, 56))
        candidate = ((density >= density_thr) & (raw_weight >= 0.045) & (inner_mask > 0)).astype(np.uint8)
    if int(candidate.sum()) < 12:
        return None

    # Close/expand dark fibers into a mass so the centroid lands in the user's
    # intended dark center, not on individual hair edges.
    merge_radius = max(5, min(21, int(math.sqrt(float(safe_area)) * 0.055)))
    merge_radius = merge_radius if merge_radius % 2 == 1 else merge_radius + 1
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_radius, merge_radius))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, merge_kernel, iterations=2)
    candidate = cv2.dilate(candidate, merge_kernel, iterations=1)
    candidate[inner_mask == 0] = 0
    candidate[roi_forbidden > 0] = 0
    candidate[roi_metal > 0] = 0
    if int(candidate.sum()) < 12:
        return None

    # Select central/thick dark components.  A component that is dark but very
    # far from the mask centroid is usually a tail/edge branch, not the main body.
    selected = _select_broad_dark_core_components(candidate, density, dist_obj, safe_mask)
    if selected is None or int(selected.sum()) < 12:
        return None

    sy, sx = np.where(selected > 0)
    if len(sx) == 0:
        return None

    # Weighted dark-core centroid.  Blend with the mask centroid more strongly
    # than previous versions; this is what pulls broad objects from tail/edge
    # strands back toward the visual center of the dark dense body.
    sw = (0.72 * density[sy, sx].astype(np.float64) + 0.28 * raw_weight[sy, sx].astype(np.float64) + 1e-6)
    dark_cx = float(np.sum(sx * sw) / np.sum(sw))
    dark_cy = float(np.sum(sy * sw) / np.sum(sw))
    target_x = 0.62 * dark_cx + 0.38 * mask_cx
    target_y = 0.62 * dark_cy + 0.38 * mask_cy

    # Snap to a safe pixel near the target that still has dark support and is not
    # on the object boundary.  Do not let metal distance pull the point to an edge;
    # metal is handled as a safety filter first.
    selected_dist = cv2.distanceTransform((selected > 0).astype(np.uint8), cv2.DIST_L2, 5)
    max_selected_dist = max(float(selected_dist.max()), 1.0)
    diag = max(float(np.hypot(selected.shape[1], selected.shape[0])), 1.0)

    best_xy: tuple[int, int, float | None] | None = None
    best_score = -1e9
    fallback_xy: tuple[int, int, float | None] | None = None
    fallback_score = -1e9
    ys, xs = np.where(selected > 0)
    for y, x in zip(ys, xs):
        full_x = int(x + rx1)
        full_y = int(y + ry1)
        metal_dist = float(metal_distance[full_y, full_x]) if metal_distance is not None else None
        target_dist = float(np.hypot(x - target_x, y - target_y))
        score = (
            2.7 * float(density[y, x])
            + 1.6 * float(raw_weight[y, x])
            + 1.1 * float(selected_dist[y, x]) / max_selected_dist
            + 0.9 * float(dist_obj[y, x]) / max_obj_dist
            - 3.4 * target_dist / diag
        )
        if score > fallback_score:
            fallback_score = score
            fallback_xy = (int(x), int(y), metal_dist)
        if metal_dist is not None and metal_dist < float(safety_margin_px):
            continue
        if score > best_score:
            best_score = score
            best_xy = (int(x), int(y), metal_dist)

    chosen = best_xy or fallback_xy
    if chosen is None:
        return None
    x, y, metal_dist = chosen

    geom = _mask_geometry(selected)
    edge_px = float(dist_obj[y, x]) if 0 <= y < dist_obj.shape[0] and 0 <= x < dist_obj.shape[1] else None
    dark_support = min(int(selected.sum()) / max(float(safe_area), 1.0), 0.75)
    density_strength = min(float(valid_density.max()), 1.0)
    safe_bonus = 0.0 if metal_dist is None else min(float(metal_dist) / max(float(safety_margin_px), 1.0), 1.0) * 0.06
    score = min(0.70 + dark_support * 0.15 + density_strength * 0.11 + safe_bonus, 0.96)

    return PickPointResult(
        pick_point_xy=[float(x + rx1), float(y + ry1)],
        pick_angle_deg=_image_axis_angle_to_pick_u_deg(geom["angle_deg"]),
        pick_method="broad_dark_core_center",
        pick_score=float(score),
        length_px=geom["length_px"],
        width_px=geom["width_px"],
        aspect_ratio=geom["aspect_ratio"],
        distance_to_edge_px=edge_px,
        hair_candidate_area=int(selected.sum()),
    )



def _select_broad_dark_core_components(
    candidate: np.ndarray,
    density: np.ndarray,
    dist_obj: np.ndarray,
    safe_mask: np.ndarray,
) -> np.ndarray | None:
    """Keep dark components that best represent the broad central body."""

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return candidate.astype(np.uint8) if int(candidate.sum()) > 0 else None

    centroid = _mask_centroid(safe_mask)
    if centroid is None:
        center = np.array([candidate.shape[1] / 2.0, candidate.shape[0] / 2.0], dtype=np.float32)
    else:
        center = np.array([float(centroid[0]), float(centroid[1])], dtype=np.float32)
    diag = max(float(np.hypot(candidate.shape[1], candidate.shape[0])), 1.0)
    max_dist = max(float(dist_obj.max()), 1.0)

    scored: list[tuple[float, np.ndarray]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 10:
            continue
        comp = (labels == label).astype(np.uint8)
        bbox = _mask_bbox(comp)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        geom = _mask_geometry(comp[y1:y2, x1:x2])
        length = geom["length_px"] or float(max(x2 - x1, y2 - y1))
        comp_width = geom["width_px"] or float(min(max(x2 - x1, 1), max(y2 - y1, 1)))
        aspect = float(length) / max(float(comp_width), 1.0)
        cy, cx = np.where(comp > 0)
        if len(cx) == 0:
            continue
        comp_center = np.array([float(cx.mean()), float(cy.mean())], dtype=np.float32)
        center_penalty = float(np.linalg.norm(comp_center - center)) / diag
        mean_density = float(density[comp > 0].mean())
        inner_score = float(dist_obj[comp > 0].mean()) / max_dist
        # Broad body = larger/thicker/interior/central.  Long thin branches can
        # be dark, but they should not beat the main dark mass.
        thin_penalty = max(0.0, aspect - 4.2) * 28.0
        score = (
            area * 0.45
            + min(float(comp_width), 65.0) * 6.5
            + mean_density * 140.0
            + inner_score * 70.0
            - center_penalty * 150.0
            - thin_penalty
        )
        scored.append((score, comp))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    out = np.zeros_like(candidate, dtype=np.uint8)
    for score, comp in scored[:5]:
        if score < best_score * 0.34 and int(out.sum()) > 0:
            continue
        out[comp > 0] = 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    out[safe_mask == 0] = 0
    return out.astype(np.uint8)



def _pick_from_dense_dark_region(
    *,
    gray: np.ndarray,
    yolo_mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    metal_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    metal_distance: np.ndarray | None,
    safety_margin_px: int,
) -> PickPointResult | None:
    """Pick the center of the dense dark body for broad foreign matter.

    Large feather-like impurities are not good skeleton targets: the skeleton
    often runs along an outer branch or a mask edge.  For this case the desired
    plucking point is the visually darker, thicker body/knot.  This function
    therefore builds a dark-density map inside the YOLO mask, finds the central
    dark mass, and returns a point near that mass center while still respecting
    the metal-plate safety margin.
    """

    height, width = gray.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    pad = max(12, int(max(x2 - x1, y2 - y1) * 0.08))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(width, x2 + pad)
    ry2 = min(height, y2 + pad)
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi_gray = gray[ry1:ry2, rx1:rx2]
    roi_mask = (yolo_mask[ry1:ry2, rx1:rx2] > 0).astype(np.uint8)
    roi_forbidden = forbidden_mask[ry1:ry2, rx1:rx2]
    roi_metal = metal_mask[ry1:ry2, rx1:rx2]

    safe_mask = roi_mask.copy()
    safe_mask[roi_forbidden > 0] = 0
    safe_mask[roi_metal > 0] = 0
    safe_mask = _clean_mask(safe_mask)
    safe_area = int(safe_mask.sum())
    if safe_area < 30:
        return None

    values = roi_gray[safe_mask > 0].astype(np.float32)
    if values.size < 30:
        return None

    p15 = float(np.percentile(values, 15))
    p35 = float(np.percentile(values, 35))
    p55 = float(np.percentile(values, 55))
    p80 = float(np.percentile(values, 80))
    if (p80 - p15) < 5.0:
        return None

    dist_obj = cv2.distanceTransform((safe_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    max_obj_dist = max(float(dist_obj.max()), 1.0)
    # Broad targets need an interior point.  A strong distance gate prevents the
    # selected point from drifting to a mask edge or a thin outer branch.
    inner_thr = max(2.0, min(16.0, max_obj_dist * 0.22))
    inner_mask = ((safe_mask > 0) & (dist_obj >= inner_thr)).astype(np.uint8)
    if int(inner_mask.sum()) < max(20, int(safe_area * 0.08)):
        inner_thr = max(1.0, min(8.0, max_obj_dist * 0.12))
        inner_mask = ((safe_mask > 0) & (dist_obj >= inner_thr)).astype(np.uint8)
    if int(inner_mask.sum()) < 12:
        inner_mask = safe_mask.copy()

    # Multi-scale darkness.  The central dark body is darker than the local
    # feather/background and remains strong after Gaussian smoothing; isolated
    # boundary strands lose influence after smoothing and centroid blending.
    raw_dark = np.clip(p80 - roi_gray.astype(np.float32), 0.0, 255.0)
    k = _odd_kernel(max(25, min(111, int(max(roi_gray.shape[:2]) * 0.20))))
    local_bg = cv2.medianBlur(roi_gray, k).astype(np.float32)
    local_dark = np.clip(local_bg - roi_gray.astype(np.float32), 0.0, 255.0)
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, blackhat_kernel).astype(np.float32)

    def _norm(arr: np.ndarray) -> np.ndarray:
        vals = arr[inner_mask > 0]
        hi = float(np.percentile(vals, 97)) if vals.size else 1.0
        if hi <= 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip(arr / hi, 0.0, 1.0).astype(np.float32)

    weight = 0.52 * _norm(raw_dark) + 0.34 * _norm(local_dark) + 0.14 * _norm(blackhat)
    weight[inner_mask == 0] = 0.0
    sigma = max(4.5, min(18.0, math.sqrt(float(safe_area)) * 0.10))
    dense = cv2.GaussianBlur(weight, (0, 0), sigmaX=sigma, sigmaY=sigma)
    dense[inner_mask == 0] = 0.0

    valid = dense[inner_mask > 0]
    if valid.size == 0 or float(valid.max()) <= 1e-6:
        return None
    if float(np.percentile(valid, 95) - np.percentile(valid, 40)) < 0.025:
        return None

    # Build a dark-body mask.  Use low gray value as a second condition so the
    # point is really pulled toward the visible dark mass, not just to a large
    # light-brown branch with a mild local contrast.
    raw_dark_norm = _norm(raw_dark)
    dark_gray_gate = roi_gray.astype(np.float32) <= min(p55, p35 + 18.0)
    thr = float(np.percentile(valid, 67))
    dark_body = ((dense >= thr) & (raw_dark_norm >= 0.18) & dark_gray_gate & (inner_mask > 0)).astype(np.uint8)
    if int(dark_body.sum()) < max(10, int(safe_area * 0.006)):
        thr = float(np.percentile(valid, 57))
        dark_body = ((dense >= thr) & (raw_dark_norm >= 0.12) & (inner_mask > 0)).astype(np.uint8)
    if int(dark_body.sum()) < 8:
        return None

    # Merge nearby dark fibers into one central body, then remove anything too
    # close to metal/forbidden areas again.
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dark_body = cv2.morphologyEx(dark_body, cv2.MORPH_CLOSE, close_k, iterations=1)
    dark_body = cv2.dilate(dark_body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    dark_body[inner_mask == 0] = 0
    dark_body[roi_forbidden > 0] = 0
    dark_body[roi_metal > 0] = 0

    selected = _select_central_dark_body_components(dark_body, dense, dist_obj, safe_mask)
    if selected is None or int(selected.sum()) < 8:
        return None

    # Weighted center of the selected dark body.  Blend slightly toward the mask
    # centroid to avoid endpoint/boundary selection, but keep the dark body as
    # the dominant evidence.
    mask_centroid = _mask_centroid(safe_mask)
    if mask_centroid is None:
        mask_centroid = (safe_mask.shape[1] / 2.0, safe_mask.shape[0] / 2.0)

    sy, sx = np.where(selected > 0)
    sw = (dense[sy, sx].astype(np.float64) + 0.20 * raw_dark_norm[sy, sx].astype(np.float64) + 1e-6)
    dark_cx = float(np.sum(sx * sw) / np.sum(sw))
    dark_cy = float(np.sum(sy * sw) / np.sum(sw))
    target_x = 0.78 * dark_cx + 0.22 * float(mask_centroid[0])
    target_y = 0.78 * dark_cy + 0.22 * float(mask_centroid[1])

    selected_dist = cv2.distanceTransform((selected > 0).astype(np.uint8), cv2.DIST_L2, 5)
    max_selected_dist = max(float(selected_dist.max()), 1.0)
    diag = max(float(np.hypot(selected.shape[1], selected.shape[0])), 1.0)

    best_xy: tuple[int, int, float | None] | None = None
    best_score = -1e9
    fallback_xy: tuple[int, int, float | None] | None = None
    fallback_score = -1e9

    ys, xs = np.where(selected > 0)
    for y, x in zip(ys, xs):
        full_x = int(x + rx1)
        full_y = int(y + ry1)
        metal_dist = float(metal_distance[full_y, full_x]) if metal_distance is not None else None
        dist_to_target = float(np.hypot(x - target_x, y - target_y))
        score = (
            3.8 * float(dense[y, x])
            + 1.4 * float(selected_dist[y, x]) / max_selected_dist
            + 1.2 * float(dist_obj[y, x]) / max_obj_dist
            - 3.2 * dist_to_target / diag
        )
        if score > fallback_score:
            fallback_score = score
            fallback_xy = (int(x), int(y), metal_dist)
        if metal_dist is not None and metal_dist < float(safety_margin_px):
            continue
        if score > best_score:
            best_score = score
            best_xy = (int(x), int(y), metal_dist)

    chosen = best_xy or fallback_xy
    if chosen is None:
        return None
    x, y, metal_dist = chosen

    geom = _mask_geometry(selected)
    edge_px = float(dist_obj[y, x]) if 0 <= y < dist_obj.shape[0] and 0 <= x < dist_obj.shape[1] else None
    support = min(int(selected.sum()) / max(float(safe_area), 1.0), 0.65)
    contrast = min(float(valid.max()), 1.0)
    safe_bonus = 0.0 if metal_dist is None else min(float(metal_dist) / max(float(safety_margin_px), 1.0), 1.0) * 0.08
    score = min(0.66 + support * 0.18 + contrast * 0.13 + safe_bonus, 0.96)

    return PickPointResult(
        pick_point_xy=[float(x + rx1), float(y + ry1)],
        pick_angle_deg=_image_axis_angle_to_pick_u_deg(geom["angle_deg"]),
        pick_method="dense_dark_mass_center",
        pick_score=float(score),
        length_px=geom["length_px"],
        width_px=geom["width_px"],
        aspect_ratio=geom["aspect_ratio"],
        distance_to_edge_px=edge_px,
        hair_candidate_area=int(selected.sum()),
    )



def _select_central_dark_body_components(
    candidate: np.ndarray,
    weight: np.ndarray,
    dist_obj: np.ndarray,
    safe_mask: np.ndarray,
) -> np.ndarray | None:
    """Select components that form the central dark body, not edge branches."""

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None

    mask_centroid = _mask_centroid(safe_mask)
    if mask_centroid is None:
        center = np.array([candidate.shape[1] / 2.0, candidate.shape[0] / 2.0], dtype=np.float32)
    else:
        center = np.array([float(mask_centroid[0]), float(mask_centroid[1])], dtype=np.float32)
    diag = max(float(np.hypot(candidate.shape[1], candidate.shape[0])), 1.0)
    max_dist = max(float(dist_obj.max()), 1.0)

    comps: list[tuple[float, np.ndarray]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        comp = (labels == label).astype(np.uint8)
        bbox = _mask_bbox(comp)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        geom = _mask_geometry(comp[y1:y2, x1:x2])
        length = geom["length_px"] or float(max(x2 - x1, y2 - y1))
        width = geom["width_px"] or float(min(max(x2 - x1, 1), max(y2 - y1, 1)))
        cy, cx = np.where(comp > 0)
        if len(cx) == 0:
            continue
        comp_center = np.array([float(cx.mean()), float(cy.mean())], dtype=np.float32)
        center_penalty = float(np.linalg.norm(comp_center - center)) / diag
        mean_weight = float(weight[comp > 0].mean())
        mean_inner = float(dist_obj[comp > 0].mean()) / max_dist
        aspect = float(length) / max(float(width), 1.0)
        # Favor larger, thicker, darker, more interior components.  Penalize
        # long thin components because those are usually edge branches on broad
        # feather masks.
        thin_penalty = max(0.0, aspect - 5.5) * 10.0
        score = area * 0.32 + mean_weight * 95.0 + min(float(width), 45.0) * 4.0 + mean_inner * 45.0 - center_penalty * 80.0 - thin_penalty
        comps.append((score, comp))

    if not comps:
        return None
    comps.sort(key=lambda item: item[0], reverse=True)
    best_score = comps[0][0]
    out = np.zeros_like(candidate, dtype=np.uint8)
    for score, comp in comps[:4]:
        if score < best_score * 0.42 and int(out.sum()) > 0:
            continue
        out[comp > 0] = 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    out[safe_mask == 0] = 0
    return out.astype(np.uint8)


def _select_dark_mass_components(candidate: np.ndarray, weight: np.ndarray, safe_mask: np.ndarray) -> np.ndarray | None:
    """Keep dark components that represent the central dark mass.

    For broad targets, several dark strands may belong to the same useful pick
    body.  This keeps components by area, average darkness, width and closeness
    to the weighted dark center, instead of blindly selecting a long boundary
    strand.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None

    ys_all, xs_all = np.where(candidate > 0)
    if len(xs_all) == 0:
        return None
    ww = weight[ys_all, xs_all].astype(np.float64) + 1e-6
    center = np.array([float(np.sum(xs_all * ww) / np.sum(ww)), float(np.sum(ys_all * ww) / np.sum(ww))], dtype=np.float32)
    diag = max(float(np.hypot(candidate.shape[1], candidate.shape[0])), 1.0)

    comps: list[tuple[float, np.ndarray]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        comp = (labels == label).astype(np.uint8)
        bbox = _mask_bbox(comp)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        geom = _mask_geometry(comp[y1:y2, x1:x2])
        length = geom["length_px"] or float(max(x2 - x1, y2 - y1))
        width = geom["width_px"] or float(min(max(x2 - x1, 1), max(y2 - y1, 1)))
        cy, cx = np.where(comp > 0)
        mean_w = float(weight[comp > 0].mean())
        comp_center = np.array([float(cx.mean()), float(cy.mean())], dtype=np.float32)
        center_penalty = float(np.linalg.norm(comp_center - center)) / diag
        # Prefer dense/thicker central components; penalize very long thin edge
        # strands that caused previous points to land on boundaries.
        edge_like_penalty = max(0.0, (float(length) / max(float(width), 1.0)) - 8.0) * 0.08
        score = area * 0.20 + mean_w * 80.0 + min(float(width), 35.0) * 3.5 - center_penalty * 65.0 - edge_like_penalty * 20.0
        comps.append((score, comp))

    if not comps:
        return None
    comps.sort(key=lambda item: item[0], reverse=True)

    out = np.zeros_like(candidate, dtype=np.uint8)
    best_score = comps[0][0]
    for score, comp in comps[:5]:
        if score < best_score * 0.45 and int(out.sum()) > 0:
            continue
        out[comp > 0] = 1

    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    out[safe_mask == 0] = 0
    return out.astype(np.uint8)

def _largest_reasonable_component_or_union(candidate: np.ndarray, safe_mask: np.ndarray) -> np.ndarray | None:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None

    safe_area = max(int(safe_mask.sum()), 1)
    components: list[tuple[float, int, np.ndarray]] = []
    safe_centroid = _mask_centroid(safe_mask)
    if safe_centroid is None:
        safe_center = np.array([candidate.shape[1] / 2.0, candidate.shape[0] / 2.0], dtype=np.float32)
    else:
        safe_center = np.array([safe_centroid[0], safe_centroid[1]], dtype=np.float32)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        comp = (labels == label).astype(np.uint8)
        bbox = _mask_bbox(comp)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        # Reject components that are almost the entire object edge/rim.
        if area > safe_area * 0.82 and min(w, h) < 5:
            continue
        cent = _mask_centroid(comp)
        if cent is None:
            continue
        center_dist = float(np.hypot(cent[0] - safe_center[0], cent[1] - safe_center[1]))
        geom = _mask_geometry(comp[y1:y2, x1:x2])
        width = geom["width_px"] or float(min(w, h))
        score = area * 1.0 - center_dist * 0.45 + min(width, 28.0) * 6.0
        components.append((score, label, comp))

    if not components:
        return None

    components.sort(key=lambda item: item[0], reverse=True)
    # Use a small union of the strongest dark components; this handles feather
    # centers that are split into several dark strands.
    out = np.zeros_like(candidate, dtype=np.uint8)
    for _, _, comp in components[:3]:
        out[comp > 0] = 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    out[safe_mask == 0] = 0
    return out.astype(np.uint8)


def _pick_from_dark_line_refinement(
    *,
    gray: np.ndarray,
    yolo_mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    metal_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    metal_distance: np.ndarray | None,
    safety_margin_px: int,
) -> PickPointResult | None:
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    pad = max(18, int(max(x2 - x1, y2 - y1) * 0.22))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(width, x2 + pad)
    ry2 = min(height, y2 + pad)
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi_gray = gray[ry1:ry2, rx1:rx2]
    roi_yolo = yolo_mask[ry1:ry2, rx1:rx2]
    roi_forbidden = forbidden_mask[ry1:ry2, rx1:rx2]
    roi_metal = metal_mask[ry1:ry2, rx1:rx2]

    # Coarse search area: YOLO mask dilated plus bbox rectangle.  This lets us
    # recover hair pixels just outside an imperfect mask, but prevents searching
    # unrelated regions.
    coarse = np.zeros_like(roi_gray, dtype=np.uint8)
    bx1 = x1 - rx1
    by1 = y1 - ry1
    bx2 = x2 - rx1
    by2 = y2 - ry1
    coarse[max(0, by1):min(coarse.shape[0], by2), max(0, bx1):min(coarse.shape[1], bx2)] = 1
    if int(roi_yolo.sum()) > 0:
        coarse = np.maximum(coarse, _dilate_binary(roi_yolo, max(7, pad // 2)))
    coarse[roi_forbidden > 0] = 0
    if int(coarse.sum()) < 8:
        return None

    # Dark thin line response.  A median/blurred background estimates the local
    # light nest/fixture color; real hairs are darker than this background.
    k = _odd_kernel(max(17, min(51, int(max(roi_gray.shape[:2]) * 0.12))))
    local_bg = cv2.medianBlur(roi_gray, k)
    dark_response = cv2.subtract(local_bg, roi_gray)

    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    blackhat = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
    response = cv2.max(dark_response, blackhat)

    valid_values = response[coarse > 0]
    if valid_values.size == 0:
        return None
    # Adaptive threshold: high enough to ignore orange/gray mask-like bulk, low
    # enough to keep weak fine hairs.
    response_thr = max(7.0, float(np.percentile(valid_values, 82)))
    gray_values = roi_gray[coarse > 0]
    gray_thr = float(np.percentile(gray_values, 55))

    candidate = ((response >= response_thr) & (roi_gray <= gray_thr + 18) & (coarse > 0)).astype(np.uint8)
    candidate[roi_forbidden > 0] = 0
    candidate[roi_metal > 0] = 0
    candidate = _clean_line_candidate(candidate)
    if int(candidate.sum()) < 6:
        return None

    component = _select_best_hair_component(candidate, response, metal_distance, rx1, ry1)
    if component is None or int(component.sum()) < 6:
        return None

    comp_bbox = _mask_bbox(component)
    if comp_bbox is None:
        return None
    cx1, cy1, cx2, cy2 = comp_bbox
    comp_roi = component[cy1:cy2, cx1:cx2]
    geom = _mask_geometry(comp_roi)
    line_result = _pick_from_skeleton(
        comp_roi,
        offset_x=rx1 + cx1,
        offset_y=ry1 + cy1,
        angle_deg=geom["angle_deg"],
        aspect_ratio=geom["aspect_ratio"],
        metal_distance=metal_distance,
        forbidden_mask=forbidden_mask,
        method="dark_line_skeleton_midpoint",
    )
    if line_result is None:
        centroid = _mask_centroid(comp_roi)
        if centroid is None:
            return None
        px, py = centroid
        line_result = PickPointResult(
            pick_point_xy=[float(px + rx1 + cx1), float(py + ry1 + cy1)],
            pick_angle_deg=_image_axis_angle_to_pick_u_deg(geom["angle_deg"]),
            pick_method="dark_line_centroid",
            pick_score=0.62,
        )

    line_result.length_px = geom["length_px"]
    line_result.width_px = geom["width_px"]
    line_result.aspect_ratio = geom["aspect_ratio"]
    line_result.hair_candidate_area = int(component.sum())
    return line_result


def _select_best_hair_component(
    candidate: np.ndarray,
    response: np.ndarray,
    metal_distance: np.ndarray | None,
    offset_x: int,
    offset_y: int,
) -> np.ndarray | None:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    best_score = -1e9
    best = None
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 5:
            continue
        comp = (labels == label).astype(np.uint8)
        bbox = _mask_bbox(comp)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        comp_roi = comp[y1:y2, x1:x2]
        geom = _mask_geometry(comp_roi)
        length = geom["length_px"] or float(max(x2 - x1, y2 - y1))
        width = geom["width_px"] or float(min(max(x2 - x1, 1), max(y2 - y1, 1)))
        aspect = geom["aspect_ratio"] or (length / max(width, 1.0))
        if length < 5:
            continue
        # Very large, thick components are usually broad YOLO/mask artifacts, not hair.
        thickness_penalty = max(0.0, width - 12.0) * 0.5
        mean_response = float(response[labels == label].mean()) if area > 0 else 0.0
        metal_bonus = 0.0
        if metal_distance is not None:
            ys, xs = np.where(comp > 0)
            if len(xs):
                md = metal_distance[ys + offset_y, xs + offset_x]
                metal_bonus = min(float(np.percentile(md, 40)) / 30.0, 1.5)
        score = area * 0.08 + length * 0.45 + min(aspect, 8.0) * 4.0 + mean_response * 0.25 + metal_bonus * 8.0 - thickness_penalty
        if score > best_score:
            best_score = score
            best = comp
    return best


def _clean_line_candidate(binary: np.ndarray) -> np.ndarray:
    if binary.size == 0:
        return binary.astype(np.uint8)
    binary = binary.astype(np.uint8)
    # Close tiny gaps in weak hair pixels but avoid turning a broad orange/gray
    # area into one huge component.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    out = np.zeros_like(cleaned)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area > 2500 and min(w, h) > 45:
            continue
        out[labels == label] = 1
    return out.astype(np.uint8)


def _detect_metal_plate(
    gray: np.ndarray,
    *,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    # The metal pressing plate is the large, very dark grid.  Hair / foreign
    # matter can also be long-and-dark — geometric filtering alone cannot
    # tell them apart from real plate bars. To avoid false positives, the
    # caller passes the union of all YOLO foreign_matter masks for the
    # frame as ``exclude_mask``: pixels YOLO already claims as foreign
    # matter cannot also be metal by definition.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dark = (blur < 68).astype(np.uint8)
    if exclude_mask is not None and exclude_mask.size:
        # Bitwise-AND-NOT — remove YOLO foreign-matter pixels from candidates.
        dark[exclude_mask > 0] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    h, w = gray.shape[:2]
    image_area = h * w
    metal = np.zeros_like(dark)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        # Long thin dark bar (true plate frame), OR a single component covering
        # ≥5% of the image (catastrophic case: frame fused into one blob).
        if (min(bw, bh) >= 18 and max(bw, bh) >= 70 and aspect >= 3.0) or area >= int(image_area * 0.05):
            metal[labels == label] = 1
    return metal.astype(np.uint8)


def _finalize_result(
    result: PickPointResult,
    metal_distance: np.ndarray | None,
    forbidden_mask: np.ndarray,
    safety_margin_px: int,
) -> PickPointResult:
    # Record distance_to_metal_px on the result so the AbstainNearMetalStep
    # downstream can filter unpickable targets. _finalize_result deliberately
    # does NOT set any pick_safe flag or rewrite pick_method / pick_score —
    # the half-state "kept but marked unsafe" was replaced by an explicit
    # filter step.
    px = int(round(result.pick_point_xy[0]))
    py = int(round(result.pick_point_xy[1]))
    if metal_distance is not None and 0 <= py < metal_distance.shape[0] and 0 <= px < metal_distance.shape[1]:
        result.distance_to_metal_px = float(metal_distance[py, px])
    return result


def _clip_bbox_to_mask(
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return (
        max(0, min(width, int(math.floor(x1)))),
        max(0, min(height, int(math.floor(y1)))),
        max(0, min(width, int(math.ceil(x2)))),
        max(0, min(height, int(math.ceil(y2)))),
    )


def _largest_connected_component(binary: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary.astype(np.uint8)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)


def _clean_mask(binary: np.ndarray) -> np.ndarray:
    if binary.size == 0:
        return binary.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    if int(cleaned.sum()) == 0:
        return binary.astype(np.uint8)
    return cleaned.astype(np.uint8)


def _mask_bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _mask_centroid(binary: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments(binary.astype(np.uint8), binaryImage=True)
    if moments["m00"] == 0:
        return None
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def _mask_geometry(binary: np.ndarray) -> dict[str, float | None]:
    points_yx = np.column_stack(np.where(binary > 0)).astype(np.float32)
    if len(points_yx) < 2:
        return {"length_px": None, "width_px": None, "aspect_ratio": None, "angle_deg": None}

    points_xy = points_yx[:, ::-1]
    rect = cv2.minAreaRect(points_xy)
    (rw, rh) = rect[1]
    if rw <= 0 or rh <= 0:
        return {"length_px": None, "width_px": None, "aspect_ratio": None, "angle_deg": None}

    length_px = float(max(rw, rh))
    width_px = float(min(rw, rh))
    aspect_ratio = float(length_px / max(width_px, 1e-6))
    angle_deg = _pca_angle_deg(points_xy)
    return {
        "length_px": length_px,
        "width_px": width_px,
        "aspect_ratio": aspect_ratio,
        "angle_deg": angle_deg,
    }


def _shape_descriptors(binary: np.ndarray) -> dict[str, Any]:
    """Angle-invariant shape descriptors for the 3-class dispatch.

    Returns aspect_ratio + angle_deg + length_px + width_px from minAreaRect,
    plus the two dimensionless ratios:
      - extent   = mask_area / minAreaRect_area  (how full the rotated bbox is)
      - solidity = mask_area / convex_hull_area  (how close the mask is to its hull)
    Solidity is the linchpin for curved vs straight/clump separation: bent
    masks have a convex hull that fills in the bend, so solidity drops well
    below 1 while staying angle-invariant.
    """

    base = _mask_geometry(binary)
    out: dict[str, Any] = {
        "length_px": base["length_px"],
        "width_px": base["width_px"],
        "aspect_ratio": base["aspect_ratio"],
        "angle_deg": base["angle_deg"],
        "mask_area": int((binary > 0).sum()),
        "extent": None,
        "solidity": None,
    }
    if base["length_px"] is None or base["width_px"] is None:
        return out

    min_rect_area = float(base["length_px"]) * float(base["width_px"])
    if min_rect_area > 0:
        out["extent"] = float(out["mask_area"] / min_rect_area)

    contours, _ = cv2.findContours(
        (binary > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 0:
            out["solidity"] = float(out["mask_area"] / hull_area)

    return out


def _has_dense_core(gray: np.ndarray | None, mask: np.ndarray | None) -> bool:
    """Detect internal density structure inside the mask via gray statistics.

    Big tangled clumps of bird's nest material have a dark fibrous core
    surrounded by lighter wisps; pure thin hair has uniform darkness along
    the strand; small uniform feather puffs have low spread. We declare a
    mask to have a dense core if both:
      - contrast (p90 - p10) of gray within mask is large enough, and
      - a non-trivial fraction of pixels sit well below the mask median.
    """

    if gray is None or mask is None:
        return False
    if mask.dtype != np.bool_:
        mask = mask > 0
    if int(mask.sum()) < 100:
        return False
    vals = gray[mask].astype(np.float32)
    if vals.size < 100:
        return False
    p10 = float(np.percentile(vals, 10))
    p50 = float(np.percentile(vals, 50))
    p90 = float(np.percentile(vals, 90))
    contrast = p90 - p10
    # "dark" relative to mask interior; threshold sits halfway between p10 and p50.
    dark_threshold = p50 - 0.6 * (p50 - p10)
    dark_fraction = float((vals < dark_threshold).mean())
    return (
        contrast >= DENSE_CLUMP_MIN_CONTRAST
        and dark_fraction >= DENSE_CLUMP_MIN_DARK_FRACTION
    )


def _classify_shape(
    desc: dict[str, Any],
    *,
    gray: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> str:
    """Map shape + density signals to a class.

    Returns one of {straight_thin, curved, down_clump, dense_clump, ambiguous}.

    Order:
      1. dense_clump first when gray+mask provided — a big mask with internal
         density structure should be picked at its dark core, regardless of
         overall outline shape. This catches big tangled chunks that would
         otherwise leak into 'curved' on low solidity.
      2. curved — bent hair fails the convex-hull test (solidity drops).
      3. straight_thin — clear long thin strand.
      4. down_clump — small uniform feather puff (no internal structure).
      5. ambiguous fallback.
    """
    aspect = desc.get("aspect_ratio")
    extent = desc.get("extent")
    solidity = desc.get("solidity")
    mask_area = desc.get("mask_area", 0) or 0
    if aspect is None or extent is None or solidity is None:
        return "ambiguous"

    if (
        gray is not None
        and mask is not None
        and mask_area >= DENSE_CLUMP_MIN_AREA_PX
        and _has_dense_core(gray, mask)
    ):
        return "dense_clump"

    if solidity < CURVED_SOLIDITY_MAX and aspect >= CURVED_ASPECT_MIN:
        return "curved"
    if (
        aspect >= STRAIGHT_THIN_ASPECT_MIN
        and extent >= STRAIGHT_THIN_EXTENT_MIN
        and solidity >= STRAIGHT_THIN_SOLIDITY_MIN
    ):
        return "straight_thin"
    if (
        aspect <= DOWN_CLUMP_ASPECT_MAX
        and extent >= DOWN_CLUMP_EXTENT_MIN
        and solidity >= DOWN_CLUMP_SOLIDITY_MIN
    ):
        return "down_clump"
    return "ambiguous"


def _pca_angle_deg(points_xy: np.ndarray) -> float | None:
    if len(points_xy) < 2:
        return None
    centered = points_xy.astype(np.float32) - points_xy.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    return _normalize_angle_deg(angle)


def _normalize_angle_deg(angle: float) -> float:
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return float(angle)


def _image_axis_angle_to_pick_u_deg(angle: float | None) -> float | None:
    """Map signed image-axis angle into calibrated robot U in [0, 180)."""
    if angle is None:
        return None

    # `angle` is measured in image coordinates where +Y points downward.
    # The validated robot-U chart is defined on the displayed image with +Y up,
    # so the axis angle must be mirrored before the fixed U offset is applied.
    u_deg = -float(angle) + PICK_U_ZERO_OFFSET_DEG
    while u_deg < 0.0:
        u_deg += 180.0
    while u_deg >= 180.0:
        u_deg -= 180.0
    return u_deg


def _pick_from_skeleton(
    binary: np.ndarray,
    offset_x: int,
    offset_y: int,
    angle_deg: float | None,
    aspect_ratio: float | None,
    metal_distance: np.ndarray | None = None,
    forbidden_mask: np.ndarray | None = None,
    method: str = "skeleton_midpoint",
) -> PickPointResult | None:
    skeleton = _zhang_suen_thinning(binary)
    ys, xs = np.where(skeleton > 0)
    if len(xs) < 3:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)
    center = points.mean(axis=0)

    if angle_deg is None:
        angle_deg = _pca_angle_deg(points)
    if angle_deg is None:
        return None

    theta = math.radians(angle_deg)
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    projections = (points - center) @ direction
    order = np.argsort(projections)
    ordered = points[order]

    # Avoid endpoints.  Pick from the middle 35%-65% of the skeleton and prefer
    # points farther from the metal plate.
    n = len(ordered)
    start = max(0, int(n * 0.35))
    end = min(n, max(start + 1, int(n * 0.65)))
    middle = ordered[start:end]
    if len(middle) == 0:
        middle = ordered[[n // 2]]

    selected = middle[len(middle) // 2]
    if metal_distance is not None:
        best_score = -1e9
        for point in middle:
            px = int(round(float(point[0] + offset_x)))
            py = int(round(float(point[1] + offset_y)))
            if not (0 <= py < metal_distance.shape[0] and 0 <= px < metal_distance.shape[1]):
                continue
            if forbidden_mask is not None and forbidden_mask.size and forbidden_mask[py, px] > 0:
                continue
            dist = float(metal_distance[py, px])
            centrality = 1.0 - abs(float((point - center) @ direction)) / max(float(np.ptp(projections)), 1.0)
            score = dist + 12.0 * centrality
            if score > best_score:
                best_score = score
                selected = point

    px, py = selected
    aspect_score = min((aspect_ratio or 1.0) / 8.0, 1.0)
    support_score = min(len(xs) / 80.0, 1.0)
    score = float(0.55 + 0.30 * aspect_score + 0.15 * support_score)

    return PickPointResult(
        pick_point_xy=[float(px + offset_x), float(py + offset_y)],
        pick_angle_deg=_image_axis_angle_to_pick_u_deg(float(angle_deg)),
        pick_method=method,
        pick_score=min(score, 0.98),
        distance_to_edge_px=None,
    )


def _pick_from_curved_skeleton(
    binary: np.ndarray,
    offset_x: int,
    offset_y: int,
    metal_distance: np.ndarray | None = None,
    forbidden_mask: np.ndarray | None = None,
) -> PickPointResult | None:
    """Pick the arc-length midpoint of a curved skeleton.

    For bent hair, PCA-along-the-chord places the midpoint near the chord
    center, which can fall outside the hair entirely (think of a U-shaped
    strand: the chord midpoint is inside the bend, not on the hair). This
    walks the skeleton's longest geodesic path and returns the point at
    half arc length.

    Implementation: thin the mask, treat each 1-pixel-wide skeleton pixel as
    a graph node connected to its 8-neighbors; the longest path between any
    two skeleton pixels approximates the tree diameter (Zhang-Suen output is
    usually a quasi-tree). Two BFS passes recover that path; cumulative
    Euclidean arc length picks the midpoint; a small window around the
    midpoint is searched to maximize ``metal_distance`` while staying out of
    ``forbidden_mask``.
    """

    skeleton = _zhang_suen_thinning(binary)
    pts_yx = np.column_stack(np.where(skeleton > 0))
    if len(pts_yx) < 5:
        return None

    skel_set = {(int(y), int(x)) for y, x in pts_yx}

    def neighbors(node: tuple[int, int]):
        y, x = node
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (y + dy, x + dx)
                if nb in skel_set:
                    yield nb, math.hypot(dx, dy)

    def bfs(start: tuple[int, int]):
        dist: dict[tuple[int, int], float] = {start: 0.0}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        far = start
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            cur = queue.popleft()
            for nb, step in neighbors(cur):
                if nb in dist:
                    continue
                dist[nb] = dist[cur] + step
                parent[nb] = cur
                if dist[nb] > dist[far]:
                    far = nb
                queue.append(nb)
        return far, parent, dist

    start = (int(pts_yx[0, 0]), int(pts_yx[0, 1]))
    a, _, _ = bfs(start)
    b, parent_b, dist_b = bfs(a)
    total_arc = float(dist_b[b])
    if total_arc < 4.0:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = b
    while cur is not None:
        path.append(cur)
        cur = parent_b[cur]
    path.reverse()
    if len(path) < 3:
        return None

    # Cumulative arc length along the path.
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]))

    target = total_arc / 2.0
    mid_idx = 0
    for i, val in enumerate(cum):
        if val >= target:
            mid_idx = i
            break

    # Search ±10% of the path around the midpoint for the safest, most central point.
    window = max(2, int(len(path) * 0.10))
    lo = max(1, mid_idx - window)
    hi = min(len(path) - 2, mid_idx + window)
    if hi < lo:
        lo, hi = mid_idx, mid_idx

    best_idx = mid_idx
    best_score = -1e9
    for i in range(lo, hi + 1):
        y, x = path[i]
        gx = x + offset_x
        gy = y + offset_y
        if forbidden_mask is not None and forbidden_mask.size:
            if 0 <= gy < forbidden_mask.shape[0] and 0 <= gx < forbidden_mask.shape[1]:
                if forbidden_mask[gy, gx] > 0:
                    continue
        metal_dist = 0.0
        if metal_distance is not None and 0 <= gy < metal_distance.shape[0] and 0 <= gx < metal_distance.shape[1]:
            metal_dist = float(metal_distance[gy, gx])
        centrality = 1.0 - abs(i - mid_idx) / max(window, 1)
        score = metal_dist + 10.0 * centrality
        if score > best_score:
            best_score = score
            best_idx = i

    py, px = path[best_idx]

    # Local tangent for pick_angle_deg: small finite-difference along the path.
    j_prev = max(0, best_idx - 3)
    j_next = min(len(path) - 1, best_idx + 3)
    dy = path[j_next][0] - path[j_prev][0]
    dx = path[j_next][1] - path[j_prev][1]
    if dx == 0 and dy == 0:
        angle = None
    else:
        angle = _normalize_angle_deg(math.degrees(math.atan2(float(dy), float(dx))))

    return PickPointResult(
        pick_point_xy=[float(px + offset_x), float(py + offset_y)],
        pick_angle_deg=_image_axis_angle_to_pick_u_deg(angle),
        pick_method="curved_skeleton_arc_midpoint",
        pick_score=0.78,
    )


def _pick_from_distance_transform(
    binary: np.ndarray,
    offset_x: int,
    offset_y: int,
    angle_deg: float | None,
    aspect_ratio: float | None,
) -> PickPointResult | None:
    if int(binary.sum()) == 0:
        return None
    dist = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    if max_val <= 0:
        return None
    px, py = max_loc
    edge_score = min(float(max_val) / 12.0, 1.0)
    shape_bonus = 0.10 if aspect_ratio is not None and aspect_ratio < 2.5 else 0.0
    return PickPointResult(
        pick_point_xy=[float(px + offset_x), float(py + offset_y)],
        pick_angle_deg=_image_axis_angle_to_pick_u_deg(angle_deg),
        pick_method="safe_distance_transform_center",
        pick_score=float(min(0.50 + 0.30 * edge_score + shape_bonus, 0.90)),
        distance_to_edge_px=float(max_val),
    )


def _dilate_binary(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (binary > 0).astype(np.uint8)
    k = int(radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate((binary > 0).astype(np.uint8), kernel, iterations=1)


def _odd_kernel(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Pure NumPy Zhang-Suen thinning for small/medium ROI masks."""

    img = (binary > 0).astype(np.uint8)
    if img.ndim != 2 or int(img.sum()) == 0:
        return img

    img = np.pad(img, ((1, 1), (1, 1)), mode="constant")
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2 = img[:-2, 1:-1]
            p3 = img[:-2, 2:]
            p4 = img[1:-1, 2:]
            p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]
            p7 = img[2:, :-2]
            p8 = img[1:-1, :-2]
            p9 = img[:-2, :-2]
            center = img[1:-1, 1:-1]

            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            if step == 0:
                m1 = p2 * p4 * p6
                m2 = p4 * p6 * p8
            else:
                m1 = p2 * p4 * p8
                m2 = p2 * p6 * p8

            marker = (
                (center == 1)
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (transitions == 1)
                & (m1 == 0)
                & (m2 == 0)
            )
            if np.any(marker):
                center[marker] = 0
                changed = True

    return img[1:-1, 1:-1].astype(np.uint8)
