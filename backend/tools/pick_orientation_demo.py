#!/usr/bin/env python3
"""Offline detection + pick-orientation estimation for a test image.

Workflow:
1. Run the existing detection pipeline on one image.
2. For each detection, crop the ROI.
3. Use classical CV inside the ROI to isolate the dark target.
4. Estimate the dominant axis with PCA and convert it into a gripper yaw.
5. Save an annotated image plus a JSON report.

This is meant for debugging pick-angle estimation before wiring it into the
runtime target-response path.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.coordinate_transform import CoordinateTransformer, ExtrinsicCalibration
from src.core.pick_orientation import (
    DetectedBBox,
    OrientationEstimate,
    build_dark_mask,
    canonicalize_axis_yaw_deg,
    compute_contour_centroid,
    crop_roi,
    estimate_orientation_from_binary_mask,
    normalize_angle_deg,
    select_primary_contour,
)


@dataclass(frozen=True, slots=True)
class DetectionOrientationResult:
    """Full per-detection result for report serialization."""

    detection_index: int
    object_type: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    bbox_center_x: float
    bbox_center_y: float
    refined_center_x: float
    refined_center_y: float
    world_x_mm: float | None
    world_y_mm: float | None
    image_angle_deg: float
    world_axis_angle_deg: float
    target_yaw_deg: float
    contour_area_px: float
    elongation_ratio: float
    mask_area_px: int
    debug_roi_path: str
    debug_mask_path: str


def parse_pair(raw: str, *, name: str) -> tuple[float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be two numbers separated by a comma")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"{name} must be two numbers separated by a comma") from exc


def analyze_detection(
    image: np.ndarray,
    *,
    bbox: DetectedBBox,
    confidence: float,
    object_type: str,
    detection_index: int,
    output_dir: Path,
    padding: int,
    min_contour_area: float,
    transformer: CoordinateTransformer | None,
    arm_pose: tuple[float, float],
    flip_y: bool,
) -> DetectionOrientationResult | None:
    """Run contour-based pick analysis for one detection bbox."""
    roi, (ox1, oy1, ox2, oy2) = crop_roi(image, bbox, padding=padding)
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

    selected_area = cv2.contourArea(selected_contour)
    contour_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [selected_contour], -1, 255, thickness=-1)

    orientation = estimate_orientation_from_binary_mask(contour_mask, flip_y=flip_y)
    if orientation is None:
        return None

    refined_center_x = orientation.centroid_x + ox1
    refined_center_y = orientation.centroid_y + oy1

    world_x_mm: float | None = None
    world_y_mm: float | None = None
    if transformer is not None:
        world_point = transformer.pixel_to_world(
            refined_center_x,
            refined_center_y,
            arm_pose[0],
            arm_pose[1],
        )
        world_x_mm = world_point.x
        world_y_mm = world_point.y

    debug_roi_path = output_dir / f"detection_{detection_index:02d}_roi.png"
    debug_mask_path = output_dir / f"detection_{detection_index:02d}_mask.png"

    roi_vis = roi.copy()
    cv2.drawContours(roi_vis, [selected_contour], -1, (0, 255, 0), 2)
    center_local = (
        int(round(orientation.centroid_x)),
        int(round(orientation.centroid_y)),
    )
    cv2.circle(roi_vis, center_local, 4, (255, 255, 0), -1)

    axis_len = int(round(max(bbox.width, bbox.height) * 0.6))
    p1 = (
        int(round(orientation.centroid_x - orientation.axis_vx * axis_len)),
        int(round(orientation.centroid_y - orientation.axis_vy * axis_len)),
    )
    p2 = (
        int(round(orientation.centroid_x + orientation.axis_vx * axis_len)),
        int(round(orientation.centroid_y + orientation.axis_vy * axis_len)),
    )
    cv2.line(roi_vis, p1, p2, (0, 0, 255), 2)
    cv2.imwrite(str(debug_roi_path), roi_vis)
    cv2.imwrite(str(debug_mask_path), contour_mask)

    return DetectionOrientationResult(
        detection_index=detection_index,
        object_type=object_type,
        confidence=confidence,
        bbox_x1=bbox.x1,
        bbox_y1=bbox.y1,
        bbox_x2=bbox.x2,
        bbox_y2=bbox.y2,
        bbox_center_x=bbox_cx,
        bbox_center_y=bbox_cy,
        refined_center_x=refined_center_x,
        refined_center_y=refined_center_y,
        world_x_mm=world_x_mm,
        world_y_mm=world_y_mm,
        image_angle_deg=orientation.image_angle_deg,
        world_axis_angle_deg=orientation.world_axis_angle_deg,
        target_yaw_deg=orientation.target_yaw_deg,
        contour_area_px=selected_area,
        elongation_ratio=orientation.elongation_ratio,
        mask_area_px=int(np.count_nonzero(contour_mask)),
        debug_roi_path=str(debug_roi_path),
        debug_mask_path=str(debug_mask_path),
    )


def render_full_visualization(
    image: np.ndarray,
    *,
    results: list[DetectionOrientationResult],
    output_path: Path,
) -> None:
    """Render all detection results onto the full image."""
    canvas = image.copy()

    for result in results:
        x1 = int(round(result.bbox_x1))
        y1 = int(round(result.bbox_y1))
        x2 = int(round(result.bbox_x2))
        y2 = int(round(result.bbox_y2))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cx = int(round(result.refined_center_x))
        cy = int(round(result.refined_center_y))
        cv2.drawMarker(canvas, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)

        axis_len = int(round(max(x2 - x1, y2 - y1) * 0.6))
        theta = math.radians(result.image_angle_deg)
        p1 = (
            int(round(cx - math.cos(theta) * axis_len)),
            int(round(cy - math.sin(theta) * axis_len)),
        )
        p2 = (
            int(round(cx + math.cos(theta) * axis_len)),
            int(round(cy + math.sin(theta) * axis_len)),
        )
        cv2.line(canvas, p1, p2, (0, 0, 255), 2)

        label = (
            f"#{result.detection_index} {result.object_type} "
            f"conf={result.confidence:.2f} yaw={result.target_yaw_deg:.1f}deg"
        )
        cv2.putText(
            canvas,
            label,
            (x1, max(24, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run detection on one image and estimate pick yaw from cropped ROIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
  cd backend
  python -m tools.pick_orientation_demo \
      --image ../tmp/example.bmp \
      --config config/settings.dev.yaml \
      --arm-pose 0,0
""",
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--config",
        default="config/settings.dev.yaml",
        help="Backend config used to build the detection pipeline",
    )
    parser.add_argument(
        "--object-type",
        default=None,
        help="Optional object_type filter after detection",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum detection confidence",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=20,
        help="Maximum number of detections to analyze",
    )
    parser.add_argument(
        "--detection-index",
        type=int,
        default=None,
        help="If set, analyze only this filtered detection index (0-based)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Extra ROI padding around each bbox in pixels",
    )
    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=80.0,
        help="Minimum contour area in ROI mask",
    )
    parser.add_argument(
        "--arm-pose",
        default="0,0",
        help="Current flange position x,y in mm for optional world-coordinate output",
    )
    parser.add_argument(
        "--extrinsic",
        default="config/calibration/extrinsic.yaml",
        help="Extrinsic YAML path for world-coordinate conversion",
    )
    parser.add_argument(
        "--intrinsic",
        default="config/calibration/camera_intrinsic.yaml",
        help="Intrinsic YAML path for world-coordinate conversion",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to data/debug/pick_orientation/<image_stem>",
    )
    return parser


def load_transformer(
    *,
    extrinsic_path: str,
    intrinsic_path: str,
) -> tuple[CoordinateTransformer | None, bool]:
    extrinsic = Path(extrinsic_path)
    intrinsic = Path(intrinsic_path)
    if not extrinsic.exists() or not intrinsic.exists():
        return None, False
    calibration = ExtrinsicCalibration.load(extrinsic, intrinsic)
    return CoordinateTransformer(calibration), calibration.flip_y


def run_detection(image: np.ndarray, *, config_path: str):
    """Create and run the configured detection pipeline."""
    from main import create_pipeline
    from src.config import AppConfig

    cfg = AppConfig.from_yaml(config_path)
    pipeline = create_pipeline(cfg)
    return pipeline.run(image)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    arm_pose = parse_pair(args.arm_pose, name="arm pose")

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("data/debug/pick_orientation") / image_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    transformer, flip_y = load_transformer(
        extrinsic_path=args.extrinsic,
        intrinsic_path=args.intrinsic,
    )

    pipeline_result = run_detection(image, config_path=args.config)
    detections = sorted(
        pipeline_result.detections,
        key=lambda det: det.confidence,
        reverse=True,
    )
    detections = [
        det
        for det in detections
        if det.confidence >= args.min_confidence
        and (args.object_type is None or det.object_type == args.object_type)
    ]

    if args.detection_index is not None:
        if args.detection_index < 0 or args.detection_index >= len(detections):
            raise IndexError(
                f"--detection-index={args.detection_index} out of range for "
                f"{len(detections)} filtered detections"
            )
        detections = [detections[args.detection_index]]
    else:
        detections = detections[: args.max_detections]

    results: list[DetectionOrientationResult] = []

    for det_idx, detection in enumerate(detections):
        bbox = DetectedBBox(
            x1=float(detection.bbox.x1),
            y1=float(detection.bbox.y1),
            x2=float(detection.bbox.x2),
            y2=float(detection.bbox.y2),
        )
        result = analyze_detection(
            image,
            bbox=bbox,
            confidence=float(detection.confidence),
            object_type=str(detection.object_type),
            detection_index=det_idx,
            output_dir=output_dir,
            padding=args.padding,
            min_contour_area=args.min_contour_area,
            transformer=transformer,
            arm_pose=arm_pose,
            flip_y=flip_y,
        )
        if result is not None:
            results.append(result)

    annotated_path = output_dir / "annotated.png"
    report_path = output_dir / "report.json"
    render_full_visualization(image, results=results, output_path=annotated_path)

    report = {
        "image_path": str(image_path),
        "config_path": args.config,
        "arm_pose_mm": {"x": arm_pose[0], "y": arm_pose[1]},
        "flip_y": flip_y,
        "raw_detection_count": len(pipeline_result.detections),
        "analyzed_detection_count": len(results),
        "results": [asdict(result) for result in results],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Image: {image_path}")
    print(f"Raw detections: {len(pipeline_result.detections)}")
    print(f"Analyzed detections: {len(results)}")
    print(f"Annotated output: {annotated_path}")
    print(f"JSON report: {report_path}")
    for result in results:
        world = (
            f" world=({result.world_x_mm:.3f}, {result.world_y_mm:.3f})mm"
            if result.world_x_mm is not None and result.world_y_mm is not None
            else ""
        )
        print(
            f"  #{result.detection_index} {result.object_type} "
            f"conf={result.confidence:.3f} "
            f"yaw={result.target_yaw_deg:.2f}deg"
            f"{world}"
        )


if __name__ == "__main__":
    main()
