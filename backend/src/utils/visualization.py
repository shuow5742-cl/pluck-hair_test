"""Visualization utilities for drawing detections and targets."""

from typing import Dict, List

import cv2
import numpy as np

from autoweaver.pipeline import Detection
from src.tasks.stabilized_detection.stabilizer import StableTarget


def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    color_map: Dict[str, tuple] = None,
) -> np.ndarray:
    """Draw detection boxes on image.

    Args:
        image: Original image.
        detections: List of detections.
        color_map: Optional color mapping for object types.
            Defaults to blue (255, 0, 0) for unknown types.

    Returns:
        Annotated image copy.
    """
    if color_map is None:
        color_map = {}

    result = image.copy()

    for det in detections:
        # object_type is a string, not an enum
        obj_type = det.object_type if isinstance(det.object_type, str) else det.object_type.value
        color = color_map.get(obj_type, (255, 0, 0))

        # Draw bounding box
        pt1 = (int(det.bbox.x1), int(det.bbox.y1))
        pt2 = (int(det.bbox.x2), int(det.bbox.y2))
        cv2.rectangle(result, pt1, pt2, color, 2)

        # Draw label
        label = f"{obj_type}: {det.confidence:.2f}"
        label_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            2
        )

        # Label background
        cv2.rectangle(
            result,
            (pt1[0], pt1[1] - label_size[1] - 14),
            (pt1[0] + label_size[0], pt1[1]),
            color,
            -1
        )

        # Label text
        cv2.putText(
            result,
            label,
            (pt1[0], pt1[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

    return result


def draw_stable_targets(
    image: np.ndarray,
    stable_targets: List[StableTarget],
    color_map: Dict[str, tuple] = None,
) -> np.ndarray:
    """Draw stable target boxes on image.

    Uses center + width/height from StableTarget for stable display.
    If world_x/world_y are set on the target, displays world coordinates.

    Args:
        image: Original image.
        stable_targets: List of stable targets from Stabilizer.
        color_map: Optional color mapping for object types.
            Defaults to blue (255, 0, 0) for unknown types.

    Returns:
        Annotated image copy.
    """
    if color_map is None:
        color_map = {}

    result = image.copy()

    for target in stable_targets:
        color = color_map.get(target.object_type, (255, 0, 0))

        # Convert center + size to corner coordinates
        half_w = target.width / 2
        half_h = target.height / 2
        pt1 = (int(target.x - half_w), int(target.y - half_h))
        pt2 = (int(target.x + half_w), int(target.y + half_h))
        cv2.rectangle(result, pt1, pt2, color, 2)

        # Draw a richer label.
        # Note: in practice this function is often called with PickProcess.TrackedTarget,
        # not Stabilizer.StableTarget. We use getattr() to keep it compatible.
        target_id = getattr(target, "track_id", None)
        cluster_id = getattr(target, "cluster_id", None)
        state = getattr(target, "state", None)

        def _enum_value(v) -> str:
            return v.value if hasattr(v, "value") else str(v)

        parts: list[str] = []
        if cluster_id is not None:
            parts.append(f"cid={cluster_id}")
        if target_id is not None:
            parts.append(f"id={target_id}")

        # object_type is always present on both StableTarget and TrackedTarget
        if getattr(target, "object_type", None) is not None:
            parts.append(str(target.object_type))

        if state is not None:
            parts.append(f"state={_enum_value(state)}")

        parts.append(f"conf={target.confidence:.2f}")
        yaw_deg = getattr(target, "u", None)
        if yaw_deg is not None:
            parts.append(f"u={yaw_deg:.1f}")
        label = " ".join(parts) if parts else f"{target.confidence:.2f}"
        label_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            2
        )

        # Label background
        cv2.rectangle(
            result,
            (pt1[0], pt1[1] - label_size[1] - 14),
            (pt1[0] + label_size[0], pt1[1]),
            color,
            -1
        )

        # Label text
        cv2.putText(
            result,
            label,
            (pt1[0], pt1[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        # World coordinate: center marker + label (read directly from target)
        world_x = getattr(target, "world_x", None)
        world_y = getattr(target, "world_y", None)
        if world_x is not None and world_y is not None:
            cx_i, cy_i = int(round(target.x)), int(round(target.y))
            marker_color = (0, 255, 255)  # cyan/yellow

            # Solid dot at center
            cv2.circle(result, (cx_i, cy_i), 6, marker_color, -1)
            # Outer ring
            cv2.circle(result, (cx_i, cy_i), 12, marker_color, 2)
            # Crosshair lines extending beyond bbox
            arm_len = max(int(half_w), 20) + 8
            cv2.line(result, (cx_i - arm_len, cy_i), (cx_i + arm_len, cy_i), marker_color, 1)
            cv2.line(result, (cx_i, cy_i - arm_len), (cx_i, cy_i + arm_len), marker_color, 1)

            # Coordinate text (large, below bbox)
            coord_label = f"({world_x:.2f}, {world_y:.2f}) mm"
            coord_font_scale = 1.6
            coord_thickness = 3
            coord_size, _ = cv2.getTextSize(
                coord_label, cv2.FONT_HERSHEY_SIMPLEX,
                coord_font_scale, coord_thickness,
            )
            coord_x = int(target.x - coord_size[0] / 2)
            coord_y = pt2[1] + coord_size[1] + 14
            # Shadow for readability
            cv2.putText(
                result, coord_label, (coord_x + 2, coord_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, coord_font_scale,
                (0, 0, 0), coord_thickness + 3,
            )
            cv2.putText(
                result, coord_label, (coord_x, coord_y),
                cv2.FONT_HERSHEY_SIMPLEX, coord_font_scale,
                marker_color, coord_thickness,
            )

    return result
