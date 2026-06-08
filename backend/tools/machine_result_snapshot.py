"""Snapshot one annotated frame to data/machine_result/ per photo position.

Stand-alone helper that wraps the same drawing logic used by
``tools/segment_batch_demo.py:render_annotated`` so the operator can
audit "what did the camera see + where did the algorithm pick" after
the fact. Designed for one purpose: be easy to delete.

To remove the feature, drop the single ``try_save_machine_result`` call
in ``src/tasks/frame_loop.py`` and this file. No protocol changes, no
config wiring — the saver swallows its own exceptions so a broken
write never breaks the production pick cycle.

File layout (per nova5 photo position)::

    data/machine_result/
        photo_1-1__seg_1779350773_000001.png   ← annotated, original
        photo_1-1__seg_1779350773_000001.json  ← per-detection summary

Filename embeds the photo key (`press_index-photo_index`) and the
seg_frame_id, so the same key + new seg_frame_id == one new file per
revisit; nothing is overwritten.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def render_annotated(
    frame_bgr: np.ndarray,
    detections: Iterable[Any],
    flange_pose_mm: Optional[tuple[float, float]] = None,
    *,
    cell_box_xyxy: Optional[tuple[int, int, int, int]] = None,
    safety_margin_px: Optional[float] = None,
) -> np.ndarray:
    """Reproduce the segment_batch_demo overlay: mask + bbox + pick + label.

    Detections must have the SegDetection attribute surface
    (polygon_xy, bbox, pick_point_xy, pick_angle_deg, ...). Anything
    missing is silently skipped — this is a diagnostic, not a contract.

    Optional decorations:
        cell_box_xyxy     — (x1, y1, x2, y2) of the matched cell in frame
                            coordinates. Drawn as a cyan rectangle so the
                            operator can see what crop_single_square locked
                            onto.
        safety_margin_px  — width (in pixels) of the inset abstain band.
                            Drawn as a yellow dashed rectangle inside the
                            cell box so the operator can see the no-pick
                            zone without affecting any geometry.
    """
    out = frame_bgr.copy()
    overlay = out.copy()

    for det in detections:
        poly = getattr(det, "polygon_xy", None)
        if poly:
            arr = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [arr], (0, 128, 255))
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    # Cell box and safety band are drawn under the detection bboxes so the
    # detection rectangles stay legible on top.
    if cell_box_xyxy is not None:
        _draw_cell_and_safety(out, cell_box_xyxy, safety_margin_px)

    for det in detections:
        bbox = getattr(det, "bbox", None)
        if bbox is not None:
            x1, y1, x2, y2 = [
                int(round(v)) for v in (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
            ]
            box_color, tool_label = _tool_style(det)
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
            obj_type = getattr(det, "object_type", "?")
            conf = getattr(det, "confidence", 0.0) or 0.0
            area = getattr(det, "mask_area", 0)
            label = f"{obj_type} {conf:.2f} area={area}"
            if tool_label:
                label = f"{tool_label} {label}"
            cv2.putText(
                out, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA,
            )
        _draw_pick(out, det)

    if flange_pose_mm is not None:
        pose_str = (
            f"flange XY = ({flange_pose_mm[0]:.3f}, {flange_pose_mm[1]:.3f}) mm"
        )
        cv2.putText(
            out, pose_str, (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
        )
    return out


def _draw_cell_and_safety(
    out: np.ndarray,
    cell_box_xyxy: tuple[int, int, int, int],
    safety_margin_px: Optional[float],
) -> None:
    """Draw the matched cell rectangle (cyan) and the inset safety band (yellow dashed)."""
    x1, y1, x2, y2 = (int(round(v)) for v in cell_box_xyxy)
    # Cell border — solid cyan.
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 2)
    cv2.putText(
        out, "cell", (x1 + 6, y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA,
    )

    if safety_margin_px is None or safety_margin_px <= 0:
        return
    inset = int(round(safety_margin_px))
    sx1, sy1 = x1 + inset, y1 + inset
    sx2, sy2 = x2 - inset, y2 - inset
    if sx2 <= sx1 or sy2 <= sy1:
        return
    # Safety band — dashed yellow rectangle. The region INSIDE this rectangle
    # is the pick-allowed zone; anywhere between this and the cell border is
    # the abstain band that prevents the gripper from approaching the plate.
    _draw_dashed_rect(out, (sx1, sy1), (sx2, sy2), color=(0, 255, 255),
                      thickness=2, dash_len=18, gap_len=10)
    cv2.putText(
        out, f"safe (margin {inset}px)", (sx1 + 6, sy1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
    )


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    *,
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 12,
    gap_len: int = 8,
) -> None:
    """Draw a rectangle whose edges are dashed segments."""
    x1, y1 = pt1
    x2, y2 = pt2
    step = dash_len + gap_len
    # Top and bottom edges.
    for x in range(x1, x2, step):
        x_end = min(x + dash_len, x2)
        cv2.line(img, (x, y1), (x_end, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y2), (x_end, y2), color, thickness, cv2.LINE_AA)
    # Left and right edges.
    for y in range(y1, y2, step):
        y_end = min(y + dash_len, y2)
        cv2.line(img, (x1, y), (x1, y_end), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2, y_end), color, thickness, cv2.LINE_AA)


def _draw_pick(out: np.ndarray, det: Any) -> None:
    pick_xy = getattr(det, "pick_point_xy", None)
    if not pick_xy:
        return
    px, py = [int(round(v)) for v in pick_xy]
    color = (0, 0, 255)
    cv2.drawMarker(
        out, (px, py), color,
        markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2,
    )
    cv2.circle(out, (px, py), 5, color, thickness=2)

    angle = getattr(det, "pick_angle_deg", None)
    if angle is not None:
        theta = np.deg2rad(angle)
        dx = int(round(np.cos(theta) * 30))
        dy = int(round(np.sin(theta) * 30))
        cv2.line(
            out, (px - dx, py - dy), (px + dx, py + dy),
            (255, 0, 255), 2, cv2.LINE_AA,
        )

    text = getattr(det, "pick_method", None) or "pick"
    pick_score = getattr(det, "pick_score", None)
    if pick_score is not None:
        text += f" {pick_score:.2f}"
    dist_metal = getattr(det, "distance_to_metal_px", None)
    if dist_metal is not None:
        text += f" metal={dist_metal:.0f}px"

    world_xy = getattr(det, "world_xy", None)
    if world_xy is not None and len(world_xy) >= 2:
        text += f"  W=({world_xy[0]:.2f}, {world_xy[1]:.2f})mm"

    cv2.putText(
        out, text, (px + 8, max(0, py - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
    )


def _det_summary(det: Any) -> dict[str, Any]:
    """Lightweight dict for the side-car JSON; omits the heavy polygon."""
    def get(name: str, default: Any = None) -> Any:
        return getattr(det, name, default)
    bbox = get("bbox")
    bbox_list = (
        [bbox.x1, bbox.y1, bbox.x2, bbox.y2] if bbox is not None else None
    )
    return {
        "detection_id": get("detection_id"),
        "object_type": get("object_type"),
        "preferred_epson_tool": get("preferred_epson_tool"),
        "confidence": get("confidence"),
        "bbox_xyxy": bbox_list,
        "mask_area": get("mask_area"),
        "pick_point_xy_px": get("pick_point_xy"),
        "pick_angle_deg": get("pick_angle_deg"),
        "pick_method": get("pick_method"),
        "pick_score": get("pick_score"),
        "shape_class": get("shape_class"),
        "distance_to_metal_px": get("distance_to_metal_px"),
        "world_xy_mm": get("world_xy"),
    }


def _tool_style(det: Any) -> tuple[tuple[int, int, int], str]:
    tool_code = getattr(det, "preferred_epson_tool", None)
    if tool_code == 2:
        return (0, 0, 255), "suck"
    return (0, 255, 0), "tweezers"


def try_save_machine_result(
    frame_bgr: np.ndarray,
    detections: Iterable[Any],
    output_dir: Path | str,
    photo_key: Optional[str] = None,
    seg_frame_id: Optional[str] = None,
    flange_pose_mm: Optional[tuple[float, float]] = None,
    *,
    cell_box_xyxy: Optional[tuple[int, int, int, int]] = None,
    safety_margin_px: Optional[float] = None,
) -> Optional[Path]:
    """Write annotated PNG + JSON sidecar. Returns the PNG path, or None on failure.

    Exceptions are caught and logged at WARNING — the snapshot is a
    diagnostic, never a failure point for the pick cycle.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem_parts: list[str] = []
        if photo_key:
            stem_parts.append(f"photo_{photo_key}")
        if seg_frame_id:
            stem_parts.append(f"seg_{seg_frame_id}")
        if not stem_parts:
            stem_parts.append("snapshot")
        stem = "__".join(stem_parts)

        png_path = output_dir / f"{stem}.png"
        json_path = output_dir / f"{stem}.json"

        det_list = list(detections)
        annotated = render_annotated(
            frame_bgr, det_list, flange_pose_mm,
            cell_box_xyxy=cell_box_xyxy,
            safety_margin_px=safety_margin_px,
        )
        if not cv2.imwrite(str(png_path), annotated):
            logger.warning("machine_result PNG write returned false: %s", png_path)
            return None

        sidecar = {
            "photo_key": photo_key,
            "seg_frame_id": seg_frame_id,
            "flange_pose_mm": list(flange_pose_mm) if flange_pose_mm else None,
            "cell_box_xyxy": list(cell_box_xyxy) if cell_box_xyxy else None,
            "safety_margin_px": safety_margin_px,
            "detections": [_det_summary(d) for d in det_list],
        }
        json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
        return png_path
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        logger.warning("machine_result snapshot failed: %s", exc, exc_info=True)
        return None
