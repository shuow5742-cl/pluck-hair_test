#!/usr/bin/env python3
"""Operator-driven Epson coord probe — capture-on-keypress, freeze, render.

Use case: nova5 is parked, the operator wants to see "if I pick this hair,
what Epson XY does the chain produce?" — without burning GPU on every
frame. The viewer streams the raw camera; pressing `c` captures one frame,
runs yolo_seg → abstain → pixel→world → nova5_to_epson grid lookup,
freezes the preview, and renders the resulting Epson XY in large text on
the image.

Coord chain (hybrid):
    pick_pixel + flange_xy
        → pixel_to_world         (rough nova5 world XY estimate)
        → nearest grid anchor    (locally calibrated correction)
        → epson_xy = anchor.epson_xy + (world - anchor.nova5_xy)

Rationale: pure pixel→world has a single global (dx, dy) offset and
drifts across the workspace because mechanical/optical errors are not
uniform. Pure flange→grid is locally accurate but ignores the pick
pixel — moving a hair on the table doesn't move the reported coord.
The hybrid uses pixel→world to feed the grid a per-detection nova5
position, so each hair gets its own anchor lookup and the global
dx/dy drift gets pulled back by the local anchor calibration.

Pipeline:
    Daheng (SN FCM26010005) → yolo_seg → abstain_near_metal
        → CoordinateTransformer (extrinsic.yaml + intrinsic.yaml)
        → ArmGridMapper (nova5_to_epson_grid.yaml)

Default flange XY: from plc_points.yaml point 1-1 (nova5 at the first photo
position) — override with --flange-x / --flange-y when nova5 is parked
elsewhere.

Keys (focus the preview window first):
    c       capture: run pipeline once, freeze preview with annotation
    space   freeze→live+overlay: keep mask/pick/Epson XY pinned to the
            same pixel coords on top of the live feed (telecentric +
            nova5 parked → pixel-aligned, so a tweezer entering the
            frame can be visually checked against the original target)
    l       clear overlay, return to clean live preview
    s       save the current view: raw PNG + annotated PNG + JSON
    q/Esc   quit

Usage:
    cd backend
    uv run python tools/pick_offset_probe.py
    # or override pose:
    uv run python tools/pick_offset_probe.py --flange-x 1.5603 --flange-y -51.8229
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.steps  # noqa: F401 — register hub-side pipeline step types

from autoweaver.camera import CameraConfig as BaseCameraConfig  # noqa: E402
from autoweaver.camera import DahengCamera  # noqa: E402
from autoweaver.pipeline import VisionPipeline  # noqa: E402

from src.core.arm_grid_mapper import ArmGridMapper, ArmGridMatch  # noqa: E402
from src.core.coordinate_transform import (  # noqa: E402
    CoordinateTransformer,
    ExtrinsicCalibration,
)
from src.types import SegDetection  # noqa: E402
from tools.machine_result_snapshot import render_annotated  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device-sn", default="FCM26010005")
    p.add_argument("--exposure-us", type=float, default=50000.0)
    p.add_argument("--flange-x", type=float, default=1.5603,
                   help="Nova5 flange X (mm). Default: plc_points point 1-1.")
    p.add_argument("--flange-y", type=float, default=-51.8229,
                   help="Nova5 flange Y (mm). Default: plc_points point 1-1.")
    p.add_argument("--extrinsic", default="config/calibration/extrinsic.yaml")
    p.add_argument("--intrinsic", default="config/calibration/camera_intrinsic.yaml")
    p.add_argument("--grid", default="config/nova5_to_epson_grid.yaml")
    p.add_argument("--model", default="assets/best_foreigh_segment_yolov8m_seg.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto")
    p.add_argument("--safety-margin-px", type=float, default=78.0)
    p.add_argument("--output-dir", default="data/pick_offset_probe")
    p.add_argument("--preview-scale", type=float, default=0.5)
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


def build_pipeline(args: argparse.Namespace) -> VisionPipeline:
    return VisionPipeline.from_config({
        "pipeline": {
            "steps": [
                {
                    "name": "segment",
                    "type": "yolo_seg",
                    "params": {
                        "model": str(resolve(args.model)),
                        "conf": args.conf,
                        "iou": args.iou,
                        "imgsz": args.imgsz,
                        "device": args.device,
                        "output_dir": str(resolve(args.output_dir)),
                        "save_artifacts": False,
                    },
                },
                {
                    "name": "abstain",
                    "type": "abstain_near_metal",
                    "params": {"safety_margin_px": args.safety_margin_px},
                },
            ]
        }
    })


def open_camera(args: argparse.Namespace) -> DahengCamera:
    cam = DahengCamera(BaseCameraConfig(
        device_sn=args.device_sn,
        exposure_auto=False,
        gain_auto=False,
        exposure_time=args.exposure_us,
        white_balance_mode="once",
    ))
    if not cam.open():
        raise RuntimeError(f"Failed to open Daheng SN={args.device_sn}")
    return cam


def render_capture(
    frame_bgr,
    seg_dets: list[SegDetection],
    epson_matches: list[Optional[ArmGridMatch]],
    flange_xy: tuple[float, float],
):
    """Build the freeze-frame image: pipeline overlay + large Epson XY text.

    Reusable for both FROZEN mode (canvas = captured frame) and
    LIVE+OVERLAY mode (canvas = current live frame, decorations stay
    pixel-aligned because the camera is telecentric and nova5 is parked).
    """
    annotated = render_annotated(frame_bgr, seg_dets, flange_pose_mm=flange_xy)

    for det, match in zip(seg_dets, epson_matches):
        if det.pick_point_xy is None:
            continue
        px, py = int(round(det.pick_point_xy[0])), int(round(det.pick_point_xy[1]))
        if match is None:
            text = "epson: N/A"
        else:
            text = f"E=({match.epson_x:+8.3f}, {match.epson_y:+8.3f}) mm"
        # Big, high-contrast label next to each pick point.
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.4
        thick = 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        anchor = (px + 18, max(th + 8, py - 12))
        cv2.rectangle(
            annotated,
            (anchor[0] - 6, anchor[1] - th - 8),
            (anchor[0] + tw + 6, anchor[1] + 8),
            (0, 0, 0), -1,
        )
        cv2.putText(annotated, text, anchor, font, scale, (0, 255, 255), thick, cv2.LINE_AA)

    return annotated


def overlay_state_banner(image, *, mode: str, flange_xy, n_dets: int, saved_count: int):
    """Top-left HUD: live/frozen mode, flange pose, key cheatsheet."""
    lines = [
        f"mode: {mode}",
        f"flange XY: ({flange_xy[0]:.4f}, {flange_xy[1]:.4f}) mm",
        f"detections: {n_dets}    saved: {saved_count}",
        "[c] capture+freeze   [space/l] live   [s] save   [q] quit",
    ]
    y = 40
    for line in lines:
        cv2.putText(image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        y += 36


def downscale_for_preview(image, scale: float):
    if scale == 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def run_pipeline_once(
    frame_bgr,
    pipeline: VisionPipeline,
    transformer: CoordinateTransformer,
    grid_mapper: ArmGridMapper,
    flange_x: float,
    flange_y: float,
):
    """Per-detection: pick px → "what flange XY would put the camera optical
    center on this hair" → nearest grid anchor → epson XY.

    Why subtract (dx, dy) before feeding the grid:

    `pixel_to_world(px, py, arm_x, arm_y)` returns
        world = flange_xy + (dx, dy) + pixel_offset_in_mm
    which is the workspace position of the **camera optical center** projected
    onto this hair pixel — NOT the flange position.

    `nova5_to_epson_grid.yaml` anchors store **flange** XY (the operator
    drove the flange to each anchor and read the teach pendant). Feeding
    the world position straight into the grid mismatches by exactly
    (dx, dy) ≈ (-26.68, -21.38) mm — the fingerprint of the bug we hit.

    Subtracting (dx, dy) recovers "what flange XY would let the optical
    center line up with this hair", which IS the language the grid speaks.
    Grid then locally corrects for non-uniform mechanical drift.
    """
    result = pipeline.run(frame_bgr)
    seg_dets = [d for d in result.detections if isinstance(d, SegDetection)]
    cal = transformer._cal  # frozen dataclass — safe to read
    epson_matches: list[Optional[ArmGridMatch]] = []
    for d in seg_dets:
        if d.pick_point_xy is None:
            d.world_xy = None
            epson_matches.append(None)
            continue
        wp = transformer.pixel_to_world(
            float(d.pick_point_xy[0]), float(d.pick_point_xy[1]),
            arm_x=flange_x, arm_y=flange_y,
        )
        d.world_xy = [wp.x, wp.y]
        # Convert "optical-center workspace position" → "flange position
        # that places the optical center here" by removing the camera
        # extrinsic offset. This is what the grid table is keyed on.
        flange_target_x = wp.x - cal.dx
        flange_target_y = wp.y - cal.dy
        try:
            match = grid_mapper.map_nova5_to_epson(flange_target_x, flange_target_y)
        except Exception:  # noqa: BLE001 — diagnostic tool, never crash on bad grid
            match = None
        epson_matches.append(match)
    return seg_dets, epson_matches, result.metadata


def save_capture(
    output_dir: Path,
    frame_bgr,
    annotated,
    seg_dets: list[SegDetection],
    epson_matches: list[Optional[ArmGridMatch]],
    flange_xy: tuple[float, float],
    pipeline_metadata: dict,
):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"probe_{stamp}_raw.png"
    annotated_path = output_dir / f"probe_{stamp}_annotated.png"
    json_path = output_dir / f"probe_{stamp}.json"

    cv2.imwrite(str(raw_path), frame_bgr)
    cv2.imwrite(str(annotated_path), annotated)

    sidecar = {
        "captured_at": stamp,
        "flange_pose_mm": [flange_xy[0], flange_xy[1]],
        "seg_frame_id": pipeline_metadata.get("seg_frame_id"),
        "detections": [
            {
                "detection_id": d.detection_id,
                "object_type": d.object_type,
                "confidence": d.confidence,
                "shape_class": d.shape_class,
                "pick_point_xy_px": d.pick_point_xy,
                "pick_angle_deg": d.pick_angle_deg,
                "pick_method": d.pick_method,
                "world_xy_mm": d.world_xy,
                "epson_xy_mm": (
                    [m.epson_x, m.epson_y] if m is not None else None
                ),
                "grid_anchor": (
                    {
                        "row": m.anchor.row,
                        "col": m.anchor.col,
                        "nova5_xy": [m.anchor.nova5_x, m.anchor.nova5_y],
                        "epson_xy": [m.anchor.epson_x, m.anchor.epson_y],
                        "distance_mm": m.distance_mm,
                    }
                    if m is not None else None
                ),
            }
            for d, m in zip(seg_dets, epson_matches)
        ],
    }
    json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
    return raw_path, annotated_path, json_path


def main() -> None:
    args = parse_args()

    extrinsic_path = resolve(args.extrinsic)
    intrinsic_path = resolve(args.intrinsic)
    grid_path = resolve(args.grid)

    calibration = ExtrinsicCalibration.load(extrinsic_path, intrinsic_path)
    transformer = CoordinateTransformer(calibration)
    grid_mapper = ArmGridMapper.load(grid_path)

    flange_x = args.flange_x
    flange_y = args.flange_y

    print(
        f"[probe] calibration: mm_per_pixel={calibration.mm_per_pixel}, "
        f"dx={calibration.dx}, dy={calibration.dy}, "
        f"axis=({calibration.flange_x_from},{calibration.flange_y_from})"
    )
    print(f"[probe] flange XY (mm) = ({flange_x:.4f}, {flange_y:.4f})")
    print(f"[probe] grid: {grid_path}")
    print("[probe] live preview running. Press 'c' to capture+freeze.")

    pipeline = build_pipeline(args)
    cam = open_camera(args)
    output_dir = resolve(args.output_dir)

    window_name = "pick offset probe (c=freeze, space=live+overlay, l=clear, s=save, q=quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Three states (frozen_view, overlay_payload):
    #   (None, None)              → LIVE (clean live preview, no inference)
    #   (img,  None)               → FROZEN (showing captured frame + annotations)
    #   (None, payload)            → LIVE+OVERLAY (live frame, freeze annotations re-rendered each tick)
    # `c` always returns to FROZEN; `space` from FROZEN moves to LIVE+OVERLAY;
    # `l` returns to clean LIVE from any state.
    frozen_view = None
    frozen_frame = None
    overlay_payload: Optional[dict] = None
    frozen_dets: list[SegDetection] = []
    frozen_matches: list[Optional[ArmGridMatch]] = []
    frozen_metadata: dict = {}
    saved_count = 0

    try:
        while True:
            if frozen_view is not None:
                # FROZEN — captured frame, annotations baked in.
                display = frozen_view.copy()
                mode = "FROZEN"
                n_dets = len(frozen_dets)
            elif overlay_payload is not None:
                # LIVE+OVERLAY — pull a fresh live frame, overlay frozen annotations.
                live_frame = cam.capture()
                display = render_capture(
                    live_frame,
                    overlay_payload["dets"],
                    overlay_payload["matches"],
                    (flange_x, flange_y),
                )
                mode = "LIVE+OVERLAY"
                n_dets = len(overlay_payload["dets"])
            else:
                # LIVE — clean preview, no inference, no overlay.
                display = cam.capture().copy()
                mode = "LIVE"
                n_dets = 0

            overlay_state_banner(
                display, mode=mode,
                flange_xy=(flange_x, flange_y),
                n_dets=n_dets, saved_count=saved_count,
            )
            cv2.imshow(window_name, downscale_for_preview(display, args.preview_scale))

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                print(f"[probe] key pressed: code={key} char={chr(key) if 32 <= key < 127 else '?'}")
            if key in (ord("q"), 27):
                break

            if key == ord("c"):
                capture_frame = cam.capture()
                seg_dets, epson_matches, meta = run_pipeline_once(
                    capture_frame, pipeline, transformer, grid_mapper,
                    flange_x, flange_y,
                )
                frozen_frame = capture_frame
                frozen_dets = seg_dets
                frozen_matches = epson_matches
                frozen_metadata = dict(meta)
                frozen_view = render_capture(
                    capture_frame, seg_dets, epson_matches, (flange_x, flange_y),
                )
                overlay_payload = None  # leaving overlay mode if we were in it
                print(f"[probe] captured: {len(seg_dets)} detection(s)")
                for d, m in zip(seg_dets, epson_matches):
                    if m is None:
                        print(f"  [{d.detection_id}] pick={d.pick_point_xy} epson=N/A")
                    else:
                        print(
                            f"  [{d.detection_id}] pick={d.pick_point_xy} "
                            f"world=({d.world_xy[0]:.3f},{d.world_xy[1]:.3f}) "
                            f"epson=({m.epson_x:.3f},{m.epson_y:.3f}) "
                            f"anchor=r{m.anchor.row}c{m.anchor.col} "
                            f"d={m.distance_mm:.2f}mm"
                        )
            elif key == ord(" "):
                # Promote the frozen frame's annotations into a live overlay so
                # the operator can move tweezers in and see them against the
                # current scene (telecentric → pixel-aligned).
                if frozen_view is None and overlay_payload is None:
                    print("[probe] nothing to overlay — capture first with 'c'")
                elif frozen_view is not None:
                    overlay_payload = {
                        "dets": frozen_dets,
                        "matches": frozen_matches,
                    }
                    frozen_view = None
                # If already in LIVE+OVERLAY, space is a no-op (still showing it).
            elif key == ord("l"):
                # Clear everything → clean LIVE.
                frozen_view = None
                frozen_frame = None
                overlay_payload = None
                frozen_dets = []
                frozen_matches = []
                frozen_metadata = {}
            elif key == ord("s"):
                # Save whatever's currently on screen, plus the JSON sidecar.
                if frozen_frame is None and overlay_payload is None:
                    print("[probe] nothing to save — capture first with 'c'")
                else:
                    save_frame = frozen_frame if frozen_frame is not None else cam.capture()
                    save_view = display.copy()
                    raw_p, ann_p, json_p = save_capture(
                        output_dir, save_frame, save_view,
                        frozen_dets, frozen_matches,
                        (flange_x, flange_y), frozen_metadata,
                    )
                    saved_count += 1
                    print(f"[probe] saved: raw={raw_p.name} annotated={ann_p.name} json={json_p.name}")
    finally:
        cv2.destroyAllWindows()
        cam.close()


if __name__ == "__main__":
    main()
