#!/usr/bin/env python3
"""Live capture + segment + world-coord render, no PLC.

A debugging viewer for the calibration loop. Runs the same pipeline as
settings.live.yaml (yolo_seg → abstain_near_metal), keeps the camera
streaming, draws each frame's annotation in an OpenCV window, and only
writes to disk when the operator presses ``s``.

Pipeline matches settings.live.yaml exactly:
    Daheng (SN FCM26010005) → yolo_seg → abstain_near_metal
        → CoordinateTransformer (extrinsic.yaml + intrinsic.yaml)

Default flange XY: ``(-dx, -dy)`` from extrinsic.yaml. By construction
the calibration writes ``(dx, dy) = -flange_xy_at_alignment``, so the
default puts the camera back at "world-frame origin" — the same pose
the calibration was taken at. Override with ``--flange-x/--flange-y``
when nova5 is parked elsewhere.

Keys (focus the preview window first):
    s  save the current frame: raw PNG + annotated PNG + JSON sidecar
    q  quit (also: Esc)

Usage:
    cd backend
    uv run python tools/oneshot_capture_render.py
    # or override pose:
    uv run python tools/oneshot_capture_render.py --flange-x 50 --flange-y 0
"""

from __future__ import annotations

import argparse
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

from src.core.coordinate_transform import (  # noqa: E402
    CoordinateTransformer,
    ExtrinsicCalibration,
)
from src.types import SegDetection  # noqa: E402
from tools.machine_result_snapshot import (  # noqa: E402
    render_annotated,
    try_save_machine_result,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--device-sn", default="FCM26010005",
        help="Daheng camera serial number (matches settings.live.yaml).",
    )
    p.add_argument(
        "--exposure-us", type=float, default=50000.0,
        help="Manual exposure time in microseconds.",
    )
    p.add_argument(
        "--flange-x", type=float, default=None,
        help="Nova5 flange X (mm). Default: -dx from extrinsic.yaml "
             "(i.e. the alignment pose used during calibration).",
    )
    p.add_argument(
        "--flange-y", type=float, default=None,
        help="Nova5 flange Y (mm). Default: -dy from extrinsic.yaml.",
    )
    p.add_argument(
        "--extrinsic", default="config/calibration/extrinsic.yaml",
    )
    p.add_argument(
        "--intrinsic", default="config/calibration/camera_intrinsic.yaml",
    )
    p.add_argument(
        "--model", default="assets/best_foreigh_segment_yolov8m_seg.pt",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto")
    p.add_argument("--safety-margin-px", type=float, default=78.0)
    p.add_argument(
        "--output-dir", default="data/machine_result",
        help="Where saved PNG and JSON go (only on operator save).",
    )
    p.add_argument(
        "--preview-scale", type=float, default=0.5,
        help="Downscale the preview window so the 2048x1536 frame fits "
             "on a typical screen. Saved files use the full-resolution image.",
    )
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
    cam_cfg = BaseCameraConfig(
        device_sn=args.device_sn,
        exposure_auto=False,
        gain_auto=False,
        exposure_time=args.exposure_us,
        white_balance_mode="once",
    )
    cam = DahengCamera(cam_cfg)
    if not cam.open():
        raise RuntimeError(f"Failed to open Daheng SN={args.device_sn}")
    return cam


def compute_world_xy(
    detections: list[SegDetection],
    transformer: CoordinateTransformer,
    flange_x: float,
    flange_y: float,
) -> None:
    """Mutate each detection's ``world_xy`` in place. Mirrors PixelToWorldTask."""
    for d in detections:
        if d.pick_point_xy is None:
            d.world_xy = None
            continue
        wp = transformer.pixel_to_world(
            float(d.pick_point_xy[0]), float(d.pick_point_xy[1]),
            arm_x=flange_x, arm_y=flange_y,
        )
        d.world_xy = [wp.x, wp.y]


def overlay_hud(
    annotated,
    *,
    flange_xy: tuple[float, float],
    n_detections: int,
    saved_count: int,
    last_saved_at: Optional[float],
):
    """Draw on-screen banner: pose + detection count + save state."""
    h, w = annotated.shape[:2]
    lines = [
        f"flange XY: ({flange_xy[0]:.3f}, {flange_xy[1]:.3f}) mm",
        f"detections: {n_detections}    saved: {saved_count}",
        "[s] save current frame   [q] quit",
    ]
    if last_saved_at is not None:
        ago = time.time() - last_saved_at
        if ago < 2.0:
            lines.append(f"...saved {ago:.1f}s ago")
    y = 30
    for line in lines:
        cv2.putText(
            annotated, line, (15, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            annotated, line, (15, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
        )
        y += 32


def main() -> None:
    args = parse_args()

    extrinsic_path = resolve(args.extrinsic)
    intrinsic_path = resolve(args.intrinsic)
    calibration = ExtrinsicCalibration.load(extrinsic_path, intrinsic_path)
    transformer = CoordinateTransformer(calibration)

    # Default flange XY = the alignment pose the extrinsic was taken at.
    # extrinsic.yaml stores (dx, dy) = -flange_xy_at_alignment by
    # convention (see calibrate_extrinsic_viewer.py / commit 9b33eaa),
    # so negating recovers the pose that puts the camera back on the
    # crosshair origin.
    flange_x = args.flange_x if args.flange_x is not None else -calibration.dx
    flange_y = args.flange_y if args.flange_y is not None else -calibration.dy

    print(
        f"[live] calibration: mm_per_pixel={calibration.mm_per_pixel}, "
        f"dx={calibration.dx}, dy={calibration.dy}, "
        f"axis=({calibration.flange_x_from},{calibration.flange_y_from})"
    )
    print(f"[live] flange XY (mm) = ({flange_x:.4f}, {flange_y:.4f})")
    print("[live] press 's' in the preview window to save, 'q' to quit")

    pipeline = build_pipeline(args)

    cam = open_camera(args)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_name = "oneshot live preview (s=save, q=quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    saved_count = 0
    last_saved_at: Optional[float] = None
    frame_idx = 0
    try:
        while True:
            frame = cam.capture()
            frame_idx += 1

            pipeline_result = pipeline.run(frame)
            seg_dets = [
                d for d in pipeline_result.detections
                if isinstance(d, SegDetection)
            ]
            compute_world_xy(seg_dets, transformer, flange_x, flange_y)

            annotated = render_annotated(
                frame, seg_dets, flange_pose_mm=(flange_x, flange_y),
            )
            overlay_hud(
                annotated,
                flange_xy=(flange_x, flange_y),
                n_detections=len(seg_dets),
                saved_count=saved_count,
                last_saved_at=last_saved_at,
            )

            if args.preview_scale != 1.0:
                preview = cv2.resize(
                    annotated, None,
                    fx=args.preview_scale, fy=args.preview_scale,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                preview = annotated
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                raw_path = output_dir / f"oneshot_{stamp}_raw.png"
                cv2.imwrite(str(raw_path), frame)
                annotated_path = try_save_machine_result(
                    frame_bgr=frame,
                    detections=seg_dets,
                    output_dir=output_dir,
                    photo_key=f"oneshot_{stamp}",
                    seg_frame_id=str(
                        pipeline_result.metadata.get("seg_frame_id") or stamp
                    ),
                    flange_pose_mm=(flange_x, flange_y),
                )
                saved_count += 1
                last_saved_at = time.time()
                print(
                    f"[live] saved frame {frame_idx}: dets={len(seg_dets)} "
                    f"raw={raw_path.name} annotated={annotated_path.name if annotated_path else 'FAILED'}"
                )
    finally:
        cv2.destroyAllWindows()
        cam.close()


if __name__ == "__main__":
    main()
