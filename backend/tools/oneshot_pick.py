#!/usr/bin/env python3
"""End-to-end pick verification: capture one frame, run pipeline, print world coords.

Used to sanity-check the full calibration + segmentation + coordinate
transform chain before the PLC is hooked up. The operator jogs the arm to
a known pose, reads flange XY off the teach pendant, and runs this script
with --flange-x / --flange-y. The script:

  1. Loads ExtrinsicCalibration from the configured yaml files.
  2. Opens the Daheng camera (one frame), or reads --image instead.
  3. Runs the segmentation + abstain pipeline.
  4. Applies pixel→world transform to every pick using the provided
     flange XY (mimicking what PixelToWorldTask does at runtime).
  5. Prints pixel + world coordinates per detection.
  6. Saves annotated.png under data/calibration_snapshots/<frame_id>/ for
     visual cross-checking.

Usage::

    cd backend
    uv run python tools/oneshot_pick.py \\
        --flange-x -60.9999 --flange-y 0.4798

    # Or to re-run on a saved frame (e.g. from segment_image_demo):
    uv run python tools/oneshot_pick.py \\
        --image data/labeled/05.15_4_2/<file>.png \\
        --flange-x -60.9999 --flange-y 0.4798
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flange-x", type=float, required=True,
                        help="Flange X (mm) reported by teach pendant at capture time")
    parser.add_argument("--flange-y", type=float, required=True,
                        help="Flange Y (mm) reported by teach pendant at capture time")
    parser.add_argument("--image", type=Path, default=None,
                        help="Skip camera and read this PNG instead")
    parser.add_argument("--intrinsic-path", type=Path,
                        default=ROOT / "config/calibration/camera_intrinsic.yaml")
    parser.add_argument("--extrinsic-path", type=Path,
                        default=ROOT / "config/calibration/extrinsic.yaml")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "assets/best_foreigh_segment_yolov8m_seg.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/calibration_snapshots")
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--exposure-us", type=float, default=50000)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--safety-margin-px", type=float, default=78)
    return parser.parse_args()


def capture_from_camera(args: argparse.Namespace):
    from autoweaver.camera import CameraConfig, DahengCamera

    camera = DahengCamera(CameraConfig(
        device_index=args.device_index,
        exposure_auto=False,
        gain_auto=False,
        exposure_time=args.exposure_us,
        white_balance_mode="once",
    ))
    if not camera.open():
        raise SystemExit("Failed to open Daheng camera")

    # Drop the first few frames so white-balance / exposure settle.
    for _ in range(3):
        try:
            camera.capture()
        except RuntimeError:
            pass
        time.sleep(0.05)

    frame = camera.capture()
    camera.close()
    return frame


def main() -> None:
    args = parse_args()

    from autoweaver.pipeline import VisionPipeline
    import src.steps  # registers yolo_seg + abstain_near_metal
    from src.core.coordinate_transform import (
        CoordinateTransformer,
        ExtrinsicCalibration,
    )
    from src.types import SegDetection

    cal = ExtrinsicCalibration.load(args.extrinsic_path, args.intrinsic_path)
    transformer = CoordinateTransformer(cal)
    print(
        f"calibration: dx={cal.dx:.4f}  dy={cal.dy:.4f}  "
        f"mm_per_pixel={cal.mm_per_pixel:.6f}\n"
        f"  cx={cal.cx:.4f}  cy={cal.cy:.4f}  "
        f"axis_mapping=({cal.flange_x_from}, {cal.flange_y_from})"
    )
    print(
        f"flange (operator-supplied): "
        f"X={args.flange_x:.4f} mm   Y={args.flange_y:.4f} mm"
    )

    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"cannot read {args.image}")
        print(f"\nframe source: file {args.image}  shape={frame.shape}")
    else:
        print("\nframe source: live camera (one shot)")
        frame = capture_from_camera(args)
        print(f"  captured shape={frame.shape}")

    pipe = VisionPipeline.from_config({"pipeline": {"steps": [
        {"name": "segment", "type": "yolo_seg", "params": {
            "model": str(args.model),
            "conf": args.conf,
            "save_artifacts": True,
            "output_dir": str(args.output_dir),
        }},
        {"name": "abstain", "type": "abstain_near_metal", "params": {
            "safety_margin_px": args.safety_margin_px,
        }},
    ]}})

    result = pipe.run(frame)
    seg_dets = [d for d in result.detections if isinstance(d, SegDetection)]
    print(f"\n=== detections: {len(seg_dets)} (abstain dropped what was filtered) ===")

    if not seg_dets:
        print("  no surviving detections — nothing to transform")
        return

    for d in seg_dets:
        if d.pick_point_xy is None:
            print(f"  [{d.detection_id}] {d.shape_class}  pick=None (skipped)")
            continue
        px, py = d.pick_point_xy
        wp = transformer.pixel_to_world(
            px=px, py=py,
            arm_x=args.flange_x, arm_y=args.flange_y,
        )
        angle = d.pick_angle_deg
        angle_str = f"{angle:.2f}°" if angle is not None else "—"
        print(
            f"  [{d.detection_id}]  shape={d.shape_class:13s}  method={d.pick_method}\n"
            f"     pixel=({px:8.2f}, {py:8.2f})  angle={angle_str}\n"
            f"     world_xy_mm=({wp.x:9.4f}, {wp.y:9.4f})"
        )

    print(
        f"\nannotated.png saved under {args.output_dir}/<frame_id>/  "
        "(frame_id printed above as part of pipeline metadata path)"
    )


if __name__ == "__main__":
    main()
