#!/usr/bin/env python3
"""Camera crosshair viewer for dx/dy extrinsic calibration.

Opens the Daheng camera at full resolution and draws a red crosshair at the
principal point ``(cx, cy)`` loaded from ``config/calibration/camera_intrinsic.yaml``.
The test operator jogs the robot arm until the crosshair points exactly at
the workbench origin marker, then reads the flange ``(X, Y)`` off the teach
pendant. The follow-up command writes ``(dx, dy) = -flange_xy`` into
``config/calibration/extrinsic.yaml``.

Background:
- The arm's flange has 3-DOF translation only (no rotation).
- Telecentric lens — ``mm_per_pixel`` is a constant, independent of Z.
- When the crosshair (which sits at the camera's optical axis) is aligned
  with world origin, the pixel→world equation collapses to:
      world_xy = flange_xy + (dx, dy) + 0
      → (dx, dy) = world_xy - flange_xy = -flange_xy   (since world_xy = 0)

Usage::

    cd backend
    uv run python tools/calibrate_extrinsic_viewer.py

Then in a second terminal, after the operator gives you flange_xy::

    uv run python tools/calibrate_extrinsic_viewer.py \\
        --write-dxdy --flange-x <X> --flange-y <Y>

Controls (viewer mode):
    q / ESC  quit
    s        save current frame to data/calibration_snapshots/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_principal_point(intrinsic_path: Path) -> tuple[float, float]:
    with intrinsic_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cam = data.get("camera_matrix") or {}
    cx = float(cam.get("cx"))
    cy = float(cam.get("cy"))
    return cx, cy


def write_extrinsic_dxdy(extrinsic_path: Path, flange_x: float, flange_y: float) -> None:
    """Update extrinsic.yaml in place — only T_cam_to_flange.{dx,dy} and timestamp."""
    if not extrinsic_path.exists():
        raise FileNotFoundError(extrinsic_path)

    with extrinsic_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    t = data.get("T_cam_to_flange") or {}
    t["dx"] = float(-flange_x)
    t["dy"] = float(-flange_y)
    data["T_cam_to_flange"] = t
    data["calibration_date"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    data["calibration_note"] = (
        f"crosshair-to-origin alignment; flange_xy at alignment = "
        f"({flange_x:.4f}, {flange_y:.4f}) mm"
    )

    with extrinsic_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def draw_crosshair(frame, cx: int, cy: int) -> None:
    """Draw a red crosshair centered at (cx, cy)."""
    h, w = frame.shape[:2]
    red = (0, 0, 255)
    # Long crosshair lines across full frame.
    cv2.line(frame, (0, cy), (w, cy), red, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, 0), (cx, h), red, 1, cv2.LINE_AA)
    # Center accent — thicker short cross + circle so the operator's eye locks on.
    arm = 30
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), red, 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), red, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 6, red, 2, cv2.LINE_AA)
    label = f"({cx},{cy})"
    cv2.putText(frame, label, (cx + 12, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2, cv2.LINE_AA)


def run_viewer(args: argparse.Namespace) -> None:
    cx_f, cy_f = load_principal_point(args.intrinsic_path)
    cx, cy = int(round(cx_f)), int(round(cy_f))
    print(f"principal point: ({cx_f}, {cy_f}) → drawing crosshair at ({cx}, {cy})")

    from autoweaver.camera import CameraConfig, DahengCamera
    cam_cfg = CameraConfig(
        device_index=args.device_index,
        exposure_auto=False,
        gain_auto=False,
        exposure_time=args.exposure_us,
        white_balance_mode="once",
    )
    camera = DahengCamera(cam_cfg)
    if not camera.open():
        raise SystemExit("Failed to open Daheng camera")

    snap_dir = ROOT / "data" / "calibration_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    win = "Calibration crosshair — align red center to workbench origin"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 960)

    try:
        while True:
            try:
                frame = camera.capture()
            except RuntimeError:
                time.sleep(0.01)
                continue
            if frame is None:
                time.sleep(0.01)
                continue

            draw_crosshair(frame, cx, cy)
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                out = snap_dir / f"calib_snap_{ts}.png"
                cv2.imwrite(str(out), frame)
                print(f"snapshot saved: {out}")
    finally:
        camera.close()
        cv2.destroyAllWindows()


def run_write_dxdy(args: argparse.Namespace) -> None:
    if args.flange_x is None or args.flange_y is None:
        raise SystemExit("--write-dxdy needs --flange-x and --flange-y (mm)")
    write_extrinsic_dxdy(args.extrinsic_path, args.flange_x, args.flange_y)
    print(
        f"wrote dx={-args.flange_x:.4f} dy={-args.flange_y:.4f} mm "
        f"to {args.extrinsic_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrinsic-path", type=Path,
                        default=ROOT / "config/calibration/camera_intrinsic.yaml")
    parser.add_argument("--extrinsic-path", type=Path,
                        default=ROOT / "config/calibration/extrinsic.yaml")
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--exposure-us", type=float, default=50000)
    parser.add_argument("--write-dxdy", action="store_true",
                        help="After alignment, write -flange_xy to extrinsic.yaml as (dx,dy)")
    parser.add_argument("--flange-x", type=float, default=None,
                        help="Flange X (mm) read from teach pendant at alignment")
    parser.add_argument("--flange-y", type=float, default=None,
                        help="Flange Y (mm) read from teach pendant at alignment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_dxdy:
        run_write_dxdy(args)
    else:
        run_viewer(args)


if __name__ == "__main__":
    main()
