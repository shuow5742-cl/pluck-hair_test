"""
Extrinsic calibration tool: compute camera-to-flange offset (dx, dy).

The robot arm is moved so that a cross-hair marker (the world origin) is
visible in the camera frame.  Given the pixel position of the marker and
the arm pose at that moment, the tool calculates the 2-D translation
between camera optical center and flange center.

Usage (CLI)::

    python -m tools.calibrate.extrinsic \\
        --intrinsic config/calibration/camera_intrinsic.yaml \\
        --mm-per-pixel 0.0069 \\
        --pixel 740,412 \\
        --arm-pose 100.0,200.0 \\
        --output config/calibration/extrinsic.yaml

If ``--pixel`` or ``--arm-pose`` are omitted the tool runs in interactive
mode and prompts for the values.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from tools.calibrate.intrinsic_models import IntrinsicCalibrationResult


def _prompt_pair(name: str) -> tuple[float, float]:
    """Interactively ask for a comma-separated pair of floats."""
    while True:
        raw = input(f"Enter {name} as x,y: ").strip()
        try:
            parts = raw.split(",")
            if len(parts) != 2:
                raise ValueError
            return float(parts[0]), float(parts[1])
        except ValueError:
            print(f"  Invalid input. Please enter two numbers separated by a comma.")


def compute_extrinsic(
    px0: float,
    py0: float,
    arm_x0: float,
    arm_y0: float,
    cx: float,
    cy: float,
    mm_per_pixel: float,
) -> tuple[float, float]:
    """Return (dx, dy) camera-to-flange offset.

    The cross-hair marker is at the world origin (0, 0).

    Parameters
    ----------
    px0, py0:
        Pixel coordinates of the cross-hair in the image.
    arm_x0, arm_y0:
        Robot arm flange position when the image was taken (mm).
    cx, cy:
        Camera principal point (pixels).
    mm_per_pixel:
        Telecentric lens scale factor.
    """
    x_cam0 = (px0 - cx) * mm_per_pixel
    y_cam0 = (py0 - cy) * mm_per_pixel
    dx = -arm_x0 - x_cam0
    dy = -arm_y0 - y_cam0
    return dx, dy


def save_result(
    path: Path,
    *,
    dx: float,
    dy: float,
    mm_per_pixel: float,
    flip_y: bool,
    pixel: tuple[float, float],
    arm_pose: tuple[float, float],
    cx: float,
    cy: float,
) -> None:
    """Write calibration result to YAML."""
    result = {
        "calibration_date": datetime.now().isoformat(timespec="seconds"),
        "mm_per_pixel": mm_per_pixel,
        "T_cam_to_flange": {
            "dx": round(dx, 4),
            "dy": round(dy, 4),
        },
        "flip_y": flip_y,
        "calibration_data": {
            "pixel": [pixel[0], pixel[1]],
            "arm_pose": [arm_pose[0], arm_pose[1]],
            "principal_point": [cx, cy],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            result,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute camera-to-flange extrinsic offset (dx, dy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Fully specified:
  python -m tools.calibrate.extrinsic \\
      --intrinsic config/calibration/camera_intrinsic.yaml \\
      --mm-per-pixel 0.0069 \\
      --pixel 740,412 \\
      --arm-pose 100.0,200.0

  # Interactive (prompts for pixel & arm-pose):
  python -m tools.calibrate.extrinsic \\
      --intrinsic config/calibration/camera_intrinsic.yaml \\
      --mm-per-pixel 0.0069
""",
    )
    parser.add_argument(
        "--intrinsic",
        type=str,
        required=True,
        help="Path to camera intrinsic YAML",
    )
    parser.add_argument(
        "--mm-per-pixel",
        type=float,
        required=True,
        help="Telecentric lens scale (mm/pixel)",
    )
    parser.add_argument(
        "--pixel",
        type=str,
        default=None,
        help="Cross-hair pixel coords as px,py (e.g. 740,412)",
    )
    parser.add_argument(
        "--arm-pose",
        type=str,
        default=None,
        help="Arm flange position as x,y in mm (e.g. 100.0,200.0)",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Set flip_y=true in output (world Y opposite to image Y)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config/calibration/extrinsic.yaml",
        help="Output YAML path (default: config/calibration/extrinsic.yaml)",
    )

    args = parser.parse_args()

    # Load intrinsics for cx, cy
    intrinsic = IntrinsicCalibrationResult.load(args.intrinsic)
    cx, cy = intrinsic.cx, intrinsic.cy
    print(f"Intrinsics loaded: cx={cx:.2f}, cy={cy:.2f}")

    # Pixel coordinates
    if args.pixel is not None:
        parts = args.pixel.split(",")
        px0, py0 = float(parts[0]), float(parts[1])
    else:
        px0, py0 = _prompt_pair("cross-hair pixel position")

    # Arm pose
    if args.arm_pose is not None:
        parts = args.arm_pose.split(",")
        arm_x0, arm_y0 = float(parts[0]), float(parts[1])
    else:
        arm_x0, arm_y0 = _prompt_pair("arm flange position (mm)")

    mm_per_pixel = args.mm_per_pixel

    # Compute
    dx, dy = compute_extrinsic(px0, py0, arm_x0, arm_y0, cx, cy, mm_per_pixel)
    print(f"\nResult:")
    print(f"  dx = {dx:.4f} mm")
    print(f"  dy = {dy:.4f} mm")

    # Save
    output_path = Path(args.output)
    save_result(
        output_path,
        dx=dx,
        dy=dy,
        mm_per_pixel=mm_per_pixel,
        flip_y=args.flip_y,
        pixel=(px0, py0),
        arm_pose=(arm_x0, arm_y0),
        cx=cx,
        cy=cy,
    )
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
