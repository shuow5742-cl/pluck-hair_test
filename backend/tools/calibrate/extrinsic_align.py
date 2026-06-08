"""
Extrinsic calibration alignment assist tool.

Opens a live camera feed with a red crosshair drawn at the camera principal
point (cx, cy).  The operator jogs the robot arm until the crosshair sits
on the physical cross-hair mark, then presses SPACE to capture.  At that
moment the pixel of the mark equals the principal point, so:

    dx = -arm_x
    dy = -arm_y

Usage::

    python -m tools.calibrate.extrinsic_align \
        --config config/settings.yaml \
        --intrinsic config/calibration/camera_intrinsic.yaml \
        --mm-per-pixel 0.0069

Keys:
    SPACE  – capture (prompts for arm pose in terminal)
    q/ESC  – quit
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from tools.calibrate.intrinsic_models import IntrinsicCalibrationResult


WINDOW_NAME = "Extrinsic Calibration - Align crosshair to mark"


def draw_crosshair(
    image: np.ndarray, cx: int, cy: int, size: int = 30, thickness: int = 2
) -> np.ndarray:
    """Draw a red crosshair + circle at the principal point."""
    color = (0, 0, 255)  # BGR red
    # Horizontal line
    cv2.line(image, (cx - size, cy), (cx + size, cy), color, thickness)
    # Vertical line
    cv2.line(image, (cx, cy - size), (cx, cy + size), color, thickness)
    # Center circle
    cv2.circle(image, (cx, cy), 6, color, -1)
    # Outer ring
    cv2.circle(image, (cx, cy), size, color, 1)
    return image


def draw_instructions(image: np.ndarray) -> np.ndarray:
    """Draw instruction text overlay."""
    h, w = image.shape[:2]
    texts = [
        "Jog arm until crosshair aligns with mark",
        "SPACE: capture  |  q/ESC: quit",
    ]
    for i, text in enumerate(texts):
        y = h - 20 - i * 30
        # Shadow
        cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        # Text
        cv2.putText(
            image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
    return image


def prompt_arm_pose() -> tuple[float, float] | None:
    """Ask operator to enter arm position in terminal."""
    print("\n--- Captured! ---")
    print("Enter current arm flange position (mm).")
    raw = input("  arm_x, arm_y: ").strip()
    if not raw:
        print("  Cancelled.")
        return None
    try:
        parts = raw.split(",")
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        print("  Invalid input, expected two numbers like: 100.0,200.0")
        return None


def save_result(
    path: Path,
    *,
    dx: float,
    dy: float,
    mm_per_pixel: float,
    flip_y: bool,
    arm_pose: tuple[float, float],
    cx: float,
    cy: float,
) -> None:
    """Write extrinsic calibration YAML."""
    result = {
        "calibration_date": datetime.now().isoformat(timespec="seconds"),
        "mm_per_pixel": mm_per_pixel,
        "T_cam_to_flange": {
            "dx": round(dx, 4),
            "dy": round(dy, 4),
        },
        "flip_y": flip_y,
        "calibration_data": {
            "pixel": [cx, cy],
            "arm_pose": [arm_pose[0], arm_pose[1]],
            "principal_point": [cx, cy],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(result, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def create_camera(config_path: str):
    """Create camera from settings.yaml (reuses main.py pattern)."""
    from src.config import AppConfig

    config = AppConfig.from_yaml(config_path)
    camera_config = config.camera

    if camera_config.type == "daheng":
        from autoweaver.camera import CameraConfig as BaseCameraConfig
        from autoweaver.camera import DahengCamera

        base_config = BaseCameraConfig(
            device_index=camera_config.device_index,
            exposure_auto=camera_config.exposure_auto,
            gain_auto=camera_config.gain_auto,
            exposure_time=camera_config.exposure_time,
            gain=camera_config.gain,
        )
        return DahengCamera(base_config)

    elif camera_config.type == "mock":
        from autoweaver.camera import CameraConfig as BaseCameraConfig
        from autoweaver.camera import MockCamera

        base_config = BaseCameraConfig()
        return MockCamera(
            base_config,
            mode=camera_config.mode,
            image_dir=camera_config.image_dir,
            width=camera_config.width,
            height=camera_config.height,
        )
    else:
        raise ValueError(f"Unknown camera type: {camera_config.type}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live alignment assist for extrinsic calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Workflow:
  1. Run this tool – a live preview opens with a red crosshair at (cx, cy).
  2. Jog the robot arm until the crosshair sits exactly on the cross-hair mark.
  3. Press SPACE, then type the arm position in the terminal.
  4. The tool computes dx, dy and saves to YAML.
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to settings.yaml for camera config",
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
        "--flip-y",
        action="store_true",
        help="Set flip_y=true in output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config/calibration/extrinsic.yaml",
        help="Output YAML path",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.5,
        help="Preview window scale factor (default: 0.5)",
    )

    args = parser.parse_args()

    # Load intrinsics
    intrinsic = IntrinsicCalibrationResult.load(args.intrinsic)
    cx_px, cy_px = int(round(intrinsic.cx)), int(round(intrinsic.cy))
    print(f"Principal point: ({intrinsic.cx:.1f}, {intrinsic.cy:.1f})")

    # Open camera
    camera = create_camera(args.config)
    camera.open()
    print("Camera opened. Showing live preview...")
    print("  SPACE = capture | q/ESC = quit\n")

    scale = args.preview_scale

    try:
        while True:
            frame = camera.capture()
            if frame is None:
                continue

            # Draw crosshair on full-res frame (copy to avoid modifying capture buffer)
            display = frame.copy()
            draw_crosshair(display, cx_px, cy_px)

            # Scale for display
            if scale != 1.0:
                h, w = display.shape[:2]
                display = cv2.resize(
                    display, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )
            draw_instructions(display)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # q or ESC
                print("Quit without saving.")
                break

            if key == ord(" "):  # SPACE
                pose = prompt_arm_pose()
                if pose is None:
                    print("Resuming preview...\n")
                    continue

                arm_x, arm_y = pose
                dx = -arm_x
                dy = -arm_y
                print(f"\n  dx = {dx:.4f} mm")
                print(f"  dy = {dy:.4f} mm")

                output_path = Path(args.output)
                save_result(
                    output_path,
                    dx=dx,
                    dy=dy,
                    mm_per_pixel=args.mm_per_pixel,
                    flip_y=args.flip_y,
                    arm_pose=(arm_x, arm_y),
                    cx=intrinsic.cx,
                    cy=intrinsic.cy,
                )
                print(f"  Saved to {output_path}")
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
