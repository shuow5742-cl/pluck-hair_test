"""
Coordinate transform verification tool for the fixed camera + fixed arm setup.

Workflow:
1. Open a live camera preview (or load a still image).
2. Click a known physical reference point on the calibration board.
3. The tool converts that pixel into world coordinates using the current
   intrinsic + extrinsic calibration.
4. Compare the measured world coordinate with the expected physical coordinate.

This is intentionally independent from the detection pipeline. It lets the
operator verify the current pixel -> mm parameters directly with a calibration
board or origin marker.

Usage::

    python -m tools.calibrate.verify_transform \\
        --config config/settings.dev.yaml \\
        --intrinsic config/calibration/camera_intrinsic.yaml \\
        --extrinsic config/calibration/extrinsic.yaml \\
        --arm-pose 0,0 \\
        --expected 0,0

Controls:
    Left Click  - select calibration point
    SPACE       - save current measurement sample
    e           - update expected world coordinate
    a           - update arm pose
    c           - clear current selection
    q / ESC     - quit and write YAML report
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.core.coordinate_transform import CoordinateTransformer, ExtrinsicCalibration

from .extrinsic_align import create_camera, draw_crosshair


WINDOW_NAME = "Verify Transform - Click known board point"


def parse_pair(raw: str, *, name: str) -> tuple[float, float]:
    """Parse a comma-separated numeric pair."""
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be two numbers separated by a comma")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"{name} must be two numbers separated by a comma") from exc


def prompt_pair(name: str, default: tuple[float, float]) -> tuple[float, float]:
    """Prompt for a comma-separated numeric pair with a default."""
    while True:
        raw = input(f"Enter {name} as x,y [{default[0]:.4f},{default[1]:.4f}]: ").strip()
        if not raw:
            return default
        try:
            return parse_pair(raw, name=name)
        except ValueError as exc:
            print(f"  {exc}")


@dataclass(frozen=True, slots=True)
class VerificationSample:
    """One coordinate verification measurement."""

    sample_id: int
    pixel_x: float
    pixel_y: float
    arm_x: float
    arm_y: float
    expected_x: float
    expected_y: float
    measured_x: float
    measured_y: float
    error_x: float
    error_y: float

    @property
    def error_norm(self) -> float:
        return math.hypot(self.error_x, self.error_y)


@dataclass
class PreviewState:
    """Mutable UI state for the OpenCV preview."""

    scale: float
    selected_pixel: tuple[float, float] | None = None
    frame_shape: tuple[int, int] | None = None


def measure_point(
    *,
    transformer: CoordinateTransformer,
    pixel: tuple[float, float],
    arm_pose: tuple[float, float],
    expected_world: tuple[float, float],
    sample_id: int,
) -> VerificationSample:
    """Measure one pixel and compare it with the expected world coordinate."""
    px, py = pixel
    arm_x, arm_y = arm_pose
    expected_x, expected_y = expected_world
    world = transformer.pixel_to_world(px, py, arm_x, arm_y)
    error_x = world.x - expected_x
    error_y = world.y - expected_y
    return VerificationSample(
        sample_id=sample_id,
        pixel_x=px,
        pixel_y=py,
        arm_x=arm_x,
        arm_y=arm_y,
        expected_x=expected_x,
        expected_y=expected_y,
        measured_x=world.x,
        measured_y=world.y,
        error_x=error_x,
        error_y=error_y,
    )


def summarize_samples(samples: list[VerificationSample]) -> dict[str, float | int | None]:
    """Aggregate session error metrics."""
    if not samples:
        return {
            "num_samples": 0,
            "mean_error_x_mm": None,
            "mean_error_y_mm": None,
            "mean_abs_error_x_mm": None,
            "mean_abs_error_y_mm": None,
            "mean_error_norm_mm": None,
            "max_error_norm_mm": None,
        }

    num_samples = len(samples)
    mean_error_x = sum(sample.error_x for sample in samples) / num_samples
    mean_error_y = sum(sample.error_y for sample in samples) / num_samples
    mean_abs_error_x = sum(abs(sample.error_x) for sample in samples) / num_samples
    mean_abs_error_y = sum(abs(sample.error_y) for sample in samples) / num_samples
    mean_error_norm = sum(sample.error_norm for sample in samples) / num_samples
    max_error_norm = max(sample.error_norm for sample in samples)

    return {
        "num_samples": num_samples,
        "mean_error_x_mm": round(mean_error_x, 4),
        "mean_error_y_mm": round(mean_error_y, 4),
        "mean_abs_error_x_mm": round(mean_abs_error_x, 4),
        "mean_abs_error_y_mm": round(mean_abs_error_y, 4),
        "mean_error_norm_mm": round(mean_error_norm, 4),
        "max_error_norm_mm": round(max_error_norm, 4),
    }


def build_report(
    *,
    samples: list[VerificationSample],
    calibration: ExtrinsicCalibration,
    extrinsic_path: str,
    intrinsic_path: str,
    config_path: str | None,
    image_path: str | None,
    preview_scale: float,
) -> dict[str, Any]:
    """Build a YAML-serializable verification report."""
    return {
        "verification_date": datetime.now().isoformat(timespec="seconds"),
        "mode": "image" if image_path else "camera",
        "config_path": config_path,
        "image_path": image_path,
        "preview_scale": preview_scale,
        "calibration": {
            "extrinsic_path": extrinsic_path,
            "intrinsic_path": intrinsic_path,
            "mm_per_pixel": calibration.mm_per_pixel,
            "T_cam_to_flange": {
                "dx": calibration.dx,
                "dy": calibration.dy,
            },
            "principal_point": {
                "cx": calibration.cx,
                "cy": calibration.cy,
            },
            "flip_y": calibration.flip_y,
        },
        "summary": summarize_samples(samples),
        "samples": [
            {
                **asdict(sample),
                "error_norm": round(sample.error_norm, 4),
            }
            for sample in samples
        ],
    }


def save_report(path: Path, report: dict[str, Any]) -> None:
    """Write YAML report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(
            report,
            handle,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def draw_selection_marker(image: np.ndarray, pixel: tuple[float, float]) -> None:
    """Draw the currently selected pixel."""
    px, py = (int(round(pixel[0])), int(round(pixel[1])))
    color = (0, 255, 0)
    cv2.circle(image, (px, py), 8, color, 2)
    cv2.line(image, (px - 20, py), (px + 20, py), color, 1)
    cv2.line(image, (px, py - 20), (px, py + 20), color, 1)


def put_text_block(image: np.ndarray, lines: list[str], x: int = 16, y: int = 28) -> None:
    """Draw a compact multi-line text block with shadow."""
    for index, line in enumerate(lines):
        line_y = y + index * 24
        cv2.putText(
            image,
            line,
            (x, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )


def render_frame(
    *,
    frame: np.ndarray,
    calibration: ExtrinsicCalibration,
    transformer: CoordinateTransformer,
    preview_state: PreviewState,
    arm_pose: tuple[float, float],
    expected_world: tuple[float, float],
    samples: list[VerificationSample],
) -> np.ndarray:
    """Render a preview frame with selection, live coordinates, and instructions."""
    display = frame.copy()
    preview_state.frame_shape = frame.shape[:2]

    draw_crosshair(display, int(round(calibration.cx)), int(round(calibration.cy)))

    lines = [
        f"Principal: ({calibration.cx:.1f}, {calibration.cy:.1f}) px",
        f"Arm pose: ({arm_pose[0]:.3f}, {arm_pose[1]:.3f}) mm",
        f"Expected: ({expected_world[0]:.3f}, {expected_world[1]:.3f}) mm",
        f"Samples: {len(samples)}",
    ]

    if preview_state.selected_pixel is not None:
        draw_selection_marker(display, preview_state.selected_pixel)
        live_sample = measure_point(
            transformer=transformer,
            pixel=preview_state.selected_pixel,
            arm_pose=arm_pose,
            expected_world=expected_world,
            sample_id=len(samples) + 1,
        )
        lines.extend(
            [
                f"Pixel: ({live_sample.pixel_x:.1f}, {live_sample.pixel_y:.1f}) px",
                f"Measured: ({live_sample.measured_x:.3f}, {live_sample.measured_y:.3f}) mm",
                f"Error: ({live_sample.error_x:.3f}, {live_sample.error_y:.3f}) mm",
                f"|error|: {live_sample.error_norm:.3f} mm",
            ]
        )
    else:
        lines.extend(
            [
                "Pixel: <click a known point>",
                "Measured: -",
                "Error: -",
                "|error|: -",
            ]
        )

    put_text_block(display, lines)

    help_lines = [
        "Left Click: select point",
        "SPACE: save sample",
        "e: expected coord",
        "a: arm pose",
        "c: clear",
        "q/ESC: quit",
    ]
    height = display.shape[0]
    for index, line in enumerate(reversed(help_lines)):
        put_text_block(display, [line], x=16, y=height - 20 - index * 24)

    if preview_state.scale != 1.0:
        display = cv2.resize(
            display,
            None,
            fx=preview_state.scale,
            fy=preview_state.scale,
            interpolation=cv2.INTER_AREA,
        )
    return display


def make_mouse_callback(preview_state: PreviewState):
    """Build the OpenCV mouse callback."""

    def _callback(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if preview_state.frame_shape is None:
            return

        height, width = preview_state.frame_shape
        px = x / preview_state.scale
        py = y / preview_state.scale
        px = min(max(px, 0.0), width - 1.0)
        py = min(max(py, 0.0), height - 1.0)
        preview_state.selected_pixel = (px, py)

    return _callback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the current pixel -> world coordinate transform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Fixed arm + fixed camera, verify board origin at (0, 0)
  python -m tools.calibrate.verify_transform \\
      --config config/settings.dev.yaml \\
      --intrinsic config/calibration/camera_intrinsic.yaml \\
      --extrinsic config/calibration/extrinsic.yaml \\
      --arm-pose 0,0 \\
      --expected 0,0

  # Validate a saved image instead of a live camera feed
  python -m tools.calibrate.verify_transform \\
      --image data/calibration/check_origin.jpg \\
      --intrinsic config/calibration/camera_intrinsic.yaml \\
      --extrinsic config/calibration/extrinsic.yaml
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.dev.yaml",
        help="Camera config YAML used for live preview (default: config/settings.dev.yaml)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional still image path. If set, camera preview is skipped.",
    )
    parser.add_argument(
        "--intrinsic",
        type=str,
        default="config/calibration/camera_intrinsic.yaml",
        help="Path to camera intrinsic YAML",
    )
    parser.add_argument(
        "--extrinsic",
        type=str,
        default="config/calibration/extrinsic.yaml",
        help="Path to extrinsic calibration YAML",
    )
    parser.add_argument(
        "--arm-pose",
        type=str,
        default="0,0",
        help="Current arm flange position as x,y in mm (default: 0,0)",
    )
    parser.add_argument(
        "--expected",
        type=str,
        default="0,0",
        help="Expected world coordinate of the clicked point as x,y in mm (default: 0,0)",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.5,
        help="Preview window scale factor (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional YAML report path. Defaults to data/calibration/verify_transform/<timestamp>.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    arm_pose = parse_pair(args.arm_pose, name="arm pose")
    expected_world = parse_pair(args.expected, name="expected coordinate")
    calibration = ExtrinsicCalibration.load(args.extrinsic, args.intrinsic)
    transformer = CoordinateTransformer(calibration)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else Path(
        f"data/calibration/verify_transform/verify_{timestamp}.yaml"
    )

    print("Loaded coordinate transform:")
    print(f"  mm_per_pixel = {calibration.mm_per_pixel:.6f}")
    print(f"  dx, dy       = ({calibration.dx:.4f}, {calibration.dy:.4f}) mm")
    print(f"  cx, cy       = ({calibration.cx:.2f}, {calibration.cy:.2f}) px")
    print(f"  flip_y       = {calibration.flip_y}")
    print(f"  arm pose     = ({arm_pose[0]:.4f}, {arm_pose[1]:.4f}) mm")
    print(f"  expected     = ({expected_world[0]:.4f}, {expected_world[1]:.4f}) mm")
    print()

    base_image: np.ndarray | None = None
    camera = None
    if args.image:
        base_image = cv2.imread(args.image)
        if base_image is None:
            raise RuntimeError(f"Failed to read image: {args.image}")
        print(f"Loaded image: {args.image}")
    else:
        camera = create_camera(args.config)
        if not camera.open():
            raise RuntimeError("Failed to open camera")
        print("Camera opened. Live verification started.")

    samples: list[VerificationSample] = []
    preview_state = PreviewState(scale=args.preview_scale)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_callback(preview_state))

    try:
        while True:
            if base_image is not None:
                frame = base_image
            else:
                frame = camera.capture() if camera is not None else None
                if frame is None:
                    continue

            preview = render_frame(
                frame=frame,
                calibration=calibration,
                transformer=transformer,
                preview_state=preview_state,
                arm_pose=arm_pose,
                expected_world=expected_world,
                samples=samples,
            )
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):
                break

            if key == ord("c"):
                preview_state.selected_pixel = None
                continue

            if key == ord("e"):
                expected_world = prompt_pair("expected world coordinate (mm)", expected_world)
                print(
                    f"Updated expected coordinate to ({expected_world[0]:.4f}, "
                    f"{expected_world[1]:.4f}) mm"
                )
                continue

            if key == ord("a"):
                arm_pose = prompt_pair("arm flange position (mm)", arm_pose)
                print(f"Updated arm pose to ({arm_pose[0]:.4f}, {arm_pose[1]:.4f}) mm")
                continue

            if key == ord(" "):
                if preview_state.selected_pixel is None:
                    print("No pixel selected yet. Left-click a known board point first.")
                    continue

                sample = measure_point(
                    transformer=transformer,
                    pixel=preview_state.selected_pixel,
                    arm_pose=arm_pose,
                    expected_world=expected_world,
                    sample_id=len(samples) + 1,
                )
                samples.append(sample)

                print(f"Sample #{sample.sample_id}")
                print(f"  pixel      = ({sample.pixel_x:.2f}, {sample.pixel_y:.2f}) px")
                print(f"  measured   = ({sample.measured_x:.4f}, {sample.measured_y:.4f}) mm")
                print(f"  expected   = ({sample.expected_x:.4f}, {sample.expected_y:.4f}) mm")
                print(f"  error      = ({sample.error_x:.4f}, {sample.error_y:.4f}) mm")
                print(f"  |error|    = {sample.error_norm:.4f} mm")

                report = build_report(
                    samples=samples,
                    calibration=calibration,
                    extrinsic_path=args.extrinsic,
                    intrinsic_path=args.intrinsic,
                    config_path=None if args.image else args.config,
                    image_path=args.image,
                    preview_scale=args.preview_scale,
                )
                save_report(output_path, report)
                print(f"  report     = {output_path}")
                continue
    finally:
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()

    report = build_report(
        samples=samples,
        calibration=calibration,
        extrinsic_path=args.extrinsic,
        intrinsic_path=args.intrinsic,
        config_path=None if args.image else args.config,
        image_path=args.image,
        preview_scale=args.preview_scale,
    )
    save_report(output_path, report)
    summary = report["summary"]
    print("\nVerification session finished.")
    print(f"  samples     = {summary['num_samples']}")
    print(f"  mean |err|  = {summary['mean_error_norm_mm']}")
    print(f"  max |err|   = {summary['max_error_norm_mm']}")
    print(f"  report      = {output_path}")


if __name__ == "__main__":
    main()
