#!/usr/bin/env python3
"""
Calibration verification tool

Verifies camera intrinsics and Eye-in-Hand calibration results.

Usage:
    # Verify camera intrinsics (undistortion effect)
    python -m tools.calibrate.verify intrinsic \
        --calibration config/calibration/camera_intrinsic.yaml \
        --image data/test_image.jpg
    
    # Verify hand-eye calibration (coordinate transform accuracy)
    python -m tools.calibrate.verify eye_in_hand \
        --intrinsic config/calibration/camera_intrinsic.yaml \
        --eye-in-hand config/calibration/eye_in_hand.yaml \
        --data data/calibration/verify/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from .eye_in_hand import EyeInHandCalibrationResult, euler_to_rotation_matrix
from .intrinsic_models import (
    CameraIntrinsicCalibrator,
    IntrinsicCalibrationResult,
)


def verify_intrinsic(
    calibration_path: str,
    image_path: str | None = None,
    pattern_type: str = "circle",
    grid_size: tuple[int, int] = (11, 11),
    spacing_mm: float = 1.0,
) -> None:
    """Verify camera intrinsic calibration."""
    print("=" * 60)
    print("Camera Intrinsics Verification")
    print("=" * 60)

    # Load calibration result
    result = IntrinsicCalibrationResult.load(calibration_path)
    print("\nCalibration info:")
    print(f"  Date: {result.calibration_date}")
    print(f"  Images: {result.num_images}")
    print(f"  Image size: {result.image_size[0]} x {result.image_size[1]}")
    print(f"  Reprojection error: {result.reprojection_error:.4f} px")

    print("\nIntrinsics:")
    print(f"  fx = {result.fx:.2f}")
    print(f"  fy = {result.fy:.2f}")
    print(f"  cx = {result.cx:.2f}")
    print(f"  cy = {result.cy:.2f}")

    print("\nDistortion:")
    print(f"  k1 = {result.distortion_coeffs[0]:.6f}")
    print(f"  k2 = {result.distortion_coeffs[1]:.6f}")
    print(f"  p1 = {result.distortion_coeffs[2]:.6f}")
    print(f"  p2 = {result.distortion_coeffs[3]:.6f}")
    if len(result.distortion_coeffs) > 4:
        print(f"  k3 = {result.distortion_coeffs[4]:.6f}")

    # Quick quality assessment
    print("\nQuality assessment:")
    if result.reprojection_error < 0.3:
        print("  ✓ Excellent")
    elif result.reprojection_error < 0.5:
        print("  ✓ Good")
    elif result.reprojection_error < 1.0:
        print("  ⚠ Acceptable")
    else:
        print("  ✗ Poor")

    # If a test image is provided, show undistortion result
    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            print(f"\nWarning: failed to read image {image_path}")
            return

        # Undistort
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            result.camera_matrix,
            result.distortion_coeffs,
            (w, h),
            1,
            (w, h),
        )

        undistorted = cv2.undistort(
            image,
            result.camera_matrix,
            result.distortion_coeffs,
            None,
            new_camera_matrix,
        )

        # Detect target (if present)
        calibrator = CameraIntrinsicCalibrator(
            pattern_type=pattern_type,
            grid_size=grid_size,
            spacing_mm=spacing_mm,
        )

        found_orig, corners_orig = calibrator.detect_pattern(image)
        found_undist, corners_undist = calibrator.detect_pattern(undistorted)

        # Visualize results
        vis_orig = image.copy()
        vis_undist = undistorted.copy()

        if found_orig:
            cv2.drawChessboardCorners(vis_orig, grid_size, corners_orig, found_orig)
        if found_undist:
            cv2.drawChessboardCorners(vis_undist, grid_size, corners_undist, found_undist)

        # Side-by-side visualization
        combined = np.hstack([vis_orig, vis_undist])
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Undistorted", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize if needed
        scale = min(1920 / combined.shape[1], 1080 / combined.shape[0], 1.0)
        if scale < 1.0:
            combined = cv2.resize(combined, None, fx=scale, fy=scale)

        cv2.imshow("Distortion Correction", combined)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def verify_eye_in_hand(
    intrinsic_path: str,
    eye_in_hand_path: str,
    data_dir: str | None = None,
) -> None:
    """Verify Eye-in-Hand hand-eye calibration."""
    print("=" * 60)
    print("Eye-in-Hand Verification")
    print("=" * 60)

    # Load calibration results
    intrinsic = IntrinsicCalibrationResult.load(intrinsic_path)
    eye_in_hand = EyeInHandCalibrationResult.load(eye_in_hand_path)

    print("\nCalibration info:")
    print(f"  Date: {eye_in_hand.calibration_date}")
    print(f"  Method: {eye_in_hand.method}")
    print(f"  Poses: {eye_in_hand.num_poses}")
    print(f"  Error: {eye_in_hand.reprojection_error:.4f}")

    print("\nTransform T_cam_to_ee:")
    print(f"  Translation: x={eye_in_hand.translation[0, 0]:.2f}, "
          f"y={eye_in_hand.translation[1, 0]:.2f}, "
          f"z={eye_in_hand.translation[2, 0]:.2f} mm")

    # Compute Euler angles
    from .eye_in_hand import rotation_matrix_to_euler
    rx, ry, rz = rotation_matrix_to_euler(eye_in_hand.rotation_matrix)
    print(f"  Rotation: rx={rx:.2f}, ry={ry:.2f}, rz={rz:.2f} °")

    print("\nRotation matrix:")
    for row in eye_in_hand.rotation_matrix:
        print(f"    [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")

    # If verification data is provided, run additional checks
    if data_dir:
        verify_dir = Path(data_dir)
        poses_file = verify_dir / "poses.yaml"

        if not poses_file.exists():
            print(f"\nWarning: verification pose file not found: {poses_file}")
            return

        print(f"\nAccuracy check (using {data_dir}):")
        # TODO: implement verification logic
        print("  (Verification logic not implemented yet)")


def main():
    parser = argparse.ArgumentParser(
        description="Calibration verification tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Verification type")

    # Intrinsics verification
    intrinsic_parser = subparsers.add_parser("intrinsic", help="Verify camera intrinsics")
    intrinsic_parser.add_argument(
        "--calibration",
        type=str,
        required=True,
        help="Camera intrinsics YAML",
    )
    intrinsic_parser.add_argument(
        "--image",
        type=str,
        help="Test image (optional; shows undistortion result)",
    )
    intrinsic_parser.add_argument(
        "--pattern",
        type=str,
        choices=["chessboard", "circle"],
        default="circle",
        help="Target type",
    )
    intrinsic_parser.add_argument(
        "--grid",
        type=str,
        default="11,11",
        help="Target grid size",
    )
    intrinsic_parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="Grid spacing (mm)",
    )

    # Hand-eye verification
    eye_parser = subparsers.add_parser("eye_in_hand", help="Verify Eye-in-Hand calibration")
    eye_parser.add_argument(
        "--intrinsic",
        type=str,
        required=True,
        help="Camera intrinsics YAML",
    )
    eye_parser.add_argument(
        "--eye-in-hand",
        type=str,
        required=True,
        help="Eye-in-Hand calibration YAML",
    )
    eye_parser.add_argument(
        "--data",
        type=str,
        help="Verification data directory (optional)",
    )

    args = parser.parse_args()

    if args.command == "intrinsic":
        grid_size = tuple(map(int, args.grid.split(",")))
        verify_intrinsic(
            args.calibration,
            args.image,
            args.pattern,
            grid_size,
            args.spacing,
        )
    elif args.command == "eye_in_hand":
        verify_eye_in_hand(
            args.intrinsic,
            getattr(args, "eye_in_hand"),
            args.data,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


