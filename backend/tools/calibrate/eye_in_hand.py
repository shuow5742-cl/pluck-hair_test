#!/usr/bin/env python3
"""
Eye-in-Hand hand-eye calibration tool

Calibrates the transform between a camera mounted on the robot end-effector and the end-effector frame.

Usage:
    python -m tools.calibrate.eye_in_hand \
        --data data/calibration/eye_in_hand/ \
        --intrinsic config/calibration/camera_intrinsic.yaml \
        --pattern circle \
        --grid 11,11 \
        --spacing 1.0 \
        --output config/calibration/eye_in_hand.yaml

Data preparation:
    Prepare the following files under `data/calibration/eye_in_hand/`:
    
    1. Image files: `pose_001.jpg`, `pose_002.jpg`, ...
    2. Pose file: `poses.yaml`, format:
    
        poses:
          - name: pose_001
            # End-effector pose relative to the base
            position: [x, y, z]       # mm
            orientation: [rx, ry, rz]  # deg, Euler XYZ
          - name: pose_002
            position: [x, y, z]
            orientation: [rx, ry, rz]
          ...

Workflow:
    1. Fix the calibration target in the workspace (do not move it).
    2. Move the vision arm to multiple poses (10-20); the target must be visible at each pose.
    3. For each pose: record the end-effector pose + capture an image.
    4. Run the calibration script.
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml

from .intrinsic_models import CameraIntrinsicCalibrator, IntrinsicCalibrationResult


def euler_to_rotation_matrix(rx: float, ry: float, rz: float, degrees: bool = True) -> np.ndarray:
    """
    Convert Euler angles to a rotation matrix (XYZ order).

    Args:
        rx, ry, rz: rotation around X, Y, Z axes
        degrees: whether the inputs are in degrees (otherwise radians)
    """
    if degrees:
        rx, ry, rz = np.radians([rx, ry, rz])

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def rotation_matrix_to_euler(R: np.ndarray, degrees: bool = True) -> tuple[float, float, float]:
    """
    Convert a rotation matrix to Euler angles (XYZ order).

    Args:
        R: 3x3 rotation matrix
        degrees: return degrees if True (otherwise radians)
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0

    if degrees:
        return np.degrees(rx), np.degrees(ry), np.degrees(rz)
    return rx, ry, rz


def pose_to_matrix(position: list[float], orientation: list[float]) -> np.ndarray:
    """
    Convert a pose to a 4x4 transform matrix.

    Args:
        position: [x, y, z] in mm
        orientation: [rx, ry, rz] Euler angles in degrees
    """
    R = euler_to_rotation_matrix(*orientation, degrees=True)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def matrix_to_pose(T: np.ndarray) -> tuple[list[float], list[float]]:
    """
    Convert a 4x4 transform matrix to a pose.

    Returns:
        (position, orientation): position [x, y, z] mm, orientation [rx, ry, rz] deg
    """
    position = T[:3, 3].tolist()
    orientation = list(rotation_matrix_to_euler(T[:3, :3], degrees=True))
    return position, orientation


@dataclass
class EyeInHandCalibrationResult:
    """Hand-eye calibration result."""

    # Transform from camera frame → end-effector frame
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3x1 translation vector (mm)
    reprojection_error: float  # error metric
    num_poses: int  # number of poses used
    method: str  # calibration method
    calibration_date: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def transform_matrix(self) -> np.ndarray:
        """4x4 homogeneous transform matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.translation.flatten()
        return T

    def to_dict(self) -> dict:
        """Convert to a dict suitable for YAML serialization."""
        position, orientation = matrix_to_pose(self.transform_matrix)
        return {
            "calibration_date": self.calibration_date,
            "num_poses": self.num_poses,
            "method": self.method,
            "reprojection_error": self.reprojection_error,
            "T_cam_to_ee": {
                "position": {
                    "x": position[0],
                    "y": position[1],
                    "z": position[2],
                },
                "orientation": {
                    "rx": orientation[0],
                    "ry": orientation[1],
                    "rz": orientation[2],
                },
                "rotation_matrix": self.rotation_matrix.tolist(),
                "translation": self.translation.flatten().tolist(),
            },
        }

    def save(self, path: str | Path) -> None:
        """Save calibration result to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        print(f"Calibration result saved to: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "EyeInHandCalibrationResult":
        """Load a calibration result from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        t_cam_to_ee = data["T_cam_to_ee"]
        rotation_matrix = np.array(t_cam_to_ee["rotation_matrix"], dtype=np.float64)
        translation = np.array(t_cam_to_ee["translation"], dtype=np.float64)

        return cls(
            rotation_matrix=rotation_matrix,
            translation=translation,
            reprojection_error=data.get("reprojection_error", 0.0),
            num_poses=data.get("num_poses", 0),
            method=data.get("method", "unknown"),
            calibration_date=data.get("calibration_date", ""),
        )


@dataclass
class PoseData:
    """Single pose record."""

    name: str
    image_path: Path
    ee_position: list[float]  # [x, y, z] mm
    ee_orientation: list[float]  # [rx, ry, rz] degrees


class EyeInHandCalibrator:
    """Eye-in-Hand hand-eye calibrator."""

    METHODS = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    def __init__(
        self,
        intrinsic: IntrinsicCalibrationResult,
        pattern_type: Literal["chessboard", "circle"] = "circle",
        grid_size: tuple[int, int] = (11, 11),
        spacing_mm: float = 1.0,
        method: str = "tsai",
    ):
        """
        Args:
            intrinsic: camera intrinsic calibration result
            pattern_type: calibration target type
            grid_size: target grid size
            spacing_mm: grid spacing (mm)
            method: hand-eye calibration method
        """
        self.intrinsic = intrinsic
        self.pattern_calibrator = CameraIntrinsicCalibrator(
            pattern_type=pattern_type,
            grid_size=grid_size,
            spacing_mm=spacing_mm,
        )
        self.method = method

        if method not in self.METHODS:
            raise ValueError(f"Unsupported method: {method}. Options: {list(self.METHODS.keys())}")

    def load_poses(self, data_dir: str | Path) -> list[PoseData]:
        """
        Load pose data.

        Args:
            data_dir: directory containing images and `poses.yaml`

        Returns:
            list of pose records
        """
        data_dir = Path(data_dir)
        poses_file = data_dir / "poses.yaml"

        if not poses_file.exists():
            raise FileNotFoundError(f"Pose file not found: {poses_file}")

        with open(poses_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        poses = []
        for pose in data.get("poses", []):
            name = pose["name"]
            # Try multiple image extensions
            image_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = data_dir / f"{name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                print(f"Warning: image not found {name}.*")
                continue

            poses.append(
                PoseData(
                    name=name,
                    image_path=image_path,
                    ee_position=pose["position"],
                    ee_orientation=pose["orientation"],
                )
            )

        return poses

    def calibrate(
        self,
        poses: list[PoseData],
        show_detection: bool = False,
    ) -> EyeInHandCalibrationResult:
        """
        Run hand-eye calibration.

        Args:
            poses: list of pose records
            show_detection: show detection visualization

        Returns:
            calibration result
        """
        R_gripper2base_list = []  # end-effector rotation relative to base
        t_gripper2base_list = []  # end-effector translation relative to base
        R_target2cam_list = []  # target rotation relative to camera
        t_target2cam_list = []  # target translation relative to camera

        print(f"Processing {len(poses)} poses...")

        for i, pose in enumerate(poses):
            # Read image
            image = cv2.imread(str(pose.image_path))
            if image is None:
                print(f"  [{i + 1}] Failed to read image: {pose.image_path}")
                continue

            # Detect calibration target
            found, corners = self.pattern_calibrator.detect_pattern(image)
            if not found:
                print(f"  [{i + 1}] Detection failed: {pose.name}")
                continue

            # Estimate target pose relative to camera (solvePnP)
            success, rvec, tvec = cv2.solvePnP(
                self.pattern_calibrator.object_points,
                corners,
                self.intrinsic.camera_matrix,
                self.intrinsic.distortion_coeffs,
            )

            if not success:
                print(f"  [{i + 1}] solvePnP failed: {pose.name}")
                continue

            # Convert to rotation matrix
            R_target2cam, _ = cv2.Rodrigues(rvec)

            # End-effector pose relative to base
            R_gripper2base = euler_to_rotation_matrix(*pose.ee_orientation)
            t_gripper2base = np.array(pose.ee_position).reshape(3, 1)

            R_gripper2base_list.append(R_gripper2base)
            t_gripper2base_list.append(t_gripper2base)
            R_target2cam_list.append(R_target2cam)
            t_target2cam_list.append(tvec)

            print(f"  [{i + 1}] ✓ OK: {pose.name}")

            if show_detection:
                vis = image.copy()
                cv2.drawChessboardCorners(
                    vis,
                    self.pattern_calibrator.grid_size,
                    corners,
                    found,
                )
                cv2.imshow("Detection", vis)
                cv2.waitKey(500)

        if show_detection:
            cv2.destroyAllWindows()

        if len(R_gripper2base_list) < 3:
            raise ValueError(f"Not enough valid poses ({len(R_gripper2base_list)} < 3)")

        print(f"\nProcessed {len(R_gripper2base_list)} poses. Starting calibration...")

        # Perform hand-eye calibration
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base_list,
            t_gripper2base_list,
            R_target2cam_list,
            t_target2cam_list,
            method=self.METHODS[self.method],
        )

        # Compute an error metric (simplified; transform consistency)
        error = self._compute_calibration_error(
            R_gripper2base_list,
            t_gripper2base_list,
            R_target2cam_list,
            t_target2cam_list,
            R_cam2gripper,
            t_cam2gripper,
        )

        result = EyeInHandCalibrationResult(
            rotation_matrix=R_cam2gripper,
            translation=t_cam2gripper,
            reprojection_error=error,
            num_poses=len(R_gripper2base_list),
            method=self.method,
        )

        position, orientation = matrix_to_pose(result.transform_matrix)
        print("\nCalibration completed!")
        print(f"  Method: {self.method}")
        print(f"  Translation: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f} mm")
        print(f"  Rotation: rx={orientation[0]:.2f}, ry={orientation[1]:.2f}, rz={orientation[2]:.2f} °")
        print(f"  Error: {error:.4f}")

        return result

    def _compute_calibration_error(
        self,
        R_gripper2base_list,
        t_gripper2base_list,
        R_target2cam_list,
        t_target2cam_list,
        R_cam2gripper,
        t_cam2gripper,
    ) -> float:
        """Compute calibration error (transform consistency metric)."""
        errors = []

        for i in range(len(R_gripper2base_list)):
            for j in range(i + 1, len(R_gripper2base_list)):
                # Derive relative motion from gripper motion
                R_gi = R_gripper2base_list[i]
                R_gj = R_gripper2base_list[j]
                t_gi = t_gripper2base_list[i]
                t_gj = t_gripper2base_list[j]

                # Relative motion
                R_g = R_gj @ R_gi.T
                t_g = t_gj - R_g @ t_gi

                # Relative motion from target-to-camera estimates
                R_ci = R_target2cam_list[i]
                R_cj = R_target2cam_list[j]
                t_ci = t_target2cam_list[i]
                t_cj = t_target2cam_list[j]

                R_c = R_cj @ R_ci.T
                t_c = t_cj - R_c @ t_ci

                # Consistency check: R_g @ R_cam2gripper = R_cam2gripper @ R_c
                R_left = R_g @ R_cam2gripper
                R_right = R_cam2gripper @ R_c

                # Rotation error
                R_diff = R_left @ R_right.T
                angle_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
                errors.append(np.degrees(angle_error))

        return np.mean(errors) if errors else 0.0

    def calibrate_from_directory(
        self,
        data_dir: str | Path,
        show_detection: bool = False,
    ) -> EyeInHandCalibrationResult:
        """Load data from a directory and run calibration."""
        poses = self.load_poses(data_dir)
        return self.calibrate(poses, show_detection)


def main():
    parser = argparse.ArgumentParser(
        description="Eye-in-Hand hand-eye calibration tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.calibrate.eye_in_hand \\
      --data data/calibration/eye_in_hand/ \\
      --intrinsic config/calibration/camera_intrinsic.yaml \\
      --pattern circle \\
      --grid 11,11 \\
      --spacing 1.0

Data directory structure:
  data/calibration/eye_in_hand/
  ├── poses.yaml        # pose file
  ├── pose_001.jpg      # image file
  ├── pose_002.jpg
  └── ...

poses.yaml format:
  poses:
    - name: pose_001
      position: [100, 200, 300]    # mm
      orientation: [0, 0, 0]       # degrees (rx, ry, rz)
    - name: pose_002
      position: [150, 200, 300]
      orientation: [10, 0, 0]
        """,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Calibration data directory (images + poses.yaml)",
    )
    parser.add_argument(
        "--intrinsic",
        type=str,
        required=True,
        help="Path to camera intrinsic calibration YAML",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        choices=["chessboard", "circle"],
        default="circle",
        help="Calibration target type (default: circle)",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="11,11",
        help="Target grid size, format: rows,cols (default: 11,11)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="Grid spacing in mm (default: 1.0)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=list(EyeInHandCalibrator.METHODS.keys()),
        default="tsai",
        help="Calibration method (default: tsai)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config/calibration/eye_in_hand.yaml",
        help="Output file path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show detection visualization",
    )

    args = parser.parse_args()

    # Parse grid size
    try:
        grid_size = tuple(map(int, args.grid.split(",")))
        if len(grid_size) != 2:
            raise ValueError
    except ValueError:
        print(f"Error: invalid grid size format: {args.grid}")
        sys.exit(1)

    # Load camera intrinsics
    try:
        intrinsic = IntrinsicCalibrationResult.load(args.intrinsic)
        print(f"Loaded camera intrinsics: {args.intrinsic}")
    except Exception as e:
        print(f"Error: failed to load camera intrinsics: {e}")
        sys.exit(1)

    # Create calibrator
    calibrator = EyeInHandCalibrator(
        intrinsic=intrinsic,
        pattern_type=args.pattern,
        grid_size=grid_size,
        spacing_mm=args.spacing,
        method=args.method,
    )

    # Run calibration
    try:
        result = calibrator.calibrate_from_directory(
            args.data,
            show_detection=args.show,
        )
        result.save(args.output)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


