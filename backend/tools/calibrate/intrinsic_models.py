"""
Shared utilities for loading backend camera intrinsics and detecting calibration targets.

This replaces the old ``intrinsic.py`` module that was removed in favor of ROS2-based
calibration. The helpers here provide just enough structure for the verification and
hand-eye calibration tools to load intrinsics YAML files and locate calibration targets
in images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class IntrinsicCalibrationResult:
    """Minimal representation of camera intrinsics loaded from backend YAML."""

    calibration_date: str
    num_images: int
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    reprojection_error: float

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    @classmethod
    def load(cls, path: str | Path) -> "IntrinsicCalibrationResult":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        def _require(mapping: dict, key: str) -> dict:
            value = mapping.get(key)
            if not isinstance(value, dict):
                raise ValueError(f"Missing or invalid '{key}' section in {path}")
            return value

        image_size = _require(data, "image_size")
        camera_matrix = _require(data, "camera_matrix")
        distortion = _require(data, "distortion")

        width = int(image_size.get("width", 0))
        height = int(image_size.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image_size in {path}: {image_size!r}")

        fx = float(camera_matrix.get("fx", 0))
        fy = float(camera_matrix.get("fy", 0))
        cx = float(camera_matrix.get("cx", 0))
        cy = float(camera_matrix.get("cy", 0))

        camera_matrix_np = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

        k1 = float(distortion.get("k1", 0.0))
        k2 = float(distortion.get("k2", 0.0))
        p1 = float(distortion.get("p1", 0.0))
        p2 = float(distortion.get("p2", 0.0))
        k3 = float(distortion.get("k3", 0.0))
        distortion_coeffs_np = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

        return cls(
            calibration_date=str(data.get("calibration_date", "")),
            num_images=int(data.get("num_images", 0)),
            image_size=(width, height),
            camera_matrix=camera_matrix_np,
            distortion_coeffs=distortion_coeffs_np,
            reprojection_error=float(data.get("reprojection_error_px", -1.0)),
        )


class CameraIntrinsicCalibrator:
    """Pattern detection helper for calibration targets."""

    def __init__(
        self,
        *,
        pattern_type: Literal["chessboard", "circle"] = "circle",
        grid_size: tuple[int, int] = (11, 11),
        spacing_mm: float = 1.0,
    ) -> None:
        if pattern_type not in {"chessboard", "circle"}:
            raise ValueError("pattern_type must be 'chessboard' or 'circle'")

        self.pattern_type = pattern_type
        self.grid_size = grid_size
        self.spacing_mm = spacing_mm
        self.object_points = self._build_object_points()

    def _build_object_points(self) -> np.ndarray:
        if self.pattern_type == "chessboard":
            objp = np.zeros((self.grid_size[0] * self.grid_size[1], 3), np.float32)
            objp[:, :2] = (
                np.mgrid[0 : self.grid_size[0], 0 : self.grid_size[1]]
                .T.reshape(-1, 2)
                * self.spacing_mm
            )
        else:
            objp = []
            for i in range(self.grid_size[1]):
                for j in range(self.grid_size[0]):
                    objp.append(
                        [
                            (2 * j + i % 2) * self.spacing_mm,
                            i * self.spacing_mm,
                            0.0,
                        ]
                    )
            objp = np.array(objp, dtype=np.float32)
        return objp

    def detect_pattern(self, image: np.ndarray) -> tuple[bool, np.ndarray | None]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.pattern_type == "chessboard":
            found, corners = cv2.findChessboardCorners(gray, self.grid_size)
            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        else:
            found, corners = cv2.findCirclesGrid(
                gray,
                self.grid_size,
                flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
            )

        return found, corners


__all__ = ["IntrinsicCalibrationResult", "CameraIntrinsicCalibrator"]
