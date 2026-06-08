#!/usr/bin/env python3
"""
Convert ROS camera_calibration YAML into backend `camera_intrinsic.yaml` format.

ROS (camera_calibration) commonly produces `ost.yaml`/`camera.yaml` like:
  image_width, image_height
  camera_matrix.data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  distortion_coefficients.data: [k1, k2, p1, p2, k3]

Backend format (this repo):
  config/calibration/camera_intrinsic.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RosCameraModel:
    image_width: int
    image_height: int
    k: list[float]  # 9 elements
    d: list[float]  # at least 4 elements, commonly 5

    @property
    def fx(self) -> float:
        return float(self.k[0])

    @property
    def fy(self) -> float:
        return float(self.k[4])

    @property
    def cx(self) -> float:
        return float(self.k[2])

    @property
    def cy(self) -> float:
        return float(self.k[5])

    @property
    def k1(self) -> float:
        return float(self.d[0]) if len(self.d) > 0 else 0.0

    @property
    def k2(self) -> float:
        return float(self.d[1]) if len(self.d) > 1 else 0.0

    @property
    def p1(self) -> float:
        return float(self.d[2]) if len(self.d) > 2 else 0.0

    @property
    def p2(self) -> float:
        return float(self.d[3]) if len(self.d) > 3 else 0.0

    @property
    def k3(self) -> float:
        return float(self.d[4]) if len(self.d) > 4 else 0.0


def _require_dict(data: Any, *, what: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping for {what}, got: {type(data).__name__}")
    return data


def load_ros_camera_yaml(path: str | Path) -> RosCameraModel:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data = _require_dict(data, what=str(path))

    try:
        image_width = int(data["image_width"])
        image_height = int(data["image_height"])
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Missing/invalid image size fields in {path}") from e

    camera_matrix = _require_dict(data.get("camera_matrix"), what="camera_matrix")
    distortion_coeffs = _require_dict(data.get("distortion_coefficients"), what="distortion_coefficients")

    k = camera_matrix.get("data")
    d = distortion_coeffs.get("data")

    if not isinstance(k, list) or len(k) != 9:
        raise ValueError(f"camera_matrix.data must be a 9-element list, got: {k!r}")
    if not isinstance(d, list) or len(d) < 4:
        raise ValueError(f"distortion_coefficients.data must have >=4 elements, got: {d!r}")

    k_f = [float(x) for x in k]
    d_f = [float(x) for x in d]
    return RosCameraModel(image_width=image_width, image_height=image_height, k=k_f, d=d_f)


def to_backend_yaml_dict(
    ros: RosCameraModel,
    *,
    calibration_date: str | None = None,
    num_images: int = 0,
    reprojection_error_px: float = -1.0,
) -> dict:
    return {
        "calibration_date": calibration_date or datetime.now().isoformat(),
        "num_images": int(num_images),
        "image_size": {"width": int(ros.image_width), "height": int(ros.image_height)},
        "camera_matrix": {"fx": ros.fx, "fy": ros.fy, "cx": ros.cx, "cy": ros.cy},
        "distortion": {"k1": ros.k1, "k2": ros.k2, "p1": ros.p1, "p2": ros.p2, "k3": ros.k3},
        "reprojection_error_px": float(reprojection_error_px),
    }


def save_backend_yaml(data: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ros-yaml", required=True, help="ROS camera_calibration YAML (camera.yaml / ost.yaml)")
    parser.add_argument("--output", required=True, help="Output backend YAML (config/calibration/camera_intrinsic.yaml)")
    parser.add_argument(
        "--num-images",
        type=int,
        default=0,
        help="Number of images used (ROS yaml doesn't include this; default: 0)",
    )
    parser.add_argument(
        "--reprojection-error-px",
        type=float,
        default=-1.0,
        help="Reprojection error px (ROS yaml doesn't include this; default: -1 meaning unknown)",
    )
    args = parser.parse_args()

    ros_model = load_ros_camera_yaml(args.ros_yaml)
    backend = to_backend_yaml_dict(
        ros_model,
        num_images=args.num_images,
        reprojection_error_px=args.reprojection_error_px,
    )
    save_backend_yaml(backend, args.output)
    print(f"Wrote backend intrinsics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

