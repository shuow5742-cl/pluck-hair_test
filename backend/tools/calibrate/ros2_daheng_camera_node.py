#!/usr/bin/env python3
"""
ROS2 image publisher for Daheng (gxipy) industrial cameras.

Publishes:
  - sensor_msgs/Image on `/image_raw` (BGR8)
  - sensor_msgs/CameraInfo on `/camera_info` (empty or loaded from a ROS YAML)

This is intended to feed `camera_calibration`'s `cameracalibrator`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def _load_ros_camera_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid ROS camera yaml: {path}")
    return data


def _device_info_str(info: Any) -> str:
    if isinstance(info, dict):
        keys = ["sn", "serial_number", "serial", "model", "device_id", "vendor", "name"]
        parts = []
        for k in keys:
            if k in info:
                parts.append(f"{k}={info[k]}")
        if parts:
            return ", ".join(parts)
    return repr(info)


def list_daheng_devices() -> int:
    try:
        import gxipy as gx
    except Exception as e:  # noqa: BLE001
        print(f"Failed to import gxipy: {e}", file=sys.stderr)
        return 2

    try:
        gx.gx_init_lib()
    except Exception as e:  # noqa: BLE001
        print(f"Failed to init Daheng API (gx_init_lib): {e}", file=sys.stderr)
        return 2

    dev_mgr = gx.DeviceManager()
    num, info_list = dev_mgr.update_device_list()
    print(f"Found {num} Daheng device(s)")
    if info_list is None:
        try:
            gx.gx_close_lib()
        except Exception:
            pass
        return 0
    try:
        for i, info in enumerate(info_list, start=1):
            print(f"  [{i}] {_device_info_str(info)}")
    except Exception:
        print(f"Device info: {info_list!r}")
    try:
        gx.gx_close_lib()
    except Exception:
        pass
    return 0


def _open_camera(dev_mgr: Any, device_index: int, serial: str | None, device_id: str | None) -> Any:
    num, info_list = dev_mgr.update_device_list()
    if num == 0:
        raise RuntimeError("No Daheng camera found (gxipy)")

    if serial:
        if hasattr(dev_mgr, "open_device_by_sn"):
            return dev_mgr.open_device_by_sn(serial)

        # Best-effort: find an index in info_list
        if isinstance(info_list, list):
            target_index = None
            for i, info in enumerate(info_list, start=1):
                if isinstance(info, dict):
                    for key in ("sn", "serial_number", "serial", "SN", "SerialNumber"):
                        if key in info and serial in str(info[key]):
                            target_index = i
                            break
                if target_index is None and serial in repr(info):
                    target_index = i
                if target_index is not None:
                    break
            if target_index is None:
                raise RuntimeError(f"Serial `{serial}` not found in device list")
            return dev_mgr.open_device_by_index(target_index)

        raise RuntimeError("Cannot match serial without a usable device info list; set `device_index` instead")

    if device_id:
        if isinstance(info_list, list):
            matches = []
            for i, info in enumerate(info_list, start=1):
                if device_id in repr(info):
                    matches.append(i)
                elif isinstance(info, dict) and device_id in _device_info_str(info):
                    matches.append(i)
            if len(matches) == 1:
                return dev_mgr.open_device_by_index(matches[0])
            if len(matches) > 1:
                raise RuntimeError(f"Multiple Daheng devices match device_id `{device_id}`: {matches}")
            raise RuntimeError(f"No Daheng devices match device_id `{device_id}`")
        raise RuntimeError("Cannot match device_id without a usable device info list; set `device_index` instead")

    return dev_mgr.open_device_by_index(device_index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List Daheng devices and exit")
    parser.add_argument("--device-index", type=int, default=1, help="Daheng device index (1-based)")
    parser.add_argument("--serial", type=str, default=None, help="Match by serial number (preferred if multiple)")
    parser.add_argument("--device-id", type=str, default=None, help="Match by device id substring (best-effort)")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--frame-id", type=str, default="camera_link")
    parser.add_argument("--image-topic", type=str, default="/image_raw")
    parser.add_argument("--camera-info-topic", type=str, default="/camera_info")
    parser.add_argument("--camera-info-yaml", type=str, default=None, help="ROS camera.yaml/ost.yaml to publish")
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    if args.list:
        return list_daheng_devices()

    try:
        import numpy as np
        import gxipy as gx
    except Exception as e:  # noqa: BLE001
        print(f"Missing dependency: {e}", file=sys.stderr)
        return 2

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
    except Exception as e:  # noqa: BLE001
        print(f"ROS2 Python not available (did you source ROS2 env?): {e}", file=sys.stderr)
        return 2

    camera_info_yaml = _load_ros_camera_yaml(args.camera_info_yaml) if args.camera_info_yaml else None

    class DahengNode(Node):
        def __init__(self) -> None:
            super().__init__("daheng_camera_node")
            self.image_pub = self.create_publisher(Image, args.image_topic, qos_profile_sensor_data)
            self.info_pub = self.create_publisher(CameraInfo, args.camera_info_topic, qos_profile_sensor_data)

            # Important: keep the DeviceManager alive for the lifetime of the camera.
            # If it gets garbage-collected, gxipy may de-init the underlying API and `stream_on()` will fail.
            gx.gx_init_lib()
            self._gx = gx
            self.dev_mgr = gx.DeviceManager()
            self.cam = _open_camera(self.dev_mgr, args.device_index, args.serial, args.device_id)
            self._configure_camera()
            self.cam.stream_on()

            self.period = 1.0 / max(args.fps, 0.1)
            self.last_info_publish = 0.0
            self.timer = self.create_timer(self.period, self._tick)

            self.get_logger().info("Daheng camera streaming started")

        def _configure_camera(self) -> None:
            # Best-effort resolution setting; not all models expose these setters.
            try:
                if hasattr(self.cam, "Width"):
                    self.cam.Width.set(int(args.width))
                if hasattr(self.cam, "Height"):
                    self.cam.Height.set(int(args.height))
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"Failed to set resolution: {e}")

        def _build_camera_info(self, width: int, height: int) -> CameraInfo:
            msg = CameraInfo()
            msg.header.frame_id = args.frame_id
            msg.width = int(width)
            msg.height = int(height)

            if camera_info_yaml:
                try:
                    msg.width = int(camera_info_yaml.get("image_width", msg.width))
                    msg.height = int(camera_info_yaml.get("image_height", msg.height))
                    msg.distortion_model = str(camera_info_yaml.get("distortion_model", "plumb_bob"))

                    d = camera_info_yaml.get("distortion_coefficients", {}).get("data", [])
                    k = camera_info_yaml.get("camera_matrix", {}).get("data", [])
                    r = camera_info_yaml.get("rectification_matrix", {}).get("data", [])
                    p = camera_info_yaml.get("projection_matrix", {}).get("data", [])

                    msg.d = [float(x) for x in d]
                    msg.k = [float(x) for x in k] if isinstance(k, list) and len(k) == 9 else [0.0] * 9
                    msg.r = [float(x) for x in r] if isinstance(r, list) and len(r) == 9 else [0.0] * 9
                    msg.p = [float(x) for x in p] if isinstance(p, list) and len(p) == 12 else [0.0] * 12
                except Exception as e:  # noqa: BLE001
                    self.get_logger().warn(f"Failed to parse camera_info yaml, publishing empty CameraInfo: {e}")

            return msg

        def _tick(self) -> None:
            raw = self.cam.data_stream[0].get_image()
            if raw is None or raw.get_status() != gx.GxFrameStatusList.SUCCESS:
                return

            rgb = raw.convert("RGB", channel_order=gx.DxRGBChannelOrder.ORDER_BGR)
            if rgb is None:
                return

            frame = rgb.get_numpy_array()
            if frame is None:
                return

            if frame.ndim != 3 or frame.shape[2] != 3:
                return

            height, width = frame.shape[:2]

            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = args.frame_id
            msg.height = int(height)
            msg.width = int(width)
            msg.encoding = "bgr8"
            msg.is_bigendian = 0
            msg.step = int(width * 3)
            msg.data = frame.astype(np.uint8).tobytes()
            self.image_pub.publish(msg)

            now = time.time()
            if now - self.last_info_publish > 1.0:
                info = self._build_camera_info(width, height)
                info.header.stamp = msg.header.stamp
                self.info_pub.publish(info)
                self.last_info_publish = now

        def destroy_node(self) -> bool:
            try:
                self.cam.stream_off()
            except Exception:
                pass
            try:
                self.cam.close_device()
            except Exception:
                pass
            try:
                self._gx.gx_close_lib()
            except Exception:
                pass
            return super().destroy_node()

    rclpy.init()
    node = DahengNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        # This commonly happens on external shutdown (e.g., SIGINT from a runner).
        if type(e).__name__ != "ExternalShutdownException":
            raise
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
