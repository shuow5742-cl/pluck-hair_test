#!/usr/bin/env python3
"""Auto-detect Daheng camera and write SN to settings yaml."""

import sys
import re
from pathlib import Path


def detect_camera_sn():
    try:
        import gxipy as gx
    except ImportError:
        print("gxipy not installed.")
        sys.exit(1)

    dev_mgr = gx.DeviceManager()
    num, device_info_list = dev_mgr.update_all_device_list()

    if num == 0:
        print("No Daheng cameras found.")
        sys.exit(1)

    sn = device_info_list[0].get("sn", "")
    model = device_info_list[0].get("model_name", "unknown")

    if not sn:
        print("Camera found but SN is empty.")
        sys.exit(1)

    print(f"Detected: {model} (SN: {sn})")
    return sn


def update_yaml(config_path: Path, sn: str):
    text = config_path.read_text()

    # Replace existing device_sn line
    if re.search(r"^\s*device_sn:", text, re.MULTILINE):
        text = re.sub(
            r"^(\s*)device_sn:.*$",
            rf'\1device_sn: "{sn}"',
            text,
            flags=re.MULTILINE,
        )
    # Replace device_index with device_sn
    elif re.search(r"^\s*device_index:", text, re.MULTILINE):
        text = re.sub(
            r"^(\s*)device_index:.*$",
            rf'\1device_sn: "{sn}"',
            text,
            flags=re.MULTILINE,
        )
    else:
        print(f"Cannot find device_sn or device_index in {config_path}")
        sys.exit(1)

    config_path.write_text(text)
    print(f"Updated {config_path}: device_sn = \"{sn}\"")


def main():
    config_path = Path("config/settings.dev.yaml")
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    sn = detect_camera_sn()
    update_yaml(config_path, sn)


if __name__ == "__main__":
    main()
