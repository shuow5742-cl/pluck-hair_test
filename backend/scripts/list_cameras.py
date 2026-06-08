#!/usr/bin/env python3
"""List all connected Daheng cameras with their details."""

import sys


def main():
    try:
        import gxipy as gx
    except ImportError:
        print("gxipy not installed. Install with: pip install iai-gxipy")
        sys.exit(1)

    dev_mgr = gx.DeviceManager()
    num, device_info_list = dev_mgr.update_all_device_list()

    if num == 0:
        print("No Daheng cameras found.")
        sys.exit(0)

    print(f"Found {num} Daheng camera(s):\n")
    print(f"{'Index':<8}{'SN':<25}{'Model':<25}{'Interface':<12}{'IP/MAC'}")
    print("-" * 80)

    for i, info in enumerate(device_info_list, start=1):
        sn = info.get("sn", "N/A")
        model = info.get("model_name", "N/A")
        iface = info.get("device_class", "N/A")
        ip = info.get("ip", "")
        mac = info.get("mac", "")
        addr = ip or mac or ""

        print(f"{i:<8}{sn:<25}{model:<25}{iface:<12}{addr}")

    print()
    print("Usage: set device_sn in config/settings.dev.yaml:")
    print("  camera:")
    print("    type: daheng")
    print('    device_sn: "<SN from above>"')


if __name__ == "__main__":
    main()
