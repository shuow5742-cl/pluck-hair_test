#!/usr/bin/env python3
"""Manual focus-stack probe — operator drives nova5 by hand, presses 's' per slice.

Goal of this tool: validate the focus-stack fusion idea on the rig BEFORE
wiring it into the production pipeline. No PLC, no comms — the operator
manually nudges nova5 up by 1 mm between captures, so delta_z is dictated
purely by "how many times has 's' been pressed":

    slice 0 → delta_z = 0 mm  (z_tray reference)
    slice 1 → delta_z = 1 mm
    ...
    slice 6 → delta_z = 6 mm

After 7 slices the tool fuses them via core.focus_stack.fuse_focus_stack
and writes raw / fused / height_map / sharpness to disk.

Pipeline used for the live preview is just ``crop_single_square`` so the
operator sees the matched cell at full preview resolution. The fused
stack also operates on the cropped frames (so ECC translation only has
to fix sub-pixel drift, not full-frame shifts).

Keys (focus the preview window first):
    s   capture current frame as the next slice (delta_z auto-increments)
    u   undo the last captured slice
    f   force-fuse with what we have (>=2 slices), then exit
    r   reset stack (drop all captures, start over)
    q   quit without fusing (also: Esc)

Usage::

    cd backend
    uv run python tools/focus_stack_probe.py
    # custom step count (default 7 slices = 0..6 mm in 1 mm increments):
    uv run python tools/focus_stack_probe.py --slices 7 --step-mm 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.steps  # noqa: F401 — register hub-side pipeline step types

from autoweaver.camera import CameraConfig as BaseCameraConfig  # noqa: E402
from autoweaver.camera import DahengCamera  # noqa: E402
from autoweaver.pipeline import VisionPipeline  # noqa: E402

from src.core.focus_stack import fuse_focus_stack  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--device-sn", default="FCM26010005",
        help="Daheng camera serial number (matches settings.live.yaml).",
    )
    p.add_argument(
        "--exposure-us", type=float, default=20000.0,
        help="Manual exposure time in microseconds.",
    )
    p.add_argument(
        "--slices", type=int, default=7,
        help="Total number of captures. Default 7 = delta_z 0..6 mm.",
    )
    p.add_argument(
        "--step-mm", type=float, default=1.0,
        help="Z step between consecutive captures, in mm.",
    )
    p.add_argument(
        "--output-dir", default="data/focus_stack_probe",
        help="Where the per-run directory goes.",
    )
    p.add_argument(
        "--preview-scale", type=float, default=0.5,
        help="Downscale the preview window so 2048x1536 fits on screen.",
    )
    p.add_argument(
        "--sharpness-kernel", type=int, default=31,
        help="Local window for Laplacian-variance sharpness map.",
    )
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


def build_crop_pipeline() -> VisionPipeline:
    """Match the crop_single_square params from settings.live.yaml."""
    return VisionPipeline.from_config({
        "pipeline": {
            "steps": [
                {
                    "name": "crop_square",
                    "type": "crop_single_square",
                    "params": {
                        "mm_per_pixel": 0.009857,
                        "cell_size_mm": 10.0,
                        "chamfer_mm": 0.5,
                        "frame_mm": 0.5,
                        "metal_threshold": 100,
                        "center_bias": 0.15,
                        "min_match_score": 0.40,
                        "border_px": 176,
                        "fallback_side_px": 1100,
                    },
                },
            ]
        }
    })


def open_camera(args: argparse.Namespace) -> DahengCamera:
    cam_cfg = BaseCameraConfig(
        device_sn=args.device_sn,
        exposure_auto=False,
        gain_auto=False,
        exposure_time=args.exposure_us,
        white_balance_mode="once",
    )
    cam = DahengCamera(cam_cfg)
    if not cam.open():
        raise RuntimeError(f"Failed to open Daheng SN={args.device_sn}")
    return cam


@dataclass
class Slice:
    image: np.ndarray
    delta_z_mm: float
    captured_at: float


def overlay_hud(
    canvas: np.ndarray,
    *,
    captured: int,
    target: int,
    next_delta_z_mm: float,
    last_action: Optional[str],
) -> None:
    lines = [
        f"slices: {captured}/{target}    next delta_z = +{next_delta_z_mm:.2f} mm",
        "[s] capture  [u] undo  [f] force fuse  [r] reset  [q] quit",
    ]
    if last_action:
        lines.append(last_action)
    y = 30
    for line in lines:
        cv2.putText(canvas, line, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        y += 32


def crop_frame(pipeline: VisionPipeline, frame: np.ndarray) -> np.ndarray:
    """Run only crop_single_square, return the cropped ROI."""
    result = pipeline.run(frame)
    cropped = getattr(result, "processed_image", None)
    if cropped is None or cropped.size == 0:
        return frame
    return cropped


def normalize_to_u8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros(a.shape[:2], dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def save_results(
    out_dir: Path,
    slices: list[Slice],
    fused_rgb: np.ndarray,
    height_map: np.ndarray,
    sharpness: np.ndarray,
    aligned_count: int,
    sharpness_kernel: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, sl in enumerate(slices):
        cv2.imwrite(str(out_dir / f"raw_{i:02d}.png"), sl.image)

    cv2.imwrite(str(out_dir / "fused.png"), fused_rgb)
    np.save(out_dir / "height_map.npy", height_map)
    cv2.imwrite(str(out_dir / "height_map.png"), normalize_to_u8(height_map))
    height_color = cv2.applyColorMap(normalize_to_u8(height_map), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out_dir / "height_map_color.png"), height_color)
    cv2.imwrite(str(out_dir / "sharpness.png"), normalize_to_u8(sharpness))

    stats = {
        "slice_count": len(slices),
        "delta_z_mm": [sl.delta_z_mm for sl in slices],
        "captured_at": [sl.captured_at for sl in slices],
        "aligned_count": aligned_count,
        "sharpness_kernel": sharpness_kernel,
        "height_map_min_mm": float(height_map.min()),
        "height_map_max_mm": float(height_map.max()),
        "height_map_median_mm": float(np.median(height_map)),
    }
    (out_dir / "stack.json").write_text(json.dumps(stats, indent=2))


def fuse_and_show(
    slices: list[Slice],
    out_dir: Path,
    sharpness_kernel: int,
) -> None:
    """Run fusion, save artifacts, and pop a viewer window for fused + height map."""
    if len(slices) < 2:
        print(f"[probe] need at least 2 slices to fuse; have {len(slices)}")
        return
    print(f"[probe] fusing {len(slices)} slices, this may take a few seconds...")
    t0 = time.time()
    # delta_z is measured "above the first slice" (z_tray = first slice).
    # That makes height_map directly readable as "mm of clump above the
    # touchdown plane" without any PLC reference.
    result = fuse_focus_stack(
        frames=[sl.image for sl in slices],
        z_values_mm=[sl.delta_z_mm for sl in slices],
        z_tray_flange_mm=0.0,
        sharpness_kernel=sharpness_kernel,
    )
    elapsed = time.time() - t0

    save_results(
        out_dir, slices, result.fused_rgb, result.height_map_mm,
        result.sharpness_map, result.aligned_count, sharpness_kernel,
    )
    print(
        f"[probe] fused in {elapsed:.2f}s, aligned {result.aligned_count}/"
        f"{len(slices)}, height range "
        f"{float(result.height_map_mm.min()):.2f} → "
        f"{float(result.height_map_mm.max()):.2f} mm"
    )
    print(f"[probe] artifacts written to {out_dir}")

    # Side-by-side: fused | colorized height map
    fused = result.fused_rgb
    hcolor = cv2.applyColorMap(normalize_to_u8(result.height_map_mm), cv2.COLORMAP_TURBO)
    if hcolor.shape != fused.shape:
        hcolor = cv2.resize(hcolor, (fused.shape[1], fused.shape[0]))
    panel = np.hstack([fused, hcolor])
    cv2.namedWindow("focus_stack result (any key to close)", cv2.WINDOW_NORMAL)
    cv2.imshow("focus_stack result (any key to close)", panel)
    cv2.waitKey(0)
    cv2.destroyWindow("focus_stack result (any key to close)")


def main() -> None:
    args = parse_args()

    pipeline = build_crop_pipeline()
    cam = open_camera(args)

    output_root = resolve(args.output_dir)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id

    window = "focus_stack_probe (s=capture, u=undo, f=fuse, r=reset, q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    slices: list[Slice] = []
    last_action: Optional[str] = None

    print(
        f"[probe] target slices = {args.slices}, "
        f"step = {args.step_mm:.2f} mm; output dir = {run_dir}"
    )
    print("[probe] move nova5 to the start position, then press 's' for slice 0 (delta_z=0).")

    try:
        while True:
            frame = cam.capture()
            cropped = crop_frame(pipeline, frame)

            captured = len(slices)
            next_delta_z = captured * args.step_mm
            display = cropped.copy()
            overlay_hud(
                display,
                captured=captured,
                target=args.slices,
                next_delta_z_mm=next_delta_z,
                last_action=last_action,
            )

            if args.preview_scale != 1.0:
                preview = cv2.resize(
                    display, None,
                    fx=args.preview_scale, fy=args.preview_scale,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                preview = display
            cv2.imshow(window, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[probe] quit without fusing.")
                return
            if key == ord("s"):
                if len(slices) >= args.slices:
                    last_action = (
                        f"already at {args.slices} slices, press 'f' to fuse "
                        "or 'r' to reset"
                    )
                    continue
                slices.append(Slice(
                    image=cropped.copy(),
                    delta_z_mm=next_delta_z,
                    captured_at=time.time(),
                ))
                print(
                    f"[probe] captured slice {len(slices)-1}: delta_z={next_delta_z:.2f} mm"
                )
                last_action = (
                    f"captured slice {len(slices)-1} @ +{next_delta_z:.2f} mm"
                )
                if len(slices) == args.slices:
                    fuse_and_show(slices, run_dir, args.sharpness_kernel)
                    return
            elif key == ord("u"):
                if not slices:
                    last_action = "nothing to undo"
                else:
                    dropped = slices.pop()
                    last_action = (
                        f"undid slice {len(slices)} (was +{dropped.delta_z_mm:.2f} mm)"
                    )
                    print(f"[probe] {last_action}")
            elif key == ord("r"):
                slices.clear()
                last_action = "reset (0 slices)"
                print(f"[probe] {last_action}")
            elif key == ord("f"):
                if len(slices) < 2:
                    last_action = f"need >=2 slices to fuse, have {len(slices)}"
                    continue
                fuse_and_show(slices, run_dir, args.sharpness_kernel)
                return
    finally:
        cv2.destroyAllWindows()
        cam.close()


if __name__ == "__main__":
    main()
