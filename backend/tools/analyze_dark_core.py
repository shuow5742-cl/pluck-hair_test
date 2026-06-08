#!/usr/bin/env python3
"""Decompose ``_pick_from_broad_dark_core`` into its weight layers so we can
see what's actually steering the pick point on a given mask.

Saves a row of heatmap PNGs alongside the input mask:
  01_gray_mask.png        gray pixels within the YOLO mask
  02_dark.png             absolute darkness map (p90 - gray)
  03_local_dark.png       local darkness (median-blur background - gray)
  04_dist_norm.png        distance-to-mask-boundary normalized
  05_raw_weight.png       broad_dark_core final weight (the picker's argmax)
  06_candidates.png       original image with every candidate pick overlaid

Candidates compared:
  - cyan:    current algorithm pick (broad_dark_core_center)
  - red:     argmax(raw_weight)            ← the picker's true target
  - yellow:  argmax(dark)                  ← darkest single point
  - magenta: argmax(local_dark)            ← strongest local contrast
  - green:   argmax(dist_norm) (mask thickest point, ignoring darkness)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.pick_point_estimator import _odd_kernel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", required=True,
                        help="A segmentation_outputs/<frame>/ directory with original.png, result.json, masks/")
    parser.add_argument("--target-index", type=int, default=0, help="Which detection within the frame")
    parser.add_argument("--out-dir", default=None, help="Override output directory (defaults to frame-dir)")
    return parser.parse_args()


def colorize(weight: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    w = weight.astype(np.float32)
    lo = float(w.min()) if vmin is None else float(vmin)
    hi = float(w.max()) if vmax is None else float(vmax)
    if hi - lo < 1e-9:
        norm = np.zeros_like(w, dtype=np.uint8)
    else:
        norm = np.clip(((w - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


def main() -> None:
    args = parse_args()
    frame_dir = Path(args.frame_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else frame_dir / "dark_core_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = json.loads((frame_dir / "result.json").read_text())
    det = result["detections"][args.target_index]

    image_bgr = cv2.imread(str(frame_dir / "original.png"))
    if image_bgr is None:
        raise SystemExit(f"cannot read {frame_dir/'original.png'}")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Find the mask file for this target.
    mask_paths = sorted((frame_dir / "masks").glob(f"target_{args.target_index:03d}_*_mask_full.png"))
    if not mask_paths:
        raise SystemExit(f"no mask_full.png for target index {args.target_index}")
    yolo_mask = cv2.imread(str(mask_paths[0]), cv2.IMREAD_GRAYSCALE)
    if yolo_mask is None:
        raise SystemExit(f"cannot read {mask_paths[0]}")

    # Recompute broad_dark_core internals with the SAME maths as the picker.
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
    pad = max(8, int(max(x2 - x1, y2 - y1) * 0.04))
    rx1 = max(0, x1 - pad); ry1 = max(0, y1 - pad)
    rx2 = min(w, x2 + pad); ry2 = min(h, y2 + pad)

    roi_gray = gray[ry1:ry2, rx1:rx2]
    roi_mask = (yolo_mask[ry1:ry2, rx1:rx2] > 0).astype(np.uint8)
    safe_mask = roi_mask.copy()  # no metal-plate subtraction for this analysis

    vals = roi_gray[safe_mask > 0].astype(np.float32)
    p10 = float(np.percentile(vals, 10))
    p50 = float(np.percentile(vals, 50))
    p90 = float(np.percentile(vals, 90))
    contrast = max(p90 - p10, 1.0)

    dist_obj = cv2.distanceTransform((safe_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    max_obj_dist = max(float(dist_obj.max()), 1.0)
    dist_norm = np.clip(dist_obj / max_obj_dist, 0.0, 1.0).astype(np.float32)

    inner_thr = max(1.5, min(10.0, max_obj_dist * 0.10))
    inner_mask = ((safe_mask > 0) & (dist_obj >= inner_thr)).astype(np.uint8)
    safe_area = int(safe_mask.sum())
    if int(inner_mask.sum()) < max(30, int(safe_area * 0.05)):
        inner_mask = safe_mask.copy()

    dark = np.clip((p90 - roi_gray.astype(np.float32)) / contrast, 0.0, 1.0)
    k = _odd_kernel(max(31, min(151, int(max(roi_gray.shape[:2]) * 0.24))))
    local_bg = cv2.medianBlur(roi_gray, k).astype(np.float32)
    local_dark = np.clip((local_bg - roi_gray.astype(np.float32)) / contrast, 0.0, 1.0)

    raw_weight = (0.70 * np.power(dark, 2.2) + 0.30 * np.power(local_dark, 1.6)).astype(np.float32)
    raw_weight[inner_mask == 0] = 0.0
    raw_weight *= (0.35 + 0.65 * np.power(dist_norm, 0.45)).astype(np.float32)

    def to_global(yx_local: tuple[int, int]) -> tuple[int, int]:
        return (yx_local[0] + ry1, yx_local[1] + rx1)

    # Candidate picks
    def argmax_in_mask(field: np.ndarray) -> tuple[int, int]:
        masked = field.copy()
        masked[safe_mask == 0] = -1
        flat = int(np.argmax(masked))
        y, x = np.unravel_index(flat, masked.shape)
        return (int(y), int(x))

    argmax_weight_local = argmax_in_mask(raw_weight)
    argmax_dark_local = argmax_in_mask(dark)
    argmax_local_dark_local = argmax_in_mask(local_dark)
    argmax_dist_local = argmax_in_mask(dist_norm)

    argmax_weight = to_global(argmax_weight_local)
    argmax_dark = to_global(argmax_dark_local)
    argmax_local_dark = to_global(argmax_local_dark_local)
    argmax_dist = to_global(argmax_dist_local)

    algo_pick = det.get("pick_point_xy")
    if algo_pick is not None:
        algo_pick = (int(round(algo_pick[1])), int(round(algo_pick[0])))  # (y, x)

    # --- Save layers ---
    def saved_msg(name: str, arr: np.ndarray) -> None:
        p = out_dir / name
        cv2.imwrite(str(p), arr)
        print(f"  wrote {p.relative_to(frame_dir.parent.parent)}  shape={arr.shape}")

    overlay_gray = roi_gray.copy()
    overlay_gray[safe_mask == 0] = 0
    saved_msg("01_gray_in_mask.png", overlay_gray)
    saved_msg("02_dark.png", colorize(dark * safe_mask))
    saved_msg("03_local_dark.png", colorize(local_dark * safe_mask))
    saved_msg("04_dist_norm.png", colorize(dist_norm * safe_mask))
    saved_msg("05_raw_weight.png", colorize(raw_weight))

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # mask overlay
    color_mask = np.zeros_like(overlay)
    color_mask[yolo_mask > 0] = (60, 90, 240)
    overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)

    def draw(point_yx, color, label):
        y, x = point_yx
        cv2.drawMarker(overlay, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=26, thickness=3)
        cv2.putText(overlay, label, (x + 14, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    if algo_pick is not None:
        draw(algo_pick, (255, 255, 0), f"algo (broad_dark_core)")
    draw(argmax_weight, (0, 0, 255), "argmax raw_weight")
    draw(argmax_dark, (0, 220, 220), "argmax dark")
    draw(argmax_local_dark, (255, 0, 255), "argmax local_dark")
    draw(argmax_dist, (0, 255, 0), "argmax dist_norm")

    saved_msg("06_candidates.png", overlay)

    print("\n=== Candidate pick comparison (full-image y, x) ===")
    if algo_pick is not None:
        print(f"  algo (broad_dark_core_center):  {algo_pick}")
    print(f"  argmax(raw_weight, picker truth): {argmax_weight}")
    print(f"  argmax(dark):                     {argmax_dark}")
    print(f"  argmax(local_dark):               {argmax_local_dark}")
    print(f"  argmax(dist_norm):                {argmax_dist}")
    print(f"\n=== Mask gray statistics ===")
    print(f"  p10={p10:.1f}  p50={p50:.1f}  p90={p90:.1f}  contrast={p90-p10:.1f}")
    print(f"  median_blur kernel = {k}px")
    print(f"  max_obj_dist (mask half-thickness) = {max_obj_dist:.1f}px")
    print(f"  inner_thr (edge guard) = {inner_thr:.1f}px")
    print(f"  inner_mask area = {int(inner_mask.sum())} / safe_area {safe_area}")


if __name__ == "__main__":
    main()
