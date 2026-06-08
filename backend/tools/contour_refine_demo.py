#!/usr/bin/env python3
"""Demo script: refine pick point using contour detection on a cropped ROI.

Usage:
    # With bbox (x1,y1,x2,y2):
    python -m tools.contour_refine_demo \
        --image path/to/image.jpg \
        --bbox 100,200,300,400

    # With center+size (cx,cy,w,h):
    python -m tools.contour_refine_demo \
        --image path/to/image.jpg \
        --center 200,300 --size 200,200

    # Adjust padding around bbox:
    python -m tools.contour_refine_demo \
        --image path/to/image.jpg \
        --bbox 100,200,300,400 --padding 20

Output:
    Saves visualization to ./output_contour_refine.png (or --output path).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def crop_roi(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    """Crop ROI from image with optional padding, clamped to image bounds."""
    h, w = image.shape[:2]
    x1c = max(0, x1 - padding)
    y1c = max(0, y1 - padding)
    x2c = min(w, x2 + padding)
    y2c = min(h, y2 + padding)
    roi = image[y1c:y2c, x1c:x2c]
    return roi, (x1c, y1c)


def find_dark_contours(roi_bgr: np.ndarray, block_size: int = 25, c: int = 8):
    """Find dark object contours in a ROI using adaptive thresholding.

    Args:
        roi_bgr: Cropped BGR image.
        block_size: Block size for adaptive threshold (must be odd).
        c: Constant subtracted from mean in adaptive threshold.

    Returns:
        contours: List of contours.
        mask: Binary mask (dark objects = 255).
        gray: Grayscale image used.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold: dark objects become white (255) in the mask
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block_size, c,
    )

    # Morphological close to connect fragmented hair segments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Remove tiny noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask, gray


def compute_centroid(contour) -> tuple[float, float] | None:
    """Compute centroid of a contour using moments."""
    m = cv2.moments(contour)
    if m["m00"] < 1e-6:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return (cx, cy)


def refine_pick_point(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    padding: int = 10,
    block_size: int = 25,
    c: int = 8,
    min_area: int = 20,
):
    """Find the refined pick point within a bounding box.

    Returns:
        refined_center: (x, y) in original image coordinates, or bbox center as fallback.
        debug_info: dict with intermediate results for visualization.
    """
    roi, (ox, oy) = crop_roi(image, x1, y1, x2, y2, padding)
    contours, mask, gray = find_dark_contours(roi, block_size, c)

    # Filter by minimum area
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)

    if not contours:
        return bbox_center, {
            "roi": roi, "mask": mask, "gray": gray,
            "contours": [], "origin": (ox, oy),
            "fallback": True,
        }

    # Pick the largest contour (most likely the hair)
    largest = max(contours, key=cv2.contourArea)
    centroid_local = compute_centroid(largest)

    if centroid_local is None:
        return bbox_center, {
            "roi": roi, "mask": mask, "gray": gray,
            "contours": contours, "origin": (ox, oy),
            "fallback": True,
        }

    # Convert to original image coordinates
    refined = (centroid_local[0] + ox, centroid_local[1] + oy)

    return refined, {
        "roi": roi, "mask": mask, "gray": gray,
        "contours": contours, "origin": (ox, oy),
        "largest": largest, "centroid_local": centroid_local,
        "fallback": False,
    }


def visualize(image, x1, y1, x2, y2, refined_center, debug_info, output_path):
    """Build a visualization panel and save it."""
    ox, oy = debug_info["origin"]
    roi = debug_info["roi"]
    mask = debug_info["mask"]

    # === Panel 1: Full image with bbox and centers ===
    panel_full = image.copy()
    cv2.rectangle(panel_full, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Original bbox center (red)
    bcx, bcy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    cv2.drawMarker(panel_full, (bcx, bcy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(panel_full, "bbox center", (bcx + 10, bcy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Refined center (cyan)
    rcx, rcy = int(refined_center[0]), int(refined_center[1])
    cv2.drawMarker(panel_full, (rcx, rcy), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(panel_full, "refined", (rcx + 10, rcy + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Offset text
    dx = refined_center[0] - bcx
    dy = refined_center[1] - bcy
    info_text = f"offset: ({dx:+.1f}, {dy:+.1f}) px"
    cv2.putText(panel_full, info_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # === Panel 2: ROI with contours ===
    panel_roi = roi.copy()
    if debug_info["contours"]:
        cv2.drawContours(panel_roi, debug_info["contours"], -1, (0, 255, 0), 1)
    if not debug_info["fallback"]:
        cl = debug_info["centroid_local"]
        cv2.drawMarker(panel_roi, (int(cl[0]), int(cl[1])),
                       (255, 255, 0), cv2.MARKER_CROSS, 15, 2)

    # === Panel 3: Binary mask ===
    panel_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Resize panels 2 & 3 to match height ratio for display
    target_h = min(400, panel_full.shape[0])
    panels_right = []
    for p in [panel_roi, panel_mask]:
        if p.shape[0] > 0:
            scale = target_h / 2 / p.shape[0]
            resized = cv2.resize(p, (int(p.shape[1] * scale), int(target_h / 2)))
            panels_right.append(resized)

    if panels_right:
        # Stack ROI and mask vertically
        max_w = max(p.shape[1] for p in panels_right)
        panels_padded = []
        for p in panels_right:
            if p.shape[1] < max_w:
                pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
                p = np.hstack([p, pad])
            panels_padded.append(p)
        right_col = np.vstack(panels_padded)

        # Resize full image to match
        scale_full = target_h / panel_full.shape[0]
        panel_full_resized = cv2.resize(
            panel_full, (int(panel_full.shape[1] * scale_full), target_h)
        )

        # Pad heights to match
        if right_col.shape[0] != panel_full_resized.shape[0]:
            target = max(right_col.shape[0], panel_full_resized.shape[0])
            if right_col.shape[0] < target:
                pad = np.zeros((target - right_col.shape[0], right_col.shape[1], 3), dtype=np.uint8)
                right_col = np.vstack([right_col, pad])
            if panel_full_resized.shape[0] < target:
                pad = np.zeros((target - panel_full_resized.shape[0], panel_full_resized.shape[1], 3), dtype=np.uint8)
                panel_full_resized = np.vstack([panel_full_resized, pad])

        result = np.hstack([panel_full_resized, right_col])
    else:
        result = panel_full

    cv2.imwrite(str(output_path), result)
    print(f"Saved to {output_path}")
    print(f"  BBox center:    ({bcx}, {bcy})")
    print(f"  Refined center: ({rcx}, {rcy})")
    print(f"  Offset:         ({dx:+.1f}, {dy:+.1f}) px")
    if debug_info["fallback"]:
        print("  (fallback to bbox center — no contour found)")


def main():
    parser = argparse.ArgumentParser(description="Contour-based pick point refinement demo")
    parser.add_argument("--image", required=True, help="Input image path")

    # BBox specification (either --bbox or --center + --size)
    parser.add_argument("--bbox", help="Bounding box as x1,y1,x2,y2")
    parser.add_argument("--center", help="Box center as cx,cy")
    parser.add_argument("--size", help="Box size as w,h")

    parser.add_argument("--padding", type=int, default=10, help="Padding around bbox (px)")
    parser.add_argument("--block-size", type=int, default=25, help="Adaptive threshold block size")
    parser.add_argument("-c", type=int, default=8, help="Adaptive threshold constant")
    parser.add_argument("--min-area", type=int, default=20, help="Minimum contour area")
    parser.add_argument("--output", default="output_contour_refine.png", help="Output path")

    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: cannot read image: {args.image}")
        return

    # Parse bbox
    if args.bbox:
        x1, y1, x2, y2 = map(int, args.bbox.split(","))
    elif args.center and args.size:
        cx, cy = map(int, args.center.split(","))
        w, h = map(int, args.size.split(","))
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = x1 + w, y1 + h
    else:
        print("Error: provide --bbox x1,y1,x2,y2 or --center cx,cy --size w,h")
        return

    print(f"Image: {args.image} ({image.shape[1]}x{image.shape[0]})")
    print(f"BBox:  ({x1},{y1}) -> ({x2},{y2})")

    refined, debug_info = refine_pick_point(
        image, x1, y1, x2, y2,
        padding=args.padding,
        block_size=args.block_size,
        c=args.c,
        min_area=args.min_area,
    )

    visualize(image, x1, y1, x2, y2, refined, debug_info, args.output)


if __name__ == "__main__":
    main()
