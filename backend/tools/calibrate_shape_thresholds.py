#!/usr/bin/env python3
"""Compute shape descriptors over labeled polygons and report distribution.

Used to calibrate the 6 module-level thresholds in pick_point_estimator.py
without depending on YOLO output noise: each labelme polygon is rasterized
to a binary mask and run through _shape_descriptors + _classify_shape.

Output: per-shape rows (one per labeled polygon) + summary stats per class.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.pick_point_estimator import (  # noqa: E402
    _classify_shape,
    _has_dense_core,
    _shape_descriptors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/labeled/05.15_4_2",
        help="Directory containing labelme JSON + matching PNG.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output of per-polygon descriptors.",
    )
    return parser.parse_args()


def rasterize_polygon(points: list[list[float]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return mask
    cv2.fillPoly(mask, [pts], 255)
    return mask


def summarize(values: list[float], label: str) -> None:
    if not values:
        print(f"  {label:14s} (no data)")
        return
    vs = sorted(values)
    n = len(vs)
    p10 = vs[int(n * 0.10)]
    p50 = vs[int(n * 0.50)]
    p90 = vs[int(n * 0.90)]
    print(
        f"  {label:14s} n={n:3d}  min={vs[0]:7.3f}  p10={p10:7.3f}  "
        f"median={p50:7.3f}  p90={p90:7.3f}  max={vs[-1]:7.3f}  mean={statistics.mean(vs):.3f}"
    )


def main() -> None:
    args = parse_args()
    data_dir = (ROOT / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON annotations found under {data_dir}")

    rows: list[dict] = []
    skipped = 0

    for jf in json_files:
        try:
            doc = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"WARN: cannot parse {jf.name}: {e}", file=sys.stderr)
            continue

        png = jf.with_suffix(".png")
        gray = None
        height = int(doc.get("imageHeight") or 0)
        width = int(doc.get("imageWidth") or 0)
        if png.exists():
            img = cv2.imread(str(png))
            if img is not None:
                if height <= 0 or width <= 0:
                    height, width = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if width <= 0 or height <= 0:
            skipped += 1
            continue

        for shape in doc.get("shapes", []):
            if shape.get("shape_type") != "polygon":
                skipped += 1
                continue
            label = shape.get("label", "")
            mask = rasterize_polygon(shape.get("points", []), width, height)
            if int(mask.sum()) < 50:
                skipped += 1
                continue
            desc = _shape_descriptors(mask)
            cls = _classify_shape(desc, gray=gray, mask=mask)

            # Density signal stats for tuning DENSE_CLUMP_* thresholds.
            contrast = dark_fraction = None
            has_core = False
            if gray is not None:
                vals = gray[mask > 0].astype(float)
                if vals.size >= 100:
                    p10 = float(np.percentile(vals, 10))
                    p50 = float(np.percentile(vals, 50))
                    p90 = float(np.percentile(vals, 90))
                    contrast = p90 - p10
                    dark_threshold = p50 - 0.6 * (p50 - p10)
                    dark_fraction = float((vals < dark_threshold).mean())
                has_core = _has_dense_core(gray, mask)

            rows.append({
                "file": jf.stem,
                "label": label,
                "mask_area": desc.get("mask_area"),
                "aspect": desc.get("aspect_ratio"),
                "extent": desc.get("extent"),
                "solidity": desc.get("solidity"),
                "contrast": contrast,
                "dark_fraction": dark_fraction,
                "has_dense_core": has_core,
                "shape_class": cls,
            })

    print(f"\n=== {len(rows)} polygons over {len(json_files)} files (skipped {skipped}) ===\n")

    # Overall distribution
    print("Overall descriptor distribution:")
    summarize([r["aspect"] for r in rows if r["aspect"]], "aspect_ratio")
    summarize([r["extent"] for r in rows if r["extent"]], "extent")
    summarize([r["solidity"] for r in rows if r["solidity"]], "solidity")
    summarize([float(r["mask_area"]) for r in rows if r["mask_area"]], "mask_area")
    summarize([r["contrast"] for r in rows if r["contrast"] is not None], "contrast")
    summarize([r["dark_fraction"] for r in rows if r["dark_fraction"] is not None], "dark_fraction")
    n_core = sum(1 for r in rows if r["has_dense_core"])
    print(f"  has_dense_core  {n_core}/{len(rows)} polygons "
          f"({100.0 * n_core / max(len(rows), 1):.1f}%)")

    # Class distribution under current thresholds
    cls_counts = Counter(r["shape_class"] for r in rows)
    print("\nCurrent classifier output:")
    for cls, n in cls_counts.most_common():
        pct = 100.0 * n / max(len(rows), 1)
        print(f"  {cls:14s} {n:4d}  ({pct:5.1f}%)")

    # Per-class descriptor distribution (so we can see WHY each polygon was labeled)
    print("\nPer-class descriptor distribution:")
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["shape_class"]].append(r)
    for cls in ["straight_thin", "curved", "dense_clump", "down_clump", "ambiguous"]:
        rs = by_class.get(cls, [])
        if not rs:
            print(f"\n  [{cls}] no samples")
            continue
        print(f"\n  [{cls}]  n={len(rs)}")
        summarize([r["aspect"] for r in rs if r["aspect"]], "aspect_ratio")
        summarize([r["extent"] for r in rs if r["extent"]], "extent")
        summarize([r["solidity"] for r in rs if r["solidity"]], "solidity")
        summarize([float(r["mask_area"]) for r in rs if r["mask_area"]], "mask_area")
        summarize([r["contrast"] for r in rs if r["contrast"] is not None], "contrast")
        summarize([r["dark_fraction"] for r in rs if r["dark_fraction"] is not None], "dark_fraction")

    if args.csv:
        out = Path(args.csv)
        with out.open("w", encoding="utf-8") as fp:
            fp.write(
                "file,label,mask_area,aspect,extent,solidity,"
                "contrast,dark_fraction,has_dense_core,shape_class\n"
            )
            for r in rows:
                contrast = f"{r['contrast']:.2f}" if r["contrast"] is not None else ""
                df = f"{r['dark_fraction']:.4f}" if r["dark_fraction"] is not None else ""
                fp.write(
                    f"{r['file']},{r['label']},{r['mask_area']},"
                    f"{r['aspect']:.4f},{r['extent']:.4f},{r['solidity']:.4f},"
                    f"{contrast},{df},{int(r['has_dense_core'])},{r['shape_class']}\n"
                )
        print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
